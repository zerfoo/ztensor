# ztensor Development Log

## 2026-06-06: #111 -- capture-aware ArenaPool fallback (root-cause fix for the GB10 wedge)

**Type:** finding
**Tags:** cuda, gb10, #111, #106, capture, arena, fix

**Problem:** The GB10 crossasset training wedge (chased for weeks as #106) was
pinned in #111 to `ArenaPool.Alloc`'s exhaustion fallback issuing a synchronous
`cudaMalloc` during a graph-driven CUDA graph capture. The prior devlog entries
(below) exonerated the upload path; the hang is in the first captured forward
pass (gpuSum output alloc -> arena exhausted -> synchronous fallback malloc ->
GB10 driver hang).

**Root cause:** Two capture entry points. `compute.GPUEngine.BeginCapture()`
sets the pool capture-aware (cudaMallocAsync, safe). But `graph.(*Graph).Forward`
calls `cuda.StreamBeginCapture(g.stream)` directly (cuda_graph.go:406) and never
sets the pool's capture stream, so `MemPool.IsCapturing()` is false and the arena
exhaustion fallback (arena.go) does a plain synchronous `cudaMalloc` deep inside
the captured region. On GB10 (sm_121) that hangs the driver (the #93 class).

**Fix (commit 540d396):** Track active captures with a process-level *set* of
capturing stream handles in package `cuda` (capture_state.go) -- a set, not a
counter, so the watchdog force-end + normal end double-end is idempotent. Marked
on `StreamBeginCapture` success, cleared on `StreamEndCapture`. `ArenaPool.Alloc`
now refuses the synchronous fallback when `CaptureActive() && !fallback.IsCapturing()`,
returning the new `ErrCaptureUnsafeAlloc`. The engine-driven capture-aware
fallback (cudaMallocAsync) is preserved. CPU unit tests cover the registry
set-semantics (incl. double-unmark) and the arena guard's three cases.

**Caller audit (E7):** All ~130 `e.pool.Alloc` sites in compute/ check the error
via the Go idiom; none discard it. The pinned culprit `gpuSum`
(gpu_kernels.go:1019) falls back to CPU on alloc error, so the refusal degrades
gracefully instead of hanging. The graph layer (cuda_graph.go:454) treats a
propagated captured-instruction error as a clean capture failure: it ends
capture (now also clearing the capture flag), restores KV-cache state, and
re-runs the region uncaptured. Caveat: a CPU fallback that fires *mid-capture*
(rather than propagating) can leave a hole in the captured graph; this is why the
perf-correct partner is arena sizing.

**Arena sizing (E8) already shipped:** `ZERFOO_ARENA_SIZE_GB` (compute/gpu_engine.go,
default 2 GB, bounds [1,128], validated, tested in compute/arena_size_test.go)
already exists. The crossasset training working set exceeded 2 GB, exhausting the
arena mid-capture. Operational mitigation: raise `ZERFOO_ARENA_SIZE_GB` for
crossasset training so the fallback never fires during capture -- the guard is
the no-hang backstop, correct sizing keeps the captured graph intact.

**Status:** CPU build/vet/lint/test green (27 packages, -race clean on cuda+graph).
GB10 hardware validation pending (E10): a graph-capture + exhausted-arena repro
via Spark to confirm no D-state, plus a one-shot pre-fix re-wedge confirmation
(user-authorized, host-restart risk acknowledged).

## 2026-06-06: #106 -- context-replica does NOT wedge; suspect = CUDA graph capture

**Type:** finding
**Tags:** cuda, gb10, #106, capture, repro

**Problem:** With observability fixed (host-script + ssh, see below), determine
whether the production engine context reproduces the wedge.

**Method:** dstate-watchdog + TestWedge106Context run via a host-resident script
with SHORT pod args (avoids the Spark args-mangling that exit-2'd prior pods).
Full phase log read back over ssh from the hostPath.

**Result (fully observed, pod ztensor-issue106-ctx3):** ALL PHASES COMPLETE, NO
WEDGE, test PASS in 0.80s, watchdog saw 0 D-state threads. Sequence mirrored
CrossAsset: PHASE1 UploadWeights(74 model weights) -> PHASE2 UploadWeights(
213,354 sample+param tensors, shape [1,193]) -> PHASE3 200 MatMuls. PHASE2
uploaded in 70 ms as 54 bulk chunks. No hang anywhere.

**Conclusion:** Pure ztensor CANNOT reproduce #106 -- not the bare upload, not
with prior-upload + arena + direct compute context. The one major element the
replica does NOT exercise that the real run does: zerfoo's autograd graph driven
through **CUDA graph capture** (deploy has capture ENABLED -- ZERFOO_DISABLE_CUDA
_GRAPH unset). The replica's PHASE3 used direct engine.MatMul calls, NOT the
graph/cuda_graph.go capture wrapper. Prime remaining hypothesis: the wedge is in
the CUDA-graph capture path during the first training step AFTER the 213k
upload, with "UploadWeights never returns" being a misattribution (the T4.2
print is the last clean log line before the captured step hangs). Note prior
art: the capture-hang work in the original plan / ztensor#93.

**Cheap confirming test (for the operator):** re-run train-crossasset GPU with
ZERFOO_DISABLE_CUDA_GRAPH=1. If the wedge disappears, capture is confirmed the
culprit and UploadWeights/bulkUploadF32 is exonerated.

**400k confirm (pod ztensor-issue106-ctx4):** repeated the context replica at
400,000 samples (>300k historical scale). PHASE2 uploaded 400,050 tensors in
124 ms as 99 bulk chunks; ALL PHASES COMPLETE, PASS, watchdog 0 D-state. Even at
>300k, pure ztensor does not wedge. The upload path is conclusively exonerated;
the trigger is in the real training path the replica omits (CUDA graph capture).

**Reusable infra unblocked:** host-script pattern (script at
/var/lib/zerfoo/bench-out/*.sh, short pod args `bash <path>`) + ssh read of the
hostPath works reliably. Spark /logs,/exec,/delete still hang; Spark ALSO
mis-reports container Exited(2) as "completed exited successfully" -- never
trust Spark pod status alone; verify via `sudo podman ps -a` or the hostPath log.

## 2026-06-06: #106 -- context-replica inconclusive; Spark observability wall [SUPERSEDED -- observability fixed; see entry above; context-replica ran and did NOT wedge]

**Type:** finding
**Tags:** cuda, gb10, #106, infra, spark

**Problem:** Test whether the production engine CONTEXT (prior weight upload +
arena + first-compute) reproduces the wedge, since the bare upload does not.

**What happened:** Built TestWedge106Context (UploadWeights weights -> 213k
samples -> 200 MatMuls) and ran it as guarded pod ztensor-issue106-context. The
pod reported "completed exited successfully" but in only 4.5s of running time
(scheduled->running 0.5s = image cached; running->completed 4.5s). That is
impossible for a from-scratch build + 213k-tensor upload + compute (the
comparable wedge-guard pod took ~13 min). The result is therefore NOT a real
full run and is uninterpretable.

**Root cause (of the inconclusiveness):** This Spark host gives no usable
observability. Confirmed via ssh that the hostPath mount (/work/wedge-out ->
/var/lib/zerfoo/bench-out) delivers NOTHING to the host -- ctx.log/repro.log/
watchdog.log are absent; bench-out holds only pre-existing Apr-16 files. Spark
/logs, /exec, /delete all hang; pods cannot reach webhook.site; pod detail
exposes no container exit code. So the only signals are pod phase + timing, and
the timing now contradicts the completed=ran assumption.

**Fix:** N/A. Stopping pure-ztensor GB10 experiments on this host -- results are
not trustworthy without log access. The productive path is to run
dstate-watchdog.sh alongside the REAL train-crossasset run (wedge reliably
reproduces there; operator has working host access to read the captured frame).

**Impact:** The solid, exit-code-verified findings stand (bare f32 upload at
213k x {8KB,193-float} on a fresh engine does not wedge; #106 is not the upload
primitive / chunking caps). The context question remains open, blocked on
observability, not ideas.

## 2026-06-06: #106 -- pure-ztensor 213k f32 upload does NOT reproduce the wedge

**Type:** finding
**Tags:** cuda, bulk-upload, gb10, sm_121, #106, repro

**Problem:** Determine whether the GB10 wedge reproduces with ztensor alone
(no Wolf), to localize the root cause.

**Experiment:** TestWedge106Repro built 213,304 float32 tensors x 2048 elems
(~1.7 GiB total) and called GPUEngine.UploadWeights on the GB10, default
(non-managed) path, fresh engine, no concurrent GPU activity. Run via Spark pod
ztensor-issue106-wedge-guard. Because every exfil channel on this host is
unreliable (Spark exec/logs/delete hang; pods cannot reach webhook.site;
hostPath files did not land for ssh), the verdict was encoded in the POD EXIT
CODE: exit 0 only if the test logged "UploadWeights RETURNED ok"; 3 if it
skipped (no CUDA); a true wedge would leave the pod stuck "running" forever.

**Result:** Pod COMPLETED, exit 0 (ran 10:30:10 -> 10:42:59 PDT). The upload
ran on-device and returned cleanly. NO WEDGE. (An earlier non-guarded run of the
same test also completed in ~7 min.)

**Root cause:** Not isolated. The f32 bulk-upload path -- chunked or not -- at
213k-tensor scale does NOT by itself wedge the GB10 driver. The wedge therefore
requires a condition present in the production caller but absent here. Leading
suspects, in rough priority:
  1. Non-f32 tensors in the weight set: any tensor that is not plain f32 skips
     bulkUploadF32 and takes the PER-TENSOR Malloc loop in UploadWeights
     (gpu_engine.go:520+). At 10^5 scale that is the original #103 per-tensor
     wedge, which bulk only ever fixed for f32. Worth confirming the production
     weight dtypes.
  2. Managed-memory path (ZERFOO_ENABLE_MANAGED_MEM set) -- this run was
     non-managed (mallocManaged untested here at scale).
  3. Concurrent GPU state (prior allocations, active stream/graph capture)
     during the upload, vs a fresh idle engine here.
  4. Larger total bytes / different per-tensor size mix than the uniform
     2048-elem tensors used here.

**Impact:** Redirects the investigation away from "bulkUploadF32 byte/tensor
caps" and toward the per-tensor path and/or upload context. Harness:
docs/bench/manifests/issue-106-wedge-repro.yaml, TestWedge106Repro (branch
fix/issue-106-wedge-repro).

**Follow-ups (same day):**
- Inspected the production caller (crossasset.go:513-529): the 213,304 items are
  ALL f32 -- tiny sample-data tensors of shape [1,fps] (one per sample x source)
  plus ~50 graph params. The non-f32 per-tensor-path hypothesis is REJECTED.
- Deploy (train-crossasset-gpu.yaml) sets no managed memory, no arena override,
  CUDA graph ENABLED (default).
- Ran a 2nd guarded variant: 213,304 tensors x 193 f32 (~165 MB, matching the
  tiny production tensor size) on a fresh engine. Pod COMPLETED exit 0 -- again
  NO WEDGE. Tensor size is also ruled out.
- CONCLUSION: the f32 upload path does not wedge in isolation at production
  count/dtype/size on a fresh engine. The trigger is the production ENGINE
  CONTEXT that the fresh-engine repro lacks: a prior weight upload + arena
  allocation + training-graph/capture setup preceding the 213k upload -- OR the
  hang is actually after UploadWeights (first kernel/capture) with the upload
  merely the last log line. Surest next step: run dstate-watchdog.sh alongside
  the REAL train-crossasset GPU run (which reliably wedges) to pin the ioctl in
  the true failing context, rather than more fresh-engine pure-ztensor variants.

## 2026-06-06: #106 REOPENED -- chunking is NOT the fix

**Type:** finding
**Tags:** cuda, bulk-upload, gb10, sm_121, #106, correction

**Problem:** Correct the record. The 2026-06-05 entry below claimed the chunked
`bulkUploadF32` (v1.8.1) "validated end-to-end" and "unblocked" the CrossAsset
213k-tensor upload. That is WRONG.

**Root cause of the wrong claim:** `TestGPUEngine_UploadWeights_MultiChunk` used
only 256 tensors / 256 MiB. It proved a 64 MiB chunk does not wedge at *small
scale* -- it never reproduced the 213,304-tensor regime the issue is actually
about. Passing that test said nothing about the real wedge.

**What actually happened (per issue #106 reopen, user dndungu):** Wolf
train-crossasset was rebuilt against the merged chunking code (verified in the
binary, no vendoring) and the matched repro (213,304-tensor pre-upload on GB10)
**wedged identically** -- exec/logs/ssh+logind all hang, control plane
responsive: the same uninterruptible D-state CUDA-driver wedge. So capping each
alloc/copy at 64 MiB / 4096 tensors is a defensive bound, NOT the fix. The
wedge does not correlate with single-alloc size.

**Fix:** None yet. The exact wedging CUDA ioctl was never pinned because the
D-state blocks every in-container capture path. Next step (user-proposed):
out-of-band watchdog that samples the D-state thread's
`/proc/<tid>/{wchan,syscall,stack,status}` to a persisted hostPath that survives
the data-plane wedge, to name the exact ioctl, before proposing a real fix.

**Impact:** v1.8.1 chunking stays as a defensive bound (no regression), but
#106 is OPEN. The "fixes #106" framing on PR #107 was premature; treat the
chunking as a partial mitigation only.

## 2026-06-05: bulkUploadF32 chunking validated on GB10 (#106) [SUPERSEDED -- see 2026-06-06 above; chunking did NOT fix the wedge]

**Type:** benchmark
**Tags:** cuda, bulk-upload, gb10, sm_121, #106, verification

**Problem:** Confirm on real GB10 hardware that the chunked `bulkUploadF32`
(64 MiB byte cap + 4096 tensor cap) does not wedge the driver and that
cross-chunk `GPUStorage` views round-trip, before merging the #106 fix.

**Root cause:** N/A (verification). The prior single unbounded
`Malloc(total)`+`Memcpy(total)` wedged the GB10 (sm_121) driver in
uninterruptible D-state at multi-GB scale; chunking bounds every driver call.

**Fix:** Added `TestGPUEngine_UploadWeights_MultiChunk` (uploads 256 MiB as
256x1 MiB tensors -> 4 real 64 MiB chunks; asserts `len(bulkUploadBuffers)==4`
and round-trips head/tail sentinels across every chunk boundary). Ran on GB10
via Spark with an exit-code guard that fails the pod unless the GPU test
actually PASSED (a CUDA-unavailable SKIP is treated as failure), because Spark
drops container stdout for completed pods.

**Impact:** #106 fix validated end-to-end. The chunked path uploads multi-GB
weight sets as bounded 64 MiB driver calls with no wedge. Wolf CrossAsset
213k-tensor pre-upload is unblocked.

**Evidence:**
- Pod: `ztensor-issue106-multichunk-guard-3c04539` (exit 0 = guard passed =
  `--- PASS: TestGPUEngine_UploadWeights_MultiChunk` on GB10, no SKIP).
- Ran 2026-06-05 22:07:27 -> 22:14:32 PDT on DGX Spark GB10 (Spark v1.13.1).
- Commit: 3c04539 (branch fix/bulk-upload-chunking-106), image
  golang:1.26-bookworm + GOTOOLCHAIN=auto, libkernels.so mounted from host.
- Manifest: docs/bench/manifests/issue-106-multichunk.yaml.
- Spark gotcha reconfirmed on v1.13.1: container stdout/logs are dropped for
  completed pods; the /logs endpoint hangs. Encode correctness in the pod exit
  code, not in retrievable logs. Cold arm64 image pull ~20 min (not cached
  between runs).

## 2026-06-05: CUDA graph capture-hang plan closed; bulk-upload wedge opened (#106)

**Type:** plan-trim
**Tags:** cuda, capture, gb10, bulk-upload, e2, planning

**What happened:** The GB10 CUDA-graph-capture-hang work tracked by the prior
docs/plan.md shipped in release 1.8.0 (PRs #94 wave-1 probes, #95 repro harness,
#96 WithCapture + watchdog, #97 capture-aware alloc + workspace pre-alloc, #98
LMHead non-capturable). That plan is now retired and docs/plan.md is replaced by
the issue #106 plan. The capture-hang root cause and fix are already recorded in
this devlog (2026-04-16 entries) and ADR precedent zerfoo 088; the stable
interface knowledge stays in design.md. No new ADR was needed for the retirement.

**New issue:** #106 reports the bulk-upload fast path from #103 itself wedges the
GB10 driver in uninterruptible D-state on large one-shot uploads (~213k float32
tensors, single multi-GB `Malloc`+`Memcpy`). Root cause: `bulkUploadF32` has no
upper bound on the consolidated buffer size. Fix is to chunk by a byte cap; see
docs/adr/003-bulk-upload-chunking-cap.md and docs/plan.md.

**Spark operational gotchas (carried forward, still valid):**
- Spark drops multi-element `command`; use `args: ["bash","-c", ...]`, no `command`.
- Spark truncates long `args[i]`; put scripts on host and mount them.
- Spark drops container stdout/stderr; redirect to a host file inside the script.
- ztensor `-tags cuda` is unmaintained; default build is the purego GPU path.
- Mount prebuilt `/opt/zerfoo/lib/libkernels.so` into any GPU test pod.

## 2026-04-16: T1.4 CUDA graph GB10 repro — capture PASSES on pre-upload workload

**Type:** investigation
**Tags:** cuda, capture, gb10, e1

**Problem:** Needed hardware evidence for whether TestCUDAGraph_MultiTensorUpload_GB10
(50 float32 tensors incl. 256x1024, then BeginCapture→MatMul→EndCapture) reproduces the
silent hang on GB10.

**Root cause:** The test uploads all weights BEFORE entering capture, which is the correct
ordering. The hang in production (Wolf CrossAsset) occurs when `graph/cuda_graph.go` calls
`cuda.StreamBeginCapture` without routing through `GPUEngine.BeginCapture` — causing lazy
allocations to run DURING capture on the managed-memory path. The E1 repro test does not
trigger this because `UploadWeights` completes before capture starts.

**Fix:** N/A — this confirms E1 probes work and the hang requires E2's fix (capture-aware
allocation routing in `graph/cuda_graph.go`). The `ensureNotCapturing` guard in `allocWeight`
did NOT trip, confirming no allocations during capture for the tested flow.

**Evidence:**
- Pod: `ztensor-cuda-graph-gb10-20260416-084710`
- Commit: `9bf9723` (ztensor main, post-E1)
- DGX Spark GB10, CUDA 13.0.2, driver 580.142, golang:1.26-bookworm
- Result: `PASS: TestCUDAGraph_MultiTensorUpload_GB10 (0.51s)`
- Log line: `capture completed cleanly in phase=EndCapture; fix is in place`

**Impact:** E2 (Wave 4) remains necessary to fix the production hang. The test will serve
as a regression gate once E2 lands — it must continue to PASS.

---

## 2026-04-09: Issue #79 not reproducible at ztensor primitive level

**Type:** investigation
**Tags:** gpu, issue-79, patchtst, dgx-gb10

**Problem:** zerfoo PatchTST GPU training freezes at deterministic loss
0.268357 on DGX GB10. Issue #79 hypothesized the fault lies in ztensor's
GPU engine dst-output routing (`makeGPUResult` / `SetStorage` /
`GPUStorage.Slice()`). Four hypotheses (alpha/beta/gamma/delta) were
logged in the issue.

**Investigation:** Added `TestGPUEngine_PatchTSTBackward_DstRoundTrip`
(compute/gpu_dst_roundtrip_test.go) porting the exact op sequence from
`zerfoo/timeseries/patchtst_gpu_train.go:1022-1031`:
Transpose -> Zero -> MatMul(patchesT, dX, dPEW) -> in-place Add
accumulate into pre-seeded gradW -> gradW.Data() read. Ran on DGX GB10
via Spark pod `ztensor-issue79-repro-1775759440` (manifest at
`docs/bench/manifests/issue-79-repro.yaml`, commit 3e538e6 of
`fix/issue-79-matmul-accumulate-repro`).

Full test suite on DGX:
```
TestGPUEngine_Add_DstRoundTrip_OutOfPlace        PASS
TestGPUEngine_Add_DstRoundTrip_InPlace           PASS
TestGPUEngine_Add_DstRoundTrip_RepeatedInPlace   PASS
TestGPUEngine_Add_DstRoundTrip_NoExplicitSync    PASS
TestGPUEngine_PatchTSTBackward_DstRoundTrip      PASS
```

**Root cause:** Not in ztensor primitives. The
`Transpose -> Zero -> MatMul -> in-place Add` chain with a pre-seeded
CPU-wrapper dst does NOT reproduce zero readback on small shapes
(totalRows=4, patchLen=3, dModel=2). None of the four hypotheses from
the issue body is triggered at this level.

**Fix:** N/A. Investigation narrows the search to factors the ztensor
test does not exercise:
1. Shape regime -- production PatchTST uses thousands of rows / dModel in
   the hundreds; bug may only manifest under larger allocations or
   specific arena pressure.
2. Interaction with `encoderBackward` and multi-op state carried across
   the full batch, not just the patch-embedding backward slice.
3. The CPU-loop posEmb update at `patchtst_gpu_train.go:1012-1019`
   interleaved with GPU ops on the same stream.
4. zerfoo-side gradTs wrapper rebuild logic affecting how `.Data()`
   resolves after many accumulations.

**Impact:** Rules out ztensor engine primitive routing as the direct
cause of the frozen-loss signature. Next debugging must happen
zerfoo-side with a large-shape reproducer that closer matches the real
training configuration, or by instrumenting `trainWindowedGPU` itself
rather than trying to lift primitives into ztensor tests.

## 2026-03-29 -- v1.0.0 Benchmark Baseline

Pre-v1 benchmark baseline recorded on Apple M4 (darwin/arm64, 10 cores).

### tensor/

```
BenchmarkDequantizeAWQ-10        487695     2443 ns/op   6707.32 MB/s      0 B/op   0 allocs/op
BenchmarkDequantizeGPTQ-10       489553     2447 ns/op   6694.68 MB/s      0 B/op   0 allocs/op
BenchmarkDequantizeQ8-10         758720     1569 ns/op  10441.57 MB/s      0 B/op   0 allocs/op
BenchmarkDequantizeQ4-10         657984     1854 ns/op   8838.85 MB/s      0 B/op   0 allocs/op
BenchmarkQuantizeW8A8-10         202429     5773 ns/op   2838.19 MB/s   4928 B/op   2 allocs/op
BenchmarkDequantizeW8A8-10       750740     1606 ns/op  10201.41 MB/s      0 B/op   0 allocs/op
BenchmarkGemmW8A8-10                 16 66901216 ns/op               67125584 B/op   2 allocs/op
```

### compute/

```
BenchmarkCPUEngineMatMul/64x64x64-10              36901      32517 ns/op     16568 B/op    8 allocs/op
BenchmarkCPUEngineMatMul/128x128x128-10             6021     196675 ns/op     65720 B/op    8 allocs/op
BenchmarkCPUEngineAdd-10                            3769     371526 ns/op   4194458 B/op    6 allocs/op
BenchmarkCPUEngineMul-10                            3403     364828 ns/op   4194456 B/op    6 allocs/op
BenchmarkCPUEngineDiv-10                            3507     377398 ns/op   4194456 B/op    6 allocs/op
BenchmarkCPUEngineTranspose-10                       513    2160868 ns/op   4194593 B/op   10 allocs/op
BenchmarkCPUEngineSum-10                             988    1763374 ns/op      4456 B/op   10 allocs/op
BenchmarkBinaryOpSameShape-10                     184129       5682 ns/op      8344 B/op    6 allocs/op
BenchmarkBinaryOpBroadcast-10                      54372      25177 ns/op      8704 B/op   15 allocs/op
BenchmarkCPUEngineSoftmax-10                        1701     742775 ns/op   1048732 B/op    6 allocs/op
BenchmarkPowSquare-10                              83407      14530 ns/op      8448 B/op    8 allocs/op
BenchmarkPowGeneric-10                             17540      68821 ns/op      8696 B/op   15 allocs/op
BenchmarkMulScalarF32-10                          229659       5335 ns/op      8352 B/op    6 allocs/op
BenchmarkAddScalarF32-10                          257928       5093 ns/op      8352 B/op    6 allocs/op
BenchmarkDivScalarF32-10                          270468       4804 ns/op      8328 B/op    5 allocs/op
BenchmarkQ5KGEMVvsDequantReQuant-10                  117   10247756 ns/op     17868 B/op   30 allocs/op
BenchmarkQ6KGEMVvsDequantReQuant-10                  236    4881708 ns/op     17864 B/op   30 allocs/op
BenchmarkQ4KvsQ4_0GEMV/Q4_0-10                      1867     644002 ns/op     17864 B/op   30 allocs/op
BenchmarkQ4KvsQ4_0GEMV/Q4_K-10                       295    4109990 ns/op     18200 B/op   30 allocs/op
BenchmarkFusedRMSNorm/fused/1x128x1152-10          10000     101968 ns/op    590640 B/op   10 allocs/op
BenchmarkFusedRMSNorm/unfused/1x128x1152-10         1082    1082288 ns/op   3150078 B/op   83 allocs/op
BenchmarkFusedRMSNorm/fused/1x256x2048-10           3714     312384 ns/op   2098480 B/op   10 allocs/op
BenchmarkFusedRMSNorm/unfused/1x256x2048-10           472    2569119 ns/op   6298199 B/op  109 allocs/op
BenchmarkFusedRoPE/fused/1x128x256-10              41821      24126 ns/op    131240 B/op    6 allocs/op
BenchmarkFusedRoPE/unfused/1x128x256-10              838    1464438 ns/op   2370572 B/op  66139 allocs/op
BenchmarkFusedRoPE/fused/4x64x128-10              49312      24735 ns/op    131240 B/op    6 allocs/op
BenchmarkFusedRoPE/unfused/4x64x128-10               834    1434549 ns/op   2378861 B/op  66663 allocs/op
BenchmarkFusedSiLUGate/fused/1x128x1152-10         9127     128377 ns/op    590000 B/op    6 allocs/op
BenchmarkFusedSiLUGate/unfused/1x128x1152-10        1674     677999 ns/op   3146559 B/op   27 allocs/op
BenchmarkFusedSiLUGate/fused/1x256x2048-10          2968     366888 ns/op   2097328 B/op    6 allocs/op
BenchmarkFusedSiLUGate/unfused/1x256x2048-10          987    1228453 ns/op   6292712 B/op   40 allocs/op
BenchmarkMatMul_CPU_128-10                          5892     181247 ns/op     65720 B/op    8 allocs/op
BenchmarkMatMul_CPU_512-10                           140    8632963 ns/op   1048760 B/op    8 allocs/op
BenchmarkMatMul_CPU_1024-10                           16   66232242 ns/op   4194488 B/op    8 allocs/op
BenchmarkSoftmax_CPU-10                              469    2650127 ns/op  16777384 B/op    6 allocs/op
BenchmarkTensorPool_AcquireRelease-10              17619      65503 ns/op       176 B/op   13 allocs/op
BenchmarkTensorNew_Baseline-10                      3832     312565 ns/op   1048712 B/op    4 allocs/op
BenchmarkTensorArena_GetPut-10                  12423249        113.7 ns/op       0 B/op    0 allocs/op
BenchmarkTernaryGEMV/ternary_256x256-10            34976      34179 ns/op      1024 B/op    1 allocs/op
BenchmarkTernaryGEMV/dense_f32_256x256-10          24955      48088 ns/op      1024 B/op    1 allocs/op
BenchmarkTernaryGEMV/ternary_1024x1024-10           1506     796753 ns/op      4096 B/op    1 allocs/op
BenchmarkTernaryGEMV/dense_f32_1024x1024-10         1340     895470 ns/op      4096 B/op    1 allocs/op
BenchmarkTernaryGEMV/ternary_4096x4096-10             87   13621750 ns/op     16384 B/op    1 allocs/op
BenchmarkTernaryGEMV/dense_f32_4096x4096-10           81   14770621 ns/op     16384 B/op    1 allocs/op
```

### Key Observations

- **Fused ops deliver large speedups**: FusedRMSNorm is 8-10x faster than unfused; FusedRoPE is 58-60x faster; FusedSiLUGate is 3-5x faster.
- **Zero-alloc dequantization**: All dequantize paths (AWQ, GPTQ, Q8, Q4, W8A8) are allocation-free at 6.7-10.4 GB/s.
- **Tensor arena**: 113.7 ns/op with zero allocations for get/put cycle.
- **Ternary GEMV**: 1.1-1.4x faster than dense float32 GEMV at matching sizes.

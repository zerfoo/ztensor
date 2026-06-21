# Plan: Fix ztensor #167 — capture-replay reuses save-for-backward arena memory (GB10)

## ✅ DONE + VALIDATED + MERGED (2026-06-21)
Fix landed in **zerfoo PR #910** (NOT ztensor — the fix is consumer-side in the
training capture runner; ztensor already exposes `SetResetFloor`/`ArenaUsedBytes`).
`CaptureReplayRunner` now raises the arena reset floor to the capture high-water
after `EndCapture`, reserving the captured graph's footprint for its replay
lifetime — mirroring the inference decode path (`generate/generator.go onCaptured`).
**GB10-validated:** Wolf CrossAsset dropout=0 GPU capture-on recovered fold-0
**0.6047 → 0.7257** (== CPU baseline), loss → 0.000889, capture engaged
(captures=1 replays=2559, ~12.5k samp/s). This unblocked the dropout sweep →
production model re-pinned to dropout=0.4 (wf_mean 0.7833; wolf PR #275). ADR 007
records the decision. Issue #167 closeable once it propagates (zerfoo release 1.55.1
pending org GHA minutes). Repro evidence posted to #167.

## Context

GPU capture-on training on the GB10 (sm_121, CUDA 13) produces a fast WRONG
answer: gradients collapse toward zero after the first step, loss plateaus, and
Wolf CrossAsset fold-0 accuracy lands at 0.6047 instead of the CPU baseline
0.7257 at a byte-identical config (`-batch-vectorize -batch-size=256
-precision=f32 -seed=42 -qk-norm`, 40 epochs, 2 folds, 24613 samples). Zero
NaN; `captures=1 replays=2559`. This was first seen 2026-06-18 (Wolf devlog
T-PT.0) and reconfirmed 2026-06-21 on ztensor v1.19.0. It is DROPOUT-INDEPENDENT
(reproduces at dropout=0) and is the reason the production `.zcam` is CPU-trained
even though ADR 077 prefers GPU.

Root cause (confirmed by code reading, see Discovery Summary): a training step's
graph is captured ONCE and replayed ~2559 times across all epochs. The captured
graph executable freezes the device addresses of every buffer its kernels touch
(GraphInstantiate). ADR 006 save-for-backward pins protect intermediates only
between a node's Forward and Backward and are released the moment that node's
Backward completes (graph.releaseSaved). During the single capture step the Go
Forward/Backward runs once, pins, runs Backward, then UNPINS. The captured
graph's kernels keep reading those exact addresses on every replay, but after
capture the addresses are unpinned, so per-epoch `ResetPool()` and intra-pass
free-list reuse reissue them to unrelated tensors -> the replayed graph reads
contaminated memory -> gradient collapse.

The existing `onCaptured` callback only records the plan's instruction OUTPUT
slots (`capturedSlots`) for restore-on-replay; it does not reserve the transient
arena memory the captured graph relies on (save-for-backward intermediates,
backward scratch). `SetResetFloor` exists but is never raised to the capture
high-water.

Fix (ADR 007): on capture completion, raise the arena reset floor to the arena's
current offset (capture high-water), reserving the captured graph's full
footprint for the graph's lifetime so no address it uses is ever reissued. This
is correct because the graph replays the identical kernel sequence on identical
addresses every step and no allocation occurs during the replay phase (inputs
are staged by copying into pre-existing stable tensors).

This plan is general-purpose ztensor framework work: any capture-replay training
workload benefits, not just Wolf. It is validated against the ADR 006 Wolf-hazard
contract test plus a new capture-specific regression test plus the live GB10
CrossAsset run.

## Discovery Summary (verified file:line references)

Arena allocator (internal/cuda/arena.go):
- `ArenaPool`: 256-byte-aligned bump allocator + sorted free-list (struct lines
  ~30-57). `Alloc` reuses free-list best-fit blocks (lines ~226-338, reuse at
  ~249-261) before bumping `offset`.
- `Reset()` (lines ~527-573): rewinds `offset` to `floor` (raised to highest pin
  end if pins exist, lines ~536-539), clears the free-list (line ~555), bumps an
  epoch counter. `Reset` already skips pinned spans.
- `SetResetFloor(floor int)` (lines ~613-620) sets the minimum rewind offset
  ("protects captured graph buffers"); `resetFloor` field at line ~50.
- Getters present: `Capacity()` (~687), `Stats()` (~653). NO `CurrentOffset()`
  getter yet — must be added.
- Capture-unsafe alloc guard: `if CaptureActive() && !a.fallback.IsCapturing()
  { return ErrCaptureUnsafeAlloc }` (lines ~303-305).
- Poison mode `ZTENSOR_ARENA_POISON=1` fills reclaimed spans with NaN
  (internal/cuda/arena_poison.go) — primary debugging lever.

Save-for-backward (graph/save_for_backward.go, ADR 006):
- `Saver`/`SaverAware`/`nodeSaver` (~50-75). Saved tensors stored per node in
  `savedSets.sets map[Node][]savedEntry`; each entry holds a tensor pointer +
  `pinned` handle (NO data copy — arena pointer).
- `SaveForBackward` pins arena-backed storage (`PinForBackward`,
  tensor/storage_pin.go ~36-53). `releaseSaved(node)` UNPINS immediately after
  the node's Backward (graph/graph.go ~329); `releaseAllSaved` at next Forward /
  end of Backward.

Capture-replay (graph/cuda_graph.go, compute/gpu_engine.go, compute/engine.go):
- `NewCUDAGraphExecutor(plan, stream, warmups, onCaptured, snapshotCache)`
  (cuda_graph.go ~268). Warmup steps run eager; one step is captured; the rest
  replay (zerfoo training/capture_replay.go).
- On capture (cuda_graph.go ~497-507): records `capturedSlots` (instruction
  output slots only), then calls `onCaptured()`, then first `GraphLaunch`.
- `EndCapture` -> `GraphInstantiate` bakes device addresses/kernel args
  (gpu_engine.go ~1197-1219). `ReplayGraph` relaunches on the same addresses
  (~1246-1256); inputs updated by memcpy into pre-allocated buffers, no alloc.
- Wolf's per-epoch `ResetPool` workaround (consumer side,
  wolf internal/crossasset/crossasset.go ~985-1228) bounds growth (issue #118)
  but does NOT fix replay reuse — that is this plan's job.

Test harness available:
- `testing/parity/StressEngine` (host-backed arena, same Pin/Unpin lifetime as
  GPU) catches the corruption class in CI without a GPU.
- `graph/save_for_backward_test.go` `TestSaveForBackward_WolfHazard_*` resets the
  arena between forward and backward and asserts correct reads.
- `testing/parity/training_gpu_test.go` `TestTrainingLoop_WolfPattern_GPU` runs
  the GB10 loop with poison semantics.
- `testing/gradcheck/` finite-difference checker for node correctness.

## Scope and Deliverables

In scope:
1. Arena `CurrentOffset()` getter.
2. Raise the reset floor to capture high-water at capture completion (the fix).
3. A regression test that fails on `main` and passes after the fix: a captured
   step, an arena reset, then replays whose result stays correct (StressEngine
   in CI; GB10 variant guarded by the GPU build tag).
4. GB10 validation: Wolf CrossAsset GPU capture-on reproduces fold-0 ~0.7257.
5. Release ztensor; bump zerfoo + wolf; re-run the GPU dropout grid (the
   originally-intended sweep, now on a correct baseline).

Out of scope (separate follow-ups):
- True per-step dropout-mask variation under capture (device-side seed counter);
  tracked in the wolf engine-op dropout PR. With ADR 007 the captured dropout
  mask is fixed-per-capture but correct (non-divergent).
- The Wolf `captureSafeGraph` gate that disables capture for dropout>0
  (wolf gpu_train.go:187) — re-enable AFTER this fix lands (one-line, ready).

## Use Case

- UC-167: GPU capture-on training is numerically correct (matches CPU/eager) so
  the ~56x capture-replay path is usable for production training, not just a fast
  wrong answer. Status: BROKEN (this plan fixes it).

## Checkable Work Breakdown

### Phase 1 — Reproduce + instrument (ztensor)
- [ ] T1.1 Add a minimal CI regression test that reproduces #167 WITHOUT a GPU:
      using `testing/parity/StressEngine`, build a tiny 2-op graph that (a) saves
      an intermediate via SaveForBackward in forward, (b) captures the
      forward+backward step, (c) does `ResetPool()` + allocates a clobbering
      tensor over the freed region, (d) replays and asserts the output/gradient
      equals the eager reference. Confirm it FAILS on current main (with
      `ZTENSOR_ARENA_POISON=1` it should read NaN / wrong value).
      verifies: [UC-167]  kind: test
- [ ] T1.2 Reproduce on GB10 via Spark with `ZTENSOR_ARENA_POISON=1` on the Wolf
      CrossAsset GPU run (dropout=0, 40ep/2fold, cache) and capture the
      poison-NaN site / first corrupted step in logs as ground truth.
      verifies: [UC-167]  kind: infrastructure

### Phase 2 — The fix (ztensor)
- [ ] T2.1 Add `func (a *ArenaPool) CurrentOffset() int` (internal/cuda/arena.go)
      returning the live bump `offset` under lock. Unit-test it.
      verifies: [UC-167]  kind: code
- [ ] T2.2 Locate the `onCaptured` implementation wired into
      `NewCUDAGraphExecutor` (grep `NewCUDAGraphExecutor(` and `onCaptured`
      across compute/ and the training path). At capture completion — after
      EndCapture/GraphInstantiate, before/at the first GraphLaunch — call
      `arena.SetResetFloor(arena.CurrentOffset())` on the engine's arena pool
      (type-assert `e.pool.(*gpuapi.CUDAArenaPool)` as `ResetPool` does in
      gpu_engine.go ~410). Set it UNCONDITIONALLY on every capture so re-capture
      re-reserves correctly. Add a log line: `arena: reset floor raised to
      <offset> for captured-graph lifetime`.
      verifies: [UC-167]  kind: code
- [ ] T2.3 Guard: if the arena is in MemPool/stream-ordered fallback (not the
      bump arena), the floor concept does not apply — make the floor-raise a
      no-op there (the fallback never reuses live addresses the same way).
      Document the assumption in a comment referencing ADR 007.
      verifies: [UC-167]  kind: code
- [ ] T2.4 Make T1.1 PASS with the fix in place. Confirm the poison test no
      longer trips.
      verifies: [UC-167]  kind: test

### Phase 3 — Validate (ztensor quality gates + GB10)
- [ ] T3.1 `go test ./...` green in ztensor incl. testing/parity (StressEngine)
      and testing/gradcheck. Run the race detector on the arena tests.
      verifies: [UC-167]  kind: test
- [ ] T3.2 Lint/format: `gofmt`, `go vet`, golangci-lint clean on changed files.
      verifies: [infrastructure]  kind: lint
- [ ] T3.3 Rebuild `libkernels.so` is NOT required (no kernel change) — confirm
      the fix is pure Go (arena + capture wiring). If any kernel touched, rebuild
      sm_121 via the documented Spark pod and re-verify symbols.
      verifies: [infrastructure]  kind: infrastructure
- [ ] T3.4 GB10 acceptance: build a ztensor parity image (or the Wolf
      train-crossasset image bumped to the fixed ztensor) and run the Wolf
      CrossAsset GPU capture-on config (dropout=0, 40ep/2fold, cache). PASS =
      loss descends monotonically (epoch0 0.747 -> <0.01 by epoch ~30s of CPU
      trajectory) and fold-0 acc within tolerance of 0.7257 (>= 0.72). Capture
      stays engaged (`captures=1 replays=2559`, ~12k samp/s). Poll /events
      (not phase, lore L-0003/L-0023); memory limit per L-0005.
      verifies: [UC-167]  kind: infrastructure

### Phase 4 — Release + propagate + resume the sweep
- [ ] T4.1 PR the ztensor fix (branch -> CI -> rebase-merge, NOT squash). Include
      T1.1 + T2.* + ADR 007. Reference and close #167.
      verifies: [UC-167]  kind: infrastructure
- [ ] T4.2 Cut a ztensor release (release-please) containing the fix; note the
      version.
      verifies: [infrastructure]  kind: infrastructure
- [ ] T4.3 Bump zerfoo -> fixed ztensor; release zerfoo. Bump wolf -> fixed
      zerfoo/ztensor. Rebuild the sm_121 kernel lib only if a kernel changed
      (T3.3 says no).
      verifies: [infrastructure]  kind: infrastructure
- [ ] T4.4 Re-enable Wolf capture for engine-op dropout: wolf gpu_train.go:187
      `captureSafeGraph` -> `return ok && (cfg.DropoutRate<=0 ||
      cfg.DropoutEngineOp) && !cfg.FusedSDPA`; update
      capture_counters_test.go TestTrainCaptureCounters_DropoutIneligible to
      assert capture STAYS enabled when DropoutEngineOp is set. (Wolf-side; was
      held pending this fix.)
      verifies: [UC-167]  kind: code
- [ ] T4.5 Re-run the GPU dropout grid (dropout 0.0/0.1/0.2/0.3) on the corrected
      baseline (expect d00 ~0.7257). Rank vs the pinned 0.7257. ADR 078: label
      mode=tri, folds, fold-0-and-wf_mean. Winner -> Wolf BPB.4a measure ->
      BPB.4b re-pin (gated on M0).
      verifies: [infrastructure]  kind: infrastructure

## Risk Register

- R1 (medium): reserving the capture footprint for the graph lifetime increases
  steady-state arena residency by one step's high-water. Mitigation: it is tens
  of MB for CrossAsset, far under ZERFOO_ARENA_SIZE_GB; T3.4 confirms no OOM.
- R2 (medium): the floor-raise might not cover EVERY address the captured graph
  touches if some buffers are allocated above the offset via the fallback pool,
  not the bump arena. Mitigation: T1.2 poison run identifies the exact corrupted
  region; if it is fallback-pool memory, extend the fix to pin those too. T2.3
  documents the fallback assumption.
- R3 (low): re-capture leaking the old reservation. Mitigation: set the floor
  unconditionally at each capture (T2.2).
- R4 (medium): the bug might have a second contributor beyond reuse (e.g. the
  captured-slots restore path). Mitigation: T1.1/T1.2 must show the fix fully
  closes the gap (acc 0.7257), not just improves it; if a residual remains,
  bisect the captured-slots restore (cuda_graph.go ~529-583).
- R5 (low): GB10 driver/kernel skew mid-session (reference_dgx_driver_mismatch).
  Mitigation: confirm GPU healthy before T3.4.

## Operating Procedure
- ztensor changes: branch -> PR -> CI -> rebase-merge (org rule). General
  mechanisms only (ADR 007 is framework-level; holds for any capture workload).
- GB10 runs via Spark only (manifests, memory limits, poll /events). No
  interactive ssh for workloads; ssh only for git-sync/file-transfer of the build
  source (DGX wolf checkout is broken — rsync; ztensor src via git archive).
- Validate every claim with mode+config; cite the run.

## Progress Log

### 2026-06-21 (b) — FIX IMPLEMENTED
- Located the exact gap: the TRAINING capture path (zerfoo
  training/capture_replay.go CaptureReplayRunner) uses engine BeginCapture/
  EndCapture directly and NEVER raises the reset floor, while the INFERENCE
  decode path (zerfoo generate/generator.go onCaptured) ALREADY raises the floor
  to ArenaUsedBytes at capture completion and is correct. So the fix is a
  zerfoo (not ztensor) consumer-side change mirroring the proven generate pattern.
- T2 DONE: zerfoo PR #910 (fix/capture-replay-arena-reset-floor) — after
  EndCapture, `SetArenaResetFloor(ArenaUsedBytes())`. go build/vet/test ./training
  green + gofmt clean. (Capture-replay is GPU-only so full validation = GB10.)
- ztensor itself needs NO change (SetResetFloor/ArenaUsedBytes already exist);
  T2.1/T2.3 in this plan are obviated. T1.1 StressEngine CI test is NOT feasible
  (no CUDA graph on the host engine) -> validation is the GB10 run (T3.4).
- REMAINING: GB10 validate via wolf image built against zerfoo@fix-branch
  pseudo-version (expect fold-0 ~0.7257) -> merge #910 -> release zerfoo ->
  re-pin wolf -> re-enable wolf capture gate (T4.4) -> re-run dropout grid.

### 2026-06-21 (a) — Change Summary
- Created this plan and ADR 007 (capture-lifetime arena reservation) after
  root-causing #167: GPU capture-on training reuses save-for-backward arena
  memory across replays -> grad collapse -> fold-0 0.6047 vs 0.7257.
- Root cause localized via parallel code discovery of the ztensor arena
  allocator, the ADR 006 save-for-backward Saver, and the capture-replay runner.
  Fix = raise arena reset floor to capture high-water at capture completion.
- Upstream evidence posted to ztensor #167 (v1.19.0 still broken,
  dropout-independent).
- ADRs: docs/adr/007-capture-lifetime-arena-reservation.md (new). NOTE: fix
  landed in zerfoo (consumer), not ztensor; ADR 007 stays as the cross-repo
  arena-lifetime-under-capture decision of record.

## Hand-off Notes
- The fix is pure-Go ztensor (arena getter + one floor-raise at capture
  completion) per current analysis; no CUDA kernel change expected (T3.3).
- The Wolf-side capture gate re-enable (T4.4) is a ready one-liner held until
  this lands.
- The whole effort is an OPTIMIZATION track (Wolf ADR 076): it unblocks fast GPU
  training + the beat-baseline sweep, but does NOT block paper-trade M1.

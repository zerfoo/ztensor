# ztensor Work Plan

## Title

Stop the GB10 CUDA-graph-capture wedge at its root: make the ArenaPool
exhaustion fallback capture-aware (issue #111), and resolve issue #106 by
pointing it at #111 as the real cause.

## Context

### Problem statement

On the NVIDIA GB10 (sm_121, aarch64, unified memory), Wolf `train-crossasset
-gpu` freezes at the first forward pass. The container's main OS thread enters an
uninterruptible state inside a CUDA driver call that never returns; the pod
cannot be killed and leaks on the Spark orchestrator. For weeks this was chased
as issue #106 (`bulkUploadF32` wedging on large one-shot uploads), and a
defensive chunking bound shipped in v1.8.1 -- but the wedge reproduced
identically against the chunked code at the 213,304-tensor scale.

Issue #111 pinned the real cause with an out-of-band goroutine-1 capture:

```
goroutine 1 [syscall]:
 runtime.cgocall -> cuda.Malloc(0x30)        # 48-byte synchronous cudaMalloc, hung
 -> cuda.(*MemPool).Alloc        mempool.go:126
 -> cuda.(*ArenaPool).Alloc      arena.go:147   # arena exhausted -> fallback
 -> compute.(*GPUEngine).gpuSum  gpu_kernels.go:1019
 -> gpuReduceSum / ReduceSum
 -> zerfoo LayerNormalization.Forward
 -> graph.(*Graph).Forward       graph.go:221   # inside StreamBeginCapture
 -> training.ComputeGradients -> wolf crossasset.trainWithResult
```

`UploadWeights` / `bulkUploadF32` had already completed. The hang is in the
first forward pass: `gpuSum` allocates its output through `ArenaPool.Alloc`
(`e.pool.Alloc`, gpu_kernels.go:1019); the arena is exhausted, so it falls back
to `MemPool.Alloc -> cuda.Malloc` -- a fresh synchronous `cudaMalloc` issued
while a CUDA stream capture is active. On GB10 that hangs the driver (the #93
class). The "T4.2 pre-uploading N tensors" line was just the last thing printed
before the silent captured forward pass, which is why it was misattributed to
the upload.

### Why the existing capture guards did not catch it

There are two capture entry points:

1. `compute.GPUEngine.BeginCapture()` sets the pool into capture-aware mode
   (`SetCaptureStream`) before capture, so allocations route through the
   capture-safe `cudaMallocAsync`. Safe.
2. `graph.(*Graph).Forward` calls `cuda.StreamBeginCapture(g.stream)` directly
   (graph/cuda_graph.go:406). It does NOT route through `GPUEngine.BeginCapture`,
   so the engine pool's `captureStream` is never set. `MemPool.IsCapturing()`
   returns false, and the arena fallback (arena.go:147) does a plain synchronous
   `cudaMalloc` deep inside the captured forward pass.

The existing `IsCapturing()` guards (`bulkUploadF32` gpu_engine.go:415,
`allocWeight`/`uploadBytes` :855/:880) only protect the engine-driven path; they
are blind to graph-driven capture, and they do not cover the arena exhaustion
fallback at all.

### Objectives

- Guarantee that `ArenaPool.Alloc` never issues a synchronous `cudaMalloc` while
  a CUDA stream capture is active, regardless of which entry point began capture.
- Degrade gracefully on arena exhaustion during capture: callers fall back to
  CPU (gpuSum already does) or the capture fails cleanly and the graph layer
  re-runs uncaptured -- never a D-state hang.
- Make the perf-correct path available: size the arena so the fallback does not
  fire during the crossasset forward+backward working set.
- Verify on real GB10 hardware via Spark that a forward pass under graph capture
  with an exhausted arena no longer wedges.
- Merge, release, and resolve both #111 and #106.

### Non goals

- Re-architecting the `graph` <-> `compute` capture wiring. The fix does not
  require routing graph capture through `GPUEngine.BeginCapture`; the chosen
  mechanism is a capture-active signal in the `cuda` package (ADR 004).
- Changing the bulk-upload chunking shipped in v1.8.1 (#107). It stays as a
  defensive bound; it is unrelated to this hang.
- Eliminating CPU fallback as a performance concern beyond exposing/validating
  the arena sizing knob. Tuning the captured-graph working-set size to zero
  fallbacks for every model is follow-up perf work, not this fix.
- Reproducing the wedge by intentionally re-wedging the shared GB10 in D-state.
  Validation uses the fixed code path to prove the fallback no longer mallocs
  synchronously under capture; it does not require triggering a fresh hang.

### Constraints and assumptions

- GB10 (DGX Spark, 192.168.86.250:8080) is the only hardware where the wedge
  manifests. Local CPU tests cannot reproduce it. Hardware validation MUST go
  through Spark Pod submissions, never interactive `ssh` benchmarks (zerfoo
  CLAUDE.md; Spark gotchas in docs/devlog.md).
- ztensor stays CGO-free by default; CUDA access is via purego/dlopen in
  internal/cuda. `arena.go`, `mempool.go`, and `runtime_purego.go` are all in
  package `cuda`, so the capture-active counter + guard need no new cross-package
  surface.
- Unit tests run on CPU/CI without a GPU. The capture guard is testable with a
  fake fallback pool and a togglable capture-active signal (mirror
  compute/capture_alloc_test.go `fakeCapturePool`).
- main stays green for CPU and non-capture GPU tests on every commit.

### Success metrics

- A unit test proves `ArenaPool.Alloc` returns an error (and does NOT call the
  synchronous fallback `cudaMalloc`) when a capture is active and the fallback is
  not capture-aware; and still serves from the arena / capture-safe fallback when
  it is safe.
- The crossasset-style forward-pass-under-graph-capture repro with a
  deliberately undersized arena completes on GB10 via Spark with no D-state and
  no hang (alloc refused -> CPU fallback or clean uncaptured re-run).
- Existing compute/capture_integration_test.go, compute/capture_alloc_test.go,
  and bulk_upload_test.go stay green; `go test ./...` green on CPU.
- #111 closed on merge; #106 resolved with a comment pointing to #111; release
  tag cut.

## Discovery Summary

ENGINEERING. The root cause is pinned in #111 with a goroutine stack trace and
confirmed against current source. Two open issues at start: #106 (chased the
wedge as a bulk-upload bug; chunking shipped v1.8.1) and #111 (the actual cause:
arena fallback cudaMalloc during capture). They are one root cause; #111 is the
correctly-scoped issue.

Relevant code sites (verified 2026-06-06):

- internal/cuda/arena.go:147 -- `ArenaPool.Alloc` exhaustion fallback to
  `a.fallback.Alloc` (the unguarded synchronous-malloc path).
- internal/cuda/mempool.go:114-118 -- `MemPool.Alloc` uses `MallocAsync` only
  when `captureStream != nil`, else synchronous `Malloc`.
- internal/cuda/mempool.go:68-72 -- `MemPool.IsCapturing()`.
- internal/cuda/runtime_purego.go:371 (`StreamBeginCapture`), and the
  capture-status query `StreamCaptureStatus` at :415 (wraps
  cudaStreamGetCaptureInfo; `CaptureStatusActive = 1`).
- graph/cuda_graph.go:406 -- `cuda.StreamBeginCapture(g.stream)` direct capture
  start (the entry point that leaves the pool capture-unaware).
- compute/gpu_kernels.go:1019 -- `gpuSum` `e.pool.Alloc(e.deviceID, outByteSize)`
  with CPU fallback on error at :1020-1024.
- compute/gpu_engine.go:415 / :855 / :880 -- existing `IsCapturing()` guards
  (engine-driven only).
- compute/capture_alloc_test.go:27 -- `fakeCapturePool.IsCapturing()` test
  pattern to reuse.

Use case manifest: .claude/scratch/usecases-manifest.json (UC-106, UC-111).
Decision rationale: docs/adr/004-capture-aware-arena-fallback.md.

### Prior work shipped (do not redo)

The #106 bulk-upload chunking (E0-E3 of the previous plan) is complete and
released in v1.8.1 via PR #107: `bulkUploadF32` now uploads in bounded chunks
(64 MiB byte cap + 4096-tensor cap) via the pure `bulkUploadChunkRanges` tiler.
Decision in docs/adr/003-bulk-upload-chunking-cap.md; validation and the
context-replica exoneration findings (213k and 400k pure-ztensor uploads do NOT
wedge) are in docs/devlog.md. That chunking is a defensive bound, not the fix for
the wedge. The diagnostic epic that chased the pinned ioctl is now answered by
#111 and is retired from this plan into the devlog.

## Scope and Deliverables

In scope:
- A capture-active signal in package `cuda` and a capture-aware guard on the
  `ArenaPool` exhaustion fallback (the safety fix).
- Audit and, where needed, harden arena-alloc callers so a refused alloc during
  capture degrades gracefully (no propagated nil-deref, clean CPU fallback or
  capture failure).
- Arena sizing knob: expose/validate the `NewArenaPool` capacity / env knob so
  the fallback can be avoided for the crossasset working set (perf-correct).
- CPU unit tests for the guard; GB10 Spark validation of capture-with-exhausted-
  arena.
- ADR 004, devlog entry, PR to main (rebase-and-merge), release, resolve #111 and
  #106.

Out of scope: everything in Non goals.

| ID | Deliverable | Owner | Acceptance criteria |
|----|-------------|-------|---------------------|
| D1 | Capture-active signal in `cuda` | TBD | `CaptureActive()` true between begin and end (incl. watchdog force-end); no leak after capture |
| D2 | Capture-aware arena fallback | TBD | `ArenaPool.Alloc` returns `ErrCaptureUnsafeAlloc` instead of synchronous malloc when capture active and fallback not capture-aware; safe paths unchanged |
| D3 | Caller graceful-degrade audit | TBD | gpuSum + every `e.pool.Alloc` caller in a captured region handles alloc error without nil-deref; documented |
| D4 | Arena sizing knob | TBD | Capacity configurable; documented; default covers crossasset forward+backward or fallback degrades cleanly |
| D5 | CPU unit tests | TBD | Guard proven without a GPU; existing capture/bulk tests green; `go test ./...` green |
| D6 | GB10 validation | TBD | Capture + exhausted arena completes on GB10 via Spark; no D-state; devlog entry with pod + commit |
| D7 | Shipped + issues resolved | TBD | PR merged rebase-and-merge; release tag cut; #111 closed; #106 resolved with pointer to #111 |

## Checkable Work Breakdown

### E5 -- Capture-active signal in package cuda
**Component:** cuda
Acceptance: `cuda.CaptureActive()` returns true while any stream capture begun
via `StreamBeginCapture` is in flight, and false once every capture has ended
(normal end or watchdog force-end), with no counter leak.

- [x] T5.1 Add a package-level `atomic.Int32` capture counter to internal/cuda
  (e.g., in runtime_purego.go next to `StreamBeginCapture`). Increment on
  `StreamBeginCapture` success; expose `func CaptureActive() bool`. Decision
  rationale: docs/adr/004-capture-aware-arena-fallback.md  Owner: TBD  Est: 45m
  verifies: [UC-111]
  - Dependencies: none
  - Acceptance: counter increments only on a successful `cudaStreamBeginCapture`
    (not on the "not available" error path); `CaptureActive()` reflects it.
- [x] T5.2 Decrement the counter on every capture-exit path: `StreamEndCapture`
  success/failure and the capture-watchdog force-end (graph/cuda_graph.go
  captureWatchdog). Ensure begin/end are balanced even when capture is
  invalidated.  Owner: TBD  Est: 60m  verifies: [UC-111]
  - Dependencies: T5.1
  - Acceptance: after a begin+end pair (including a watchdog-forced end and an
    invalidated capture), `CaptureActive()` is false; no path leaves the counter
    > 0.
  - Risk: missing a decrement path leaves fallbacks suppressed after capture --
    enumerate all `StreamEndCapture` callers and the watchdog path explicitly.
- [x] T5.3 Unit test the counter: begin -> CaptureActive() true; end ->
  CaptureActive() false; balanced begin/end across simulated normal and
  force-end paths. Use the existing `Stream` test doubles where possible; gate
  any device-touching assertion behind the CUDA-available check.  Owner: TBD
  Est: 45m  verifies: [UC-111]
  - Dependencies: T5.2

### E6 -- Capture-aware ArenaPool fallback
**Component:** cuda
Acceptance: `ArenaPool.Alloc` never issues a synchronous `cudaMalloc` while a
capture is active and the fallback is not capture-aware; arena hits and
capture-safe fallbacks are unchanged.

- [x] T6.1 Define a sentinel error `ErrCaptureUnsafeAlloc` in internal/cuda and
  guard the exhaustion fallback in `ArenaPool.Alloc` (arena.go:147): before
  calling `a.fallback.Alloc`, if `CaptureActive() && !a.fallback.IsCapturing()`
  return `nil, ErrCaptureUnsafeAlloc`. Leave the free-list and bump-pointer fast
  paths untouched.  Owner: TBD  Est: 60m  verifies: [UC-111]
  - Dependencies: T5.1
  - Acceptance: arena-served allocs and capture-safe (`fallback.IsCapturing()`
    true) fallbacks behave exactly as before; only the unsafe synchronous path is
    refused.
- [x] T6.2 Unit test in internal/cuda using a fake fallback pool (records
  Alloc calls) + togglable capture-active state: prove `ArenaPool.Alloc` after
  exhaustion returns `ErrCaptureUnsafeAlloc` and does NOT call the fallback when
  capture is active and the fallback is not capture-aware; and DOES call the
  fallback when capture is inactive or the fallback is capture-aware. Mirror the
  compute/capture_alloc_test.go `fakeCapturePool` pattern.  Owner: TBD  Est: 75m
  verifies: [UC-111]
  - Dependencies: T6.1
  - Acceptance: test runs on CPU (no device); asserts zero synchronous-fallback
    calls in the unsafe-capture case.

### E7 -- Caller graceful-degrade audit
**Component:** compute
Acceptance: every code path that calls `e.pool.Alloc` (or otherwise allocates
arena memory) inside a captured forward/backward region tolerates an
`ErrCaptureUnsafeAlloc` without a nil-pointer deref, and either falls back to CPU
or surfaces an error that fails the capture cleanly.

- [x] T7.1 Enumerate arena-alloc call sites reachable from `graph.(*Graph).
  Forward` capture region (start from gpuSum gpu_kernels.go:1019; grep
  `e.pool.Alloc` / `pool.Alloc` in compute/). For each, confirm the error is
  handled (CPU fallback like gpuSum:1020, or returned up to the capture caller).
  Document the list in the devlog.  Owner: TBD  Est: 75m  verifies: [UC-111]
  - Dependencies: T6.1
  - Acceptance: a written list of call sites with their error-handling
    disposition; no site dereferences a nil pointer on alloc error.
- [x] T7.2 Fix any call site that does not handle the alloc error gracefully
  (add CPU fallback or proper error return). If all sites already handle it, mark
  done with the audit as evidence.  Owner: TBD  Est: 90m  verifies: [UC-111]
  - Dependencies: T7.1
- [x] T7.3 Confirm the graph-capture layer treats a failed captured instruction
  as a clean capture failure and re-runs uncaptured (graph/cuda_graph.go captureErr
  path), so a refused alloc during graph-driven capture cannot wedge or corrupt
  state. Add/extend a test if a gap exists.  Owner: TBD  Est: 60m  verifies: [UC-111]
  - Dependencies: T7.1

### E8 -- Arena sizing (perf-correct)
**Component:** cuda
Acceptance: the arena capacity is configurable and documented, so the fallback
can be sized out of the crossasset forward+backward working set; default
behavior is no-hang regardless.

- [x] T8.1 Trace where `NewArenaPool` capacity is set (callers of
  `NewArenaPool` / `SetDefaultArenaPool`) and whether an env knob
  (`ZERFOO_ARENA_SIZE_GB` or equivalent) already exists. Document the current
  sizing source.  Owner: TBD  Est: 45m  verifies: [UC-111]
  - Dependencies: none
- [x] T8.2 If no knob exists, add `ZERFOO_ARENA_SIZE_GB` (parsed where the arena
  is constructed) with a sane default; if one exists, document it. Keep the
  default conservative; the E6 guard is the safety net if the arena is too small.
  Owner: TBD  Est: 60m  verifies: [UC-111]
  - Dependencies: T8.1
- [x] T8.3 Unit test the sizing knob parse/default (env set, unset, malformed) on
  CPU.  Owner: TBD  Est: 30m  verifies: [UC-111]
  - Dependencies: T8.2

### E9 -- Lint, format, CPU test gate
**Component:** tooling
Acceptance: gofmt + `go vet ./...` clean; `go build ./...` and `go test ./...`
green on CPU.

- [x] T9.1 gofmt + `go vet ./...` clean on changed files; `go build ./...` exit
  0  Owner: TBD  Est: 20m  verifies: [infrastructure]
  - Dependencies: T6.2, T7.2, T8.3
- [x] T9.2 `go test ./...` green on CPU incl. existing capture_integration_test.go,
  capture_alloc_test.go, bulk_upload_test.go (GPU integration tests skip without
  CUDA)  Owner: TBD  Est: 20m  verifies: [infrastructure]
  - Dependencies: T9.1

### E10 -- GB10 validation via Spark
**Component:** compute
Acceptance: a forward pass under graph capture with a deliberately undersized
arena completes on GB10 with no D-state wedge (alloc refused -> CPU fallback or
clean uncaptured re-run), proving the synchronous-malloc-under-capture path is
gone.

- [ ] T10.1 Author a repro that exercises the graph-driven capture path
  (graph/cuda_graph.go) with an arena small enough to force exhaustion during a
  captured forward pass (e.g., a small `NewArenaPool` capacity + a LayerNorm/
  ReduceSum step like the pinned stack). Build an arm64 image; submit a Spark Pod
  manifest under docs/bench/manifests/. Do NOT ssh; use the Spark HTTP API.
  Owner: TBD  Est: 2h  verifies: [UC-111]
  - Dependencies: T9.2
  - Acceptance: with the fix, the pod completes (exit-code-guarded, since Spark
    drops stdout) and the out-of-band D-state watchdog records zero D-state
    threads.
- [ ] T10.2 Run the same repro built from the PRE-fix commit (or with the guard
  disabled) once, under the watchdog, to confirm the repro actually exercises the
  wedge path (user opted in 2026-06-06 to confirm the pre-fix wedge). Capture the
  pinned frame to disk via the out-of-band hostPath watchdog. Get a final
  go-ahead immediately before submitting this destructive pod, since it may leave
  an unkillable pod and need a host restart.  Owner: TBD  Est: 1h  verifies: [UC-111]
  - Dependencies: T10.1
  - Risk: intentionally wedging the shared GB10 can leave an unkillable pod /
    need a host restart -- confirm timing with the user right before running.
- [ ] T10.3 Devlog entry with pod name, commit, arena size, observed behavior
  (refusal + CPU fallback / clean re-run), watchdog D-state count, timing.
  Owner: TBD  Est: 20m  verifies: [infrastructure]
  - Dependencies: T10.1

### E11 -- Ship and resolve issues
**Component:** release
Acceptance: PR merged rebase-and-merge; release tag cut; #111 closed; #106
resolved with a pointer to #111.

- [ ] T11.1 Open the PR against main from `fix/issue-106-wedge-repro` (decided
  2026-06-06: continue on the investigation branch, carrying its devlog/manifest
  history + the #111 fix). Title the PR for #111 and reference #106.  Owner: TBD
  Est: 30m  verifies: [UC-111]
  - Dependencies: T9.2, T10.3
- [ ] T11.2 PR CI green; rebase-and-merge into main (not squash, not merge
  commit)  Owner: TBD  Est: 30m  verifies: [UC-111]
  - Dependencies: T11.1
- [ ] T11.3 release-please cuts the patch release; verify tag + GitHub release.
  ztensor is a library, so the released module version is the deployment
  artifact.  Owner: TBD  Est: 20m  verifies: [infrastructure]
  - Dependencies: T11.2
- [ ] T11.4 #111 closed on merge (link the PR). Post a comment on #106 stating
  the chunking defensive bound shipped in v1.8.1 and the real wedge fix is the
  capture-aware arena fallback in #111; close #106 (or confirm it auto-resolves)
  referencing #111.  Owner: TBD  Est: 20m  verifies: [UC-106, UC-111]
  - Dependencies: T11.3
- [ ] T11.5 Definition of done check: merged + released + verified live on GB10
  (T10) + reported honestly. If GB10 verification (E10) could not complete in
  session, state the specific blocker here and mark this task blocked rather than
  claiming done.  Owner: TBD  Est: 15m  verifies: [infrastructure]
  - Dependencies: T11.4

## Parallel Work

The capture-active signal (E5) is the root dependency for the guard (E6). The
caller audit (E7) and arena sizing (E8) can proceed in parallel once the guard's
sentinel error exists, and E8's discovery (T8.1) needs nothing. Validation (E10)
and ship (E11) are gated on green CPU tests (E9).

| Track | Tasks | Notes |
|-------|-------|-------|
| Track A: signal + guard | T5.1 -> T5.2 -> T5.3, then T6.1 -> T6.2 | Core chain; everything else waits on T6.1's sentinel error |
| Track B: caller audit | T7.1 -> T7.2 -> T7.3 | Starts after T6.1; independent of E8 |
| Track C: arena sizing | T8.1 (now) -> T8.2 -> T8.3 | T8.1 needs no deps; can start immediately |
| Track D: validation harness | author T10.1 repro + manifest during A/B | Code-free authoring overlaps; execution waits on T9.2 |

Sync points: T6.1 unblocks B; E9 (lint+test) needs E6+E7+E8 done; E10 needs E9;
E11 needs E10.

### Wave 1: Signal + sizing discovery (2 agents)
- [x] T5.1 Capture counter + `CaptureActive()`  verifies: [UC-111]
- [x] T8.1 Trace arena sizing source / existing knob  verifies: [UC-111]

### Wave 2: Balance the counter + guard + sizing knob (3 agents)
- [x] T5.2 Decrement on all capture-exit paths  verifies: [UC-111]
- [x] T6.1 Sentinel error + capture-aware fallback guard  verifies: [UC-111]
- [x] T8.2 Add/document `ZERFOO_ARENA_SIZE_GB` knob  verifies: [UC-111]

### Wave 3: Tests + caller audit (5 agents)
- [x] T5.3 Counter unit test  verifies: [UC-111]
- [x] T6.2 Guard unit test (fake fallback + capture toggle)  verifies: [UC-111]
- [x] T7.1 Enumerate arena-alloc call sites in capture region  verifies: [UC-111]
- [x] T8.3 Sizing knob parse/default test  verifies: [UC-111]
- [ ] T10.1 Author capture+exhausted-arena repro + Spark manifest  verifies: [UC-111]

### Wave 4: Audit fixes + capture-failure check (2 agents)
- [x] T7.2 Fix any ungraceful alloc-error call site  verifies: [UC-111]
- [x] T7.3 Confirm clean capture-failure / uncaptured re-run  verifies: [UC-111]

### Wave 5: Gate (1 agent, sequential)
- [x] T9.1 gofmt + vet + build  verifies: [infrastructure]
- [x] T9.2 `go test ./...` green on CPU  verifies: [infrastructure]

### Wave 6: GB10 validation (1 agent)
- [ ] T10.2 (optional, gated) pre-fix repro confirms wedge path  verifies: [UC-111]
- [ ] T10.3 Devlog entry for GB10 run  verifies: [infrastructure]

### Wave 7: Ship (1 agent, sequential)
- [ ] T11.1 Choose branch + open PR to main  verifies: [UC-111]
- [ ] T11.2 CI green + rebase-and-merge  verifies: [UC-111]
- [ ] T11.3 Release tag cut  verifies: [infrastructure]
- [ ] T11.4 Close #111; resolve #106 with pointer to #111  verifies: [UC-106, UC-111]
- [ ] T11.5 Definition-of-done honesty check  verifies: [infrastructure]

## Timeline and Milestones

| Milestone | Description | Member tasks | Exit criteria |
|-----------|-------------|--------------|---------------|
| M0 | Capture signal live | T5.1, T5.2, T5.3 | `CaptureActive()` balanced across all begin/end paths; unit green |
| M1 | Guard implemented | T6.1, T6.2 | Arena fallback refuses synchronous malloc under capture; CPU unit proves it |
| M2 | Callers safe + arena sizable | T7.1, T7.2, T7.3, T8.1, T8.2, T8.3 | No ungraceful alloc-error site; sizing knob documented; clean capture-failure path |
| M3 | CPU gate green | T9.1, T9.2 | gofmt/vet/build clean; `go test ./...` green |
| M4 | GB10 validated | T10.1, T10.3 | Capture + exhausted arena completes on GB10 via Spark; no D-state; devlog recorded |
| M5 | Shipped | T11.1-T11.5 | PR merged rebase-and-merge; release cut; #111 closed; #106 resolved |

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | A capture-exit path is missed, leaking the counter and suppressing fallbacks after capture ends | High | Medium | T5.2 enumerates every `StreamEndCapture` caller + watchdog path; T5.3 asserts balance across normal and force-end |
| R2 | A captured-region alloc caller does NOT handle the error and nil-derefs instead of degrading | High | Medium | T7.1 audit of all call sites; T7.2 fixes gaps; gpuSum already falls back to CPU |
| R3 | Refusing the fallback mid-capture invalidates the captured graph and corrupts state | Medium | Low | T7.3 confirms the graph layer treats a failed captured instruction as a clean capture failure and re-runs uncaptured (graph/cuda_graph.go captureErr) |
| R4 | Cannot safely reproduce the capture wedge on GB10 to validate (re-wedge risk on shared host) | Medium | Medium | Validate the FIXED path completes (no need to trigger a fresh hang); T10.2 pre-fix confirm is optional and gated on user go-ahead; fall back to #111 stack as pre-fix evidence and report honestly |
| R5 | Process-global capture counter is too coarse for some future multi-stream capture | Low | Low | Existing per-stream `StreamCaptureStatus` is available to refine if needed (ADR 004 notes this); current code captures one graph at a time |
| R6 | Branch confusion: fix landing on the investigation branch vs a fresh branch off main | Low | Medium | T11.1 decides explicitly at the ship gate; main is the merge target either way |

## Operating Procedure

- Definition of done (global CLAUDE.md): merged via rebase-and-merge, CI green,
  release tag cut, and the fix verified live on GB10 (the capture + exhausted-
  arena repro completing via Spark, observed in pod logs / watchdog, not merely
  "the code should not malloc"). Report what was actually observed; if GB10
  verification cannot complete, state the blocker (T11.5).
- Add tests with every implementation change (T5.3 with T5.2, T6.2 with T6.1,
  T8.3 with T8.2).
- Run gofmt, `go vet`, and the linter after code changes (T9.1).
- Never commit files from different directories in one commit (pre-commit hook
  rejects it). Keep commits small and logical: counter, guard, tests, sizing,
  each its own commit where practical. internal/cuda changes and compute changes
  go in separate commits.
- Validate GPU behavior only via Spark Pod submissions; never interactive ssh
  benchmarks on the DGX.

## Progress Log

### Change Summary -- 2026-06-06 (replan: #106 -> #111 root cause)

- Root cause converged: the GB10 wedge is `ArenaPool.Alloc`'s exhaustion
  fallback doing a synchronous `cudaMalloc` during graph-driven CUDA graph
  capture (arena.go:147), pinned in issue #111. The #106 bulk-upload chunking
  (v1.8.1, PR #107) is a defensive bound, not the fix.
- Trimmed the shipped #106 chunking epics (E0-E3) and the now-answered diagnostic
  epic (E4) from the plan; their knowledge lives in docs/adr/003 and
  docs/devlog.md (context-replica exoneration of the upload path at 213k/400k).
- Created docs/adr/004-capture-aware-arena-fallback.md: track active captures
  with a process-level counter in package `cuda`; guard `ArenaPool.Alloc` to
  refuse the synchronous fallback when `CaptureActive() && !fallback.IsCapturing()`;
  pair with arena sizing as the perf-correct path.
- New epics: E5 (capture-active signal), E6 (capture-aware arena fallback), E7
  (caller graceful-degrade audit), E8 (arena sizing knob), E9 (lint+CPU gate),
  E10 (GB10 Spark validation), E11 (ship + resolve #111 and #106).
- Use case manifest updated: UC-106 (upload chunking, shipped) and UC-111 (no
  synchronous malloc under capture).

ADRs created: docs/adr/004-capture-aware-arena-fallback.md -- capture-aware
ArenaPool exhaustion fallback via a process-level capture-active counter, paired
with arena sizing.

## Hand-off Notes

- The fix is almost entirely in package `cuda` (internal/cuda): a capture-active
  atomic counter set by `StreamBeginCapture` / cleared by `StreamEndCapture` +
  watchdog, and a guard in `ArenaPool.Alloc` (arena.go:147). Read ADR 004 first.
- The buggy path is graph-driven capture: `graph.(*Graph).Forward` calls
  `cuda.StreamBeginCapture(g.stream)` directly (cuda_graph.go:406) and never sets
  the pool's capture stream, so `MemPool.IsCapturing()` is false and the arena
  fallback mallocs synchronously. The engine-driven path
  (`GPUEngine.BeginCapture`) is already safe and must stay unchanged.
- `gpuSum` (compute/gpu_kernels.go:1019) already falls back to CPU on alloc
  error; the audit (E7) confirms peers do too.
- GB10 validation goes through Spark only (192.168.86.250:8080); exit-code-guard
  the pod because Spark drops stdout, and read phase logs back from a hostPath.
  Spark gotchas are in docs/devlog.md.
- The current branch `fix/issue-106-wedge-repro` holds the investigation devlog
  and Spark manifests (11 commits ahead of origin/main); decide at the ship gate
  whether to PR it whole or start a fresh branch off main (T11.1).
- Wolf caller that triggers the wedge: `internal/crossasset/crossasset.go`
  `trainWithResult -> training.ComputeGradients -> graph.Forward` under capture.

## Appendix

- Issues: github.com/zerfoo/ztensor#111 (root cause), #106 (chased it; chunking
  shipped v1.8.1).
- Decisions: docs/adr/004-capture-aware-arena-fallback.md (this fix),
  docs/adr/003-bulk-upload-chunking-cap.md (shipped defensive bound).
- Code: internal/cuda/arena.go:147 (`ArenaPool.Alloc` fallback),
  internal/cuda/mempool.go:114 (`MemPool.Alloc` sync vs async),
  internal/cuda/runtime_purego.go:371/:415 (`StreamBeginCapture`,
  `StreamCaptureStatus`), graph/cuda_graph.go:406 (direct capture start),
  compute/gpu_kernels.go:1019 (`gpuSum` arena alloc + CPU fallback),
  compute/capture_alloc_test.go (fake-pool test pattern).
- Pinned stack trace + fix options: issue #111 body.

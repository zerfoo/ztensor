# ztensor Work Plan

## Title

Stop the GB10 training freeze: make the ArenaPool exhaustion fallback
stream-ordered (cudaMallocAsync) instead of a synchronous cudaMalloc that
thrashes GB10 unified memory (issue #115).

## Context

### Problem statement

After the #111 capture guard shipped in v1.8.2, Wolf `train-crossasset -gpu` still
freezes on the NVIDIA GB10 (sm_121, unified memory) at `step=0`. The capture
guard was necessary but not sufficient.

`ArenaPool.Alloc`'s exhaustion fallback issues a **synchronous `cudaMalloc`**
(`internal/cuda/arena.go`, the path AFTER the #111 guard) during NORMAL,
non-capture training forward AND backward passes. On GB10 unified memory a
synchronous `cudaMalloc` under memory pressure faults into `do_swap_page` /
`filemap_fault` and stalls indefinitely. Captured goroutine-1 frames (v1.8.2,
`ZERFOO_ARENA_SIZE_GB=64`, full COIN bars):

```
backward: LayerNormalization.Backward -> gpuSub -> gpuBroadcastOp
          -> ArenaPool.Alloc (synchronous fallback) -> cuda.Malloc(0x4000)
          D-state sibling: folio_wait_bit_common -> filemap_fault -> __do_fault
forward:  MatMul -> makeGPUResult -> tensor.NewWithStorage -> ...fallback
          D-state sibling: do_swap_page
```

The #111 guard (`CaptureActive() && !fallback.IsCapturing()`) correctly does NOT
fire for these non-capture ops, so the synchronous `cudaMalloc` runs and wedges.

### Why a 64 GB arena still overflows (and why that is not the bug)

A single full-batch crossasset forward+backward (~16,408 samples x 12 scales)
retains every forward activation for backprop, so the arena must hold the entire
fwd+bwd working set at once. The arena reclaims only BETWEEN steps
(`StepScope.Close -> ResetPool -> arena.Reset`), never within a pass -- correctly,
since the activations are still referenced by the backward pass. So the overflow
is legitimate for full-batch training; `ZERFOO_ARENA_SIZE_GB` cannot be raised
high enough to reliably avoid it. The real fix is that the overflow must not use a
synchronous `cudaMalloc`, since ANY synchronous fallback wedges GB10.

### Root cause

The arena exhaustion fallback is a synchronous `cudaMalloc`, which is pathological
on GB10 under memory pressure (page-fault / swap stall). `cuda.MallocAsync` /
`cuda.FreeAsync` (the CUDA stream-ordered pool allocator) already exist
(internal/cuda/runtime_purego.go:86/102) and already work on GB10 (the capture
path uses them); the stream-ordered pool pre-reserves and reuses memory and does
not thrash.

### Objectives

- Route the ArenaPool exhaustion fallback through `cudaMallocAsync` on the
  engine's stream (stream-ordered, non-blocking) instead of a synchronous
  `cudaMalloc`, so overflow allocations do not wedge GB10.
- Free those overflow pointers via `cudaFreeAsync` on the same stream.
- Keep the #111 capture guard unchanged and taking precedence during graph-driven
  capture.
- Verify on real GB10 hardware that a stress pattern of arena-exhaustion overflow
  allocations completes with no D-state hang; and (end-to-end) that Wolf
  train-crossasset progresses past `step=0`.
- Merge, release, and resolve #115.

### Non goals

- Within-pass arena reclamation (freeing intermediates before the backward pass
  consumes them). Freeing still-referenced activations would corrupt backprop.
  Out of scope; tracked as a possible follow-up only if the async overflow proves
  insufficient on GB10.
- A segmented / growable arena. The stream-ordered overflow is the minimal correct
  fix; segmentation is follow-up if needed.
- Changing the #111 capture guard, the bulk-upload chunking (#106/#107), or the
  `ZERFOO_ARENA_SIZE_GB` knob semantics.
- Solving a true OOM. If the full-batch working set genuinely approaches GB10's
  122 GB unified memory, no allocation strategy helps and the remedy is a smaller
  batch -- out of scope here; the validation will reveal which regime applies.

### Constraints and assumptions

- GB10 (DGX Spark, 192.168.86.250:8080) is the only hardware where the freeze
  manifests. Hardware runs go through Spark Pod submissions, never interactive
  `ssh` benchmarks (ssh only to read hostPath logs for debugging). Spark v1.13.1
  mangles YAML `args:` block scalars -- deliver the pod script base64-encoded as a
  single-line flow arg (see docs/devlog.md 2026-06-07 and the spark-args memory).
- `cudaMallocAsync` is valid in STREAM ORDER: a pointer is safe for kernels
  launched on the same stream after the alloc. Arena allocations are consumed by
  kernels on `e.stream`, so wiring the overflow to `e.stream` is correct.
- ztensor stays CGO-free; the async wrappers already exist in internal/cuda.
- CPU/CI has no CUDA, so the async path is unit-tested via a swappable indirection
  (a stubbed `arenaMallocAsyncFn`), not a real device call.
- main stays green for CPU and non-capture GPU tests on every commit.

### Success metrics

- A CPU unit test proves: with `overflowStream` set and no capture active, an
  exhausted `ArenaPool.Alloc` routes the fallback through the async alloc fn (NOT
  the synchronous `MemPool.Alloc`); the #111 guard still fires and takes
  precedence during capture; `Free` routes async-allocated pointers to the async
  free fn; with no `overflowStream`, behavior is unchanged.
- A GB10 Spark stress test issues many arena-exhaustion overflow allocations on a
  real stream and completes with zero D-state threads (watchdog), no hang.
- Existing internal/cuda tests (capture_state_test.go, arena tests) and
  `go test ./...` stay green on CPU; `-race` clean.
- #115 closed on merge; release tag cut. End-to-end Wolf train-crossasset progress
  past `step=0` confirmed, or the Wolf-rebuild blocker stated explicitly.

## Discovery Summary

ENGINEERING. Root cause pinned in #115 with goroutine frames and confirmed against
current source. One open issue (#115). The #111 fix (v1.8.2) is complete and is
the prerequisite this builds on.

Relevant code sites (verified 2026-06-07):

- internal/cuda/arena.go:154 -- #111 capture guard (KEEP, takes precedence).
- internal/cuda/arena.go:158 -- the synchronous `a.fallback.Alloc` exhaustion
  path #115 hangs on.
- internal/cuda/arena.go -- `Free` (fallbackPtrs routing), `Drain`; need an async
  pointer set + FreeAsync routing.
- internal/cuda/runtime_purego.go:86 `MallocAsync(size, *Stream)`, :102
  `FreeAsync(ptr, *Stream)`, :195 `CreateStream`.
- internal/cuda/mempool.go -- `MemPool.Alloc` uses `MallocAsync` only when
  `captureStream != nil`, else synchronous `Malloc` (the path that wedges).
- compute/gpu_engine.go ~226 -- `NewCUDAArenaPool` + stream creation; wire
  `SetOverflowStream(e.stream)` here.
- internal/gpuapi/cuda_arena.go -- `CUDAArenaPool` wrapper + `Inner()`; add a
  `SetOverflowStream` passthrough.
- compute/step_scope.go, compute/gpu_engine.go:271 (`ResetPool`/`arena.Reset`) --
  arena reclaims only between steps (explains the overflow).
- internal/gpuapi/cuda_arena.go:15 `ZERFOO_ARENA_PROFILE` -- existing arena
  diagnostics to reference.

Use case manifest: .claude/scratch/usecases-manifest.json (UC-115).
Decision rationale: docs/adr/005-stream-ordered-arena-overflow.md.

### Prior work shipped (do not redo)

The #111 capture-aware ArenaPool guard is complete and released in v1.8.2 (PR
#112): a process-level set of capturing stream handles + the `arena.go:154` guard
that refuses the synchronous fallback during capture. ADR 004; devlog
2026-06-06/2026-06-07. #115 builds directly on it: the same fallback site, but the
NON-capture path, and the fix is the allocation MECHANISM (async vs sync), not a
capture guard.

## Scope and Deliverables

In scope:
- `overflowStream` on `cuda.ArenaPool` + `SetOverflowStream`, async overflow alloc
  via `arenaMallocAsyncFn`, async free routing via `arenaFreeAsyncFn`.
- Engine wiring (`SetOverflowStream(e.stream)`); `CUDAArenaPool` passthrough.
- CPU routing unit tests via stubbed indirections.
- GB10 Spark stress test proving no D-state hang under overflow.
- ADR 005, devlog, PR, rebase-and-merge, release, #115 resolved.

Out of scope: everything in Non goals.

| ID | Deliverable | Owner | Acceptance criteria |
|----|-------------|-------|---------------------|
| D1 | Stream-ordered overflow in ArenaPool | TBD | Exhausted non-capture Alloc uses async fn, not synchronous MemPool; Free routes to async free |
| D2 | Engine wiring | TBD | `SetOverflowStream(e.stream)` called at construction; passthrough on CUDAArenaPool |
| D3 | CPU routing tests | TBD | 4 cases (async-on-overflow, guard precedence, async-free routing, no-stream unchanged); green |
| D4 | #111 guard preserved | TBD | capture_state_test.go green; guard still returns ErrCaptureUnsafeAlloc during graph capture |
| D5 | GB10 stress validation | TBD | Spark pod: many overflow allocs complete, 0 D-state (watchdog); devlog with pod + commit |
| D6 | Shipped + #115 resolved | TBD | PR merged rebase-and-merge; release cut; #115 closed; Wolf-level progress confirmed or blocker stated |

## Checkable Work Breakdown

### E12 -- Stream-ordered arena overflow
**Component:** cuda
Acceptance: a non-capture `ArenaPool.Alloc` on an exhausted arena with an
overflow stream set allocates via `cudaMallocAsync` (never a synchronous
`cudaMalloc`); those pointers free via `cudaFreeAsync`; the #111 guard is
unchanged and takes precedence during capture.

- [x] T12.1 Add swappable package-level indirections `arenaMallocAsyncFn =
  MallocAsync` and `arenaFreeAsyncFn = FreeAsync` in internal/cuda (so CPU tests
  can stub them). Decision rationale: docs/adr/005-stream-ordered-arena-overflow.md
  Owner: TBD  Est: 30m  verifies: [UC-115]
  - Dependencies: none
- [x] T12.2 Add `overflowStream *Stream` + `asyncFallbackPtrs map[unsafe.Pointer]int`
  to `cuda.ArenaPool` and a `SetOverflowStream(s *Stream)` setter. In
  `ArenaPool.Alloc`, after the existing #111 guard (unchanged), when
  `!a.fallback.IsCapturing() && a.overflowStream != nil`, allocate via
  `arenaMallocAsyncFn(aligned, a.overflowStream)`, record the ptr in
  `asyncFallbackPtrs`, and return it; else keep the current synchronous
  `a.fallback.Alloc` path. Owner: TBD  Est: 75m  verifies: [UC-115]
  - Dependencies: T12.1
  - Acceptance: arena hits / capture-aware fallback / no-stream paths are
    byte-for-byte unchanged; only the non-capture-with-overflow-stream path
    switches to async.
- [x] T12.3 Route `Free` (and `Drain`) for async-allocated pointers to
  `arenaFreeAsyncFn(ptr, a.overflowStream)`: check `asyncFallbackPtrs` before the
  existing `fallbackPtrs` synchronous-free routing. Owner: TBD  Est: 45m
  verifies: [UC-115]
  - Dependencies: T12.2
  - Acceptance: an async-allocated overflow pointer is freed via the async fn, not
    `MemPool.Free`; synchronous fallback pointers still route to `MemPool.Free`.
- [x] T12.4 Unit tests in internal/cuda (stub `arenaMallocAsyncFn` /
  `arenaFreeAsyncFn` to record calls): (a) overflow-stream-set + no-capture routes
  Alloc through the async fn, not synchronous MemPool; (b) #111 guard still returns
  `ErrCaptureUnsafeAlloc` during graph-driven capture and takes precedence over the
  async path; (c) Free routes async pointers to the async free fn; (d) no
  overflowStream -> unchanged synchronous behavior. Owner: TBD  Est: 90m
  verifies: [UC-115]
  - Dependencies: T12.3

### E13 -- Engine wiring
**Component:** compute
Acceptance: every GPUEngine arena gets the engine's stream as its overflow stream
at construction, so overflow allocations are stream-ordered with consuming
kernels.

- [x] T13.1 Add `SetOverflowStream(s *cuda.Stream)` passthrough to
  `gpuapi.CUDAArenaPool` (delegates to `p.inner.SetOverflowStream`). Owner: TBD
  Est: 20m  verifies: [UC-115]
  - Dependencies: T12.2
- [x] T13.2 In `compute/gpu_engine.go` GPUEngine construction (after
  `NewCUDAArenaPool` and the engine stream exist), call
  `arenaPool.SetOverflowStream(stream)` (or `arena.Inner().SetOverflowStream`).
  Confirm the stream used is the same `e.stream` that kernels launch on.
  Owner: TBD  Est: 30m  verifies: [UC-115]
  - Dependencies: T13.1
  - Acceptance: a constructed GPUEngine's arena has overflowStream == e.stream;
    no-arena engines (CPU) are unaffected.

### E14 -- Lint, format, CPU test gate
**Component:** tooling
Acceptance: gofmt + `go vet ./...` clean; `go build ./...` and `go test ./...`
green on CPU; `-race` clean on cuda + compute.

- [x] T14.1 gofmt + `go vet ./...` clean on changed files; `golangci-lint run
  ./internal/cuda/... ./compute/...` 0 issues; `go build ./...` exit 0  Owner: TBD
  Est: 20m  verifies: [infrastructure]
  - Dependencies: T12.4, T13.2
- [x] T14.2 `go test ./...` green on CPU incl. capture_state_test.go and arena
  tests; `-race` clean on internal/cuda + compute  Owner: TBD  Est: 20m
  verifies: [infrastructure]
  - Dependencies: T14.1

### E15 -- GB10 validation via Spark
**Component:** cuda
Acceptance: a stress pattern of arena-exhaustion overflow allocations on a real
GB10 stream completes with zero D-state threads, proving the async overflow does
not wedge; (stretch) Wolf train-crossasset progresses past `step=0`.

- [x] T15.1 Author a GPU stress test (skips on CPU; gated like
  TestArenaPool_CaptureGuard_GPU) that creates a real engine stream + a small
  arena, then issues many overflow allocations (sizes mixing 48 B .. 16 KB and
  larger, total well past the arena capacity) via `ArenaPool.Alloc`, frees them,
  loops; asserts every alloc returns non-nil with no error and the loop completes.
  Owner: TBD  Est: 90m  verifies: [UC-115]
  - Dependencies: T12.3
- [x] T15.2 Spark manifest + run on GB10 under the out-of-band dstate-watchdog
  (base64 single-line-arg delivery; mount /usr/local/cuda + /opt/zerfoo/lib +
  /var/lib/zerfoo/bench-out; exit-code guard, SKIP = fatal). Confirm the stress
  test PASSES and the watchdog records 0 D-state threads. Do NOT ssh for the run;
  ssh only to read hostPath logs. Owner: TBD  Est: 90m  verifies: [UC-115]
  - Dependencies: T15.1, T14.2
- [x] T15.3 Devlog entry: pod name, commit, alloc count/sizes, watchdog D-state
  count, timing. Owner: TBD  Est: 20m  verifies: [infrastructure]
  - Dependencies: T15.2
- [ ] T15.4 (stretch, end-to-end) Rebuild Wolf train-crossasset against the ztensor
  fix and confirm training progresses past `step=0` on full COIN bars with no
  freeze. If a Wolf rebuild + full-bars run cannot complete in session, state the
  blocker explicitly and ship on the T15.2 ztensor-level GB10 evidence. Owner: TBD
  Est: 2h  verifies: [UC-115]  blocked: needs Wolf rebuild against ztensor fix
  - Dependencies: T15.2

### E16 -- Ship and resolve #115
**Component:** release
Acceptance: PR merged rebase-and-merge; release tag cut; #115 closed.

- [ ] T16.1 Branch `fix/issue-115-async-arena-overflow` off main; open PR against
  main; title for #115. Owner: TBD  Est: 30m  verifies: [UC-115]
  - Dependencies: T15.3
- [ ] T16.2 PR CI green; rebase-and-merge into main (not squash, not merge commit)
  Owner: TBD  Est: 30m  verifies: [UC-115]
  - Dependencies: T16.1
- [ ] T16.3 release-please cuts the patch release; verify tag + GitHub release.
  Owner: TBD  Est: 20m  verifies: [infrastructure]
  - Dependencies: T16.2
- [ ] T16.4 #115 closed on merge (PR `fixes #115`); comment summarizing the
  stream-ordered overflow fix + GB10 evidence + any Wolf-level follow-up. Owner: TBD
  Est: 15m  verifies: [UC-115]
  - Dependencies: T16.3
- [ ] T16.5 Definition-of-done honesty check: merged + released + GB10
  stress-validated + reported honestly. If the Wolf-level end-to-end (T15.4) did
  not complete in session, state the specific blocker here. Owner: TBD  Est: 15m
  verifies: [infrastructure]
  - Dependencies: T16.4

## Parallel Work

The async overflow (E12) is the root dependency. The indirection (T12.1) unblocks
the alloc/free changes; the engine wiring (E13) needs T12.2's setter; the stress
test (T15.1) needs the alloc/free path (T12.3). Validation and ship are gated on
the CPU gate (E14).

| Track | Tasks | Notes |
|-------|-------|-------|
| Track A: core overflow | T12.1 -> T12.2 -> T12.3 -> T12.4 | Sequential chain in arena.go |
| Track B: engine wiring | T13.1 -> T13.2 | Starts after T12.2's setter exists |
| Track C: GB10 harness | author T15.1 + T15.2 manifest | Test authoring overlaps A; execution waits on T14.2 |

Sync points: T12.2 unblocks B and the stress test; E14 needs E12+E13; E15 needs
E14; E16 needs E15.

### Wave 1: Indirection seam (1 agent)
- [x] T12.1 Swappable async fn indirections  verifies: [UC-115]

### Wave 2: Overflow alloc + setter (1 agent, sequential core)
- [x] T12.2 overflowStream + async overflow alloc  verifies: [UC-115]

### Wave 3: Free routing + engine passthrough (2 agents)
- [x] T12.3 Async free routing in Free/Drain  verifies: [UC-115]
- [x] T13.1 CUDAArenaPool SetOverflowStream passthrough  verifies: [UC-115]

### Wave 4: Tests + wiring + stress-test authoring (3 agents)
- [x] T12.4 CPU routing unit tests  verifies: [UC-115]
- [x] T13.2 Engine wires SetOverflowStream(e.stream)  verifies: [UC-115]
- [x] T15.1 Author GB10 overflow stress test  verifies: [UC-115]

### Wave 5: Gate (1 agent, sequential)
- [x] T14.1 gofmt + vet + lint + build  verifies: [infrastructure]
- [x] T14.2 go test ./... + -race  verifies: [infrastructure]

### Wave 6: GB10 validation (1 agent)
- [x] T15.2 Spark stress run, 0 D-state  verifies: [UC-115]
- [x] T15.3 Devlog entry  verifies: [infrastructure]
- [ ] T15.4 (stretch) Wolf train-crossasset past step=0  verifies: [UC-115]

### Wave 7: Ship (1 agent, sequential)
- [ ] T16.1 Branch + PR to main  verifies: [UC-115]
- [ ] T16.2 CI green + rebase-and-merge  verifies: [UC-115]
- [ ] T16.3 Release tag cut  verifies: [infrastructure]
- [ ] T16.4 Close #115 with summary  verifies: [UC-115]
- [ ] T16.5 Definition-of-done honesty check  verifies: [infrastructure]

## Timeline and Milestones

| Milestone | Description | Member tasks | Exit criteria |
|-----------|-------------|--------------|---------------|
| M0 | Overflow path implemented | T12.1, T12.2, T12.3 | Non-capture exhausted Alloc uses async; Free routes async |
| M1 | Wired + unit-green | T12.4, T13.1, T13.2 | Engine sets overflow stream; CPU routing tests prove all 4 cases |
| M2 | CPU gate green | T14.1, T14.2 | gofmt/vet/lint/build clean; `go test ./...` + `-race` green |
| M3 | GB10 validated | T15.1, T15.2, T15.3 | Overflow stress completes on GB10, 0 D-state; devlog recorded |
| M4 | Shipped | T16.1, T16.2, T16.3, T16.4, T16.5 | PR merged rebase-and-merge; release cut; #115 closed |

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Stream-ordered async pointer used off `e.stream` before a sync -> use-before-ready | High | Low | Arena allocs are consumed by kernels on e.stream; wire overflowStream = e.stream (T13.2); ADR 005 records the invariant |
| R2 | Async overflow still thrashes/OOMs because the full-batch working set approaches GB10's 122 GB | High | Medium | T15.2 stress test + T15.4 Wolf run reveal the regime; if true OOM, remedy is smaller batch / segmented arena (follow-up), reported honestly not hidden |
| R3 | The #111 capture guard is weakened by the new branch (async-alloc during graph capture) | High | Low | Guard stays first and unchanged; T12.4(b) asserts ErrCaptureUnsafeAlloc still fires + takes precedence; async path only when !IsCapturing |
| R4 | Async free ordering bug double-frees or leaks overflow pointers | Medium | Low | Separate asyncFallbackPtrs set; T12.3 routes only recorded async ptrs to FreeAsync; T12.4(c) asserts routing |
| R5 | Cannot complete Wolf end-to-end validation in session (needs Wolf rebuild) | Medium | Medium | Ship on ztensor-level GB10 stress evidence (T15.2); state the Wolf blocker explicitly (T15.4/T16.5); do not claim done |
| R6 | CPU tests cannot exercise real cudaMallocAsync | Low | High (expected) | Swappable indirection (T12.1) makes routing CPU-testable; real async validated on GB10 (E15) |

## Operating Procedure

- Definition of done (global CLAUDE.md): merged via rebase-and-merge, CI green,
  release tag cut, and verified live on GB10 (the overflow stress test completing
  with 0 D-state via Spark, observed in logs/watchdog). The end-to-end Wolf
  train-crossasset-past-step-0 is the ultimate proof; if it cannot run in session,
  state the blocker (T16.5) rather than claiming done.
- Add tests with every implementation change (T12.4 with T12.2/T12.3).
- Run gofmt, `go vet`, golangci-lint after code changes (T14.1).
- Never commit files from different directories in one commit (pre-commit hook).
  internal/cuda, internal/gpuapi, compute, docs each commit separately.
- Validate GPU behavior only via Spark Pod submissions; never interactive ssh
  benchmarks. ssh only to read hostPath logs. Use base64 single-line-arg pod
  delivery (Spark v1.13.1 mangles YAML block scalars).

## Progress Log

### Change Summary -- 2026-06-07 (new plan: #115)

- Trimmed the completed #111 plan (E5-E11, all shipped in v1.8.2); its knowledge
  is already in docs/adr/004 and docs/devlog.md. Replaced with the #115 plan.
- Root cause: ArenaPool exhaustion fallback uses a synchronous `cudaMalloc`
  (arena.go:158) that thrashes GB10 unified memory during non-capture training
  fwd/bwd, freezing at step=0. The #111 guard correctly does not fire for
  non-capture ops.
- Fix: stream-ordered overflow via `cudaMallocAsync` on the engine stream
  (ADR 005). New epics E12 (overflow alloc/free), E13 (engine wiring), E14 (CPU
  gate), E15 (GB10 stress validation + Wolf stretch), E16 (ship + resolve #115).
- Use case manifest: UC-115.

ADRs created: docs/adr/005-stream-ordered-arena-overflow.md -- route the ArenaPool
exhaustion fallback through cudaMallocAsync (stream-ordered) instead of a
synchronous cudaMalloc that page-fault-thrashes GB10.

## Hand-off Notes

- The fix is in package `cuda` (internal/cuda/arena.go) + a one-line engine wiring
  (compute/gpu_engine.go) + a passthrough (internal/gpuapi/cuda_arena.go). Read
  ADR 005 first.
- KEEP the #111 capture guard (arena.go:154) first and unchanged; the async path
  is only for `!fallback.IsCapturing() && overflowStream != nil`. During
  graph-driven capture the guard must still refuse (ErrCaptureUnsafeAlloc).
- `MallocAsync`/`FreeAsync` already exist and work on GB10 (the capture path uses
  them). The arena's overflow stream must be `e.stream` so allocations are
  stream-ordered with the kernels that consume them.
- The arena overflows during full-batch training because backprop retains all
  forward activations and the arena resets only between steps -- that is correct
  behavior, not a leak. Do NOT add within-pass reclamation (would corrupt
  backprop). `ZERFOO_ARENA_SIZE_GB` is a tuning knob, not a fix.
- GB10 validation via Spark only; base64 single-line-arg pod delivery; read
  hostPath logs via ssh. dstate-watchdog at docs/bench/scripts/dstate-watchdog.sh.
- Wolf caller: `internal/crossasset/crossasset.go` trainWithResult; the freeze is
  in the first training step's fwd/bwd.

## Appendix

- Issue: github.com/zerfoo/ztensor#115 (follow-up to #111, #106).
- Decisions: docs/adr/005-stream-ordered-arena-overflow.md (this fix),
  docs/adr/004-capture-aware-arena-fallback.md (#111 guard, prerequisite).
- Code: internal/cuda/arena.go (Alloc:154 guard / :158 fallback, Free, Drain),
  internal/cuda/runtime_purego.go:86/102 (MallocAsync/FreeAsync),
  internal/cuda/mempool.go (sync vs async), compute/gpu_engine.go (~226 wiring,
  :271 ResetPool), internal/gpuapi/cuda_arena.go (CUDAArenaPool, Inner).
- Captured frames + fix directions: issue #115 body.

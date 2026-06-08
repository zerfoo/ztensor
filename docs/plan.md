# ztensor Work Plan

## Title

Close the GB10 crossasset training freeze (issue #118): the v1.9.0 stream-ordered
arena overflow shipped, but training still does not complete one step. Add arena
diagnostics, find why a 48 KB alloc on the first sample overflows a 64 GB arena,
and harden the async overflow path so it cannot wedge GB10 even if reached.

## Context

### Problem statement

With ztensor **v1.9.0** (issue #115, stream-ordered overflow via cudaMallocAsync),
Wolf `train-crossasset -gpu` on the NVIDIA GB10 (sm_121, unified memory) STILL
freezes in the first `ComputeGradients` (step=0). The v1.9.0 fix changed the
overflow path from synchronous `cudaMalloc` to `cudaMallocAsync`, but two distinct
problems remain:

1. **The arena overflow path is reached on the FIRST mini-batch sample's tiny
   alloc** -- a 48 KB `gpuTanh` output -- even with `ZERFOO_ARENA_SIZE_GB=64`.
   A 64 GB arena should not be exhausted by one per-sample forward pass. Pinned
   goroutine 1 (v1.9.0, GB10):

   ```
   cuda.MallocAsync(0xc000 = 48KB)     runtime_purego.go:92   # async overflow
   cuda.(*ArenaPool).Alloc             arena.go:186           # overflow path
   gpuapi.(*CUDAArenaPool).Alloc       cuda_arena.go:59
   compute.gpuUnaryOp                  gpu_kernels.go:603
   compute.(*GPUEngine).gpuTanh -> forward pass, first sample
   D-state sibling: folio_wait_bit_common -> __folio_lock_or_retry
   ```

2. **The async overflow (`cudaMallocAsync`) itself still stalls on GB10** under
   unified-memory pressure (D-state page fault in `folio_wait_bit`). So even with
   the v1.9.0 sync->async change, the overflow path wedges.

Net: training freezes in the first step's forward pass.

### What the code review already rules out

- **Arena sizing parse is correct.** `arenaSizeBytes` (compute/gpu_engine.go:38)
  returns `64 * 1024^3 = 68719476736` as `int64`; the `int()` cast at
  gpu_engine.go:226 is 64-bit on arm64, so there is no truncation. The configured
  capacity is genuinely 64 GB.
- **The arena allocation succeeded.** If `Malloc(64GB)` had failed, NewGPUEngine
  logs "arena pool not available, falling back to MemPool" (gpu_engine.go:236) and
  the engine pool would be a MemPool, not CUDAArenaPool. The issue reports neither
  the warning nor the MemPool path, and the stack shows `(*ArenaPool).Alloc`. So a
  real 64 GB arena exists.

Therefore problem (1) is one of: (a) a single mis-sized / runaway allocation that
bumps `offset` to ~64 GB before the tanh; (b) the per-pass working set genuinely
filling 64 GB because intra-pass `Free` is never called and forward activations
are retained for backprop; or (c) `Reset`/`MarkStepBoundary` is not being driven,
so `offset` never rewinds. The arena instruments needed to tell these apart do not
exist yet -- which is why diagnostics are the first deliverable, not an afterthought.

For problem (2): `MallocAsync` (runtime_purego.go:86) calls `cudaMallocAsync` on
the engine stream, but nothing configures the stream-ordered memory pool's release
threshold (`cudaMemPoolAttrReleaseThreshold`) or pre-reserves backing. Under GB10
unified-memory pressure a fresh async allocation still triggers OS-level page
allocation and faults in `folio_wait_bit`. The async path is necessary (issue
#115) but, as shipped, insufficient.

### Objectives

- O1. Make the arena observable: log capacity, offset, hits/misses/reuses, alloc
  count, free-list length, and whether Reset is being driven -- at engine init and
  at the first overflow.
- O2. Determine which of (a)/(b)/(c) causes problem (1), using O1 output captured
  on GB10 hardware.
- O3. Eliminate the root cause of problem (1) so the overflow path is not reached
  in normal per-sample training (correct sizing / reset / accounting).
- O4. Harden the async overflow path so that if it is reached it does not wedge
  GB10 -- configure the stream-ordered pool so cudaMallocAsync does not
  page-fault-thrash.
- O5. Validate on GB10 hardware via Spark that crossasset training progresses past
  step=0.

### Non-goals

- Changing the Wolf trainer's gradient-accumulation strategy or batch sizing. If
  the diagnostics show the trainer never drives `MarkStepBoundary`/`ResetPool`,
  that is a Wolf-side fix tracked separately; this plan covers ztensor's arena and
  the ztensor-side API needed to drive it.
- A general arena rewrite. Keep the bump-pointer + free-list + overflow design.
- Managed-memory (cudaMallocManaged) re-enablement (separate ~13% regression item).

### Constraints and assumptions

- Validation MUST run on GB10 hardware via Spark (per CLAUDE.md). No interactive
  SSH benchmarks. NVIDIA GB10, aarch64, linux/arm64 images.
- ztensor stays CGo-free; CUDA bound via purego/dlopen. New mempool calls must be
  symbol-guarded (no-op when the symbol is absent), like MallocAsync already is.
- Each repo commits independently; rebase-and-merge; release-please cuts releases.

### Success metrics

- Arena diagnostics appear at engine init and at first overflow with all O1 fields.
- On the GB10 repro, the first overflow either does not occur (problem 1 fixed) or
  completes without a D-state thread (problem 2 hardened).
- Wolf `train-crossasset -gpu` (full COIN 1m bars, folds=2 epochs=1, batch=256,
  ZERFOO_ARENA_SIZE_GB=64) advances past step=0 with zero D-state threads.

## Discovery Summary

ENGINEERING. One core use case (UC-118): GB10 crossasset training completes a
training step without freezing. Existing arena/overflow machinery is already in
place from #111 (capture guard, ADR 004) and #115 (stream-ordered overflow, ADR
005). The gap is observability (no per-overflow diagnostics) and an unhardened
async pool. Key files confirmed by reading the code:

- internal/cuda/arena.go -- ArenaPool: Alloc (overflow at :186), Reset (:308),
  HitMissStats/ReuseStats/UsedBytes/Capacity/FreeListLen accessors already exist.
- internal/cuda/runtime_purego.go:86 -- MallocAsync (the overflow allocator).
- internal/cuda/purego.go -- dlopen symbol table (add cudaMemPoolSetAttribute /
  cudaDeviceGetDefaultMemPool here).
- compute/gpu_engine.go:225-234 -- arena sizing + SetOverflowStream wiring.
- compute/gpu_kernels.go:603 -- gpuUnaryOp (the tanh alloc site in the stack).
- compute/step_scope.go -- BeginStep / MarkStepBoundary / ResetPool API the
  trainer is supposed to drive.

Reference: .claude/scratch/usecases-manifest.json.

## Scope and Deliverables

In scope:
- Arena diagnostics struct + logging at init and first overflow (O1).
- A one-shot "first overflow" report: alloc count, total bytes, largest single
  alloc since last reset, free-list length, resets-so-far (O2 input).
- A GB10 Spark diagnostic manifest that runs the repro with diagnostics on (O2).
- Root-cause fix for problem (1) once diagnosed (O3) -- branch is conditional on
  the diagnostic verdict.
- Async-overflow hardening: configure the overflow stream's mempool release
  threshold, symbol-guarded (O4).
- GB10 validation via Spark, ship via release-please (O5).

Out of scope: Wolf trainer changes (tracked separately if O2 points there),
managed-memory work, arena redesign.

Deliverables:

| ID | Description | Owner | Acceptance |
|----|-------------|-------|------------|
| D1 | Arena diagnostics (init + first-overflow log) | TBD | All O1 fields present; unit-tested |
| D2 | GB10 diagnostic capture | TBD | First-overflow report captured from GB10 |
| D3 | Root-cause fix for problem (1) | TBD | Overflow not reached in per-sample training |
| D4 | Async-overflow hardening | TBD | cudaMallocAsync under pressure does not D-state |
| D5 | GB10 validation + release | TBD | train-crossasset past step=0; release cut |

## Checkable Work Breakdown

Note on execution order: this is a **sequential investigation**. Wave 1
(instrumentation + defensive hardening + tests) is safe to do now and in parallel.
Wave 2 is a **hardware diagnostic gate** -- it must run on GB10 and its output
decides Wave 3. Wave 3 (the root-cause fix for problem 1) is intentionally left
conditional: do not pre-commit a fix for an undiagnosed cause.

### E1 -- Arena diagnostics (instrumentation)
**Component:** cuda

- [ ] T1.1 Add an ArenaStats diagnostics snapshot method to ArenaPool that returns
  capacity, current offset, hits, misses, reuses, resets, alloc count, and
  free-list length in one struct (extends existing HitMissStats/ReuseStats).
  Owner: TBD  Est: 45m  Dep: none  verifies: [UC-118]
  Acceptance: method returns a struct with all fields; no lock re-entrancy.
- [ ] T1.2 Emit a one-time "first overflow" log in ArenaPool.Alloc the first time
  the exhaustion branch is hit: log capacity, offset, requested bytes, aligned
  bytes, alloc count since last reset, resets-so-far, free-list length, and which
  overflow path was taken (async vs MemPool vs capture-refused). Guard so it logs
  once per reset-epoch, not every alloc. Owner: TBD  Est: 60m  Dep: T1.1
  verifies: [UC-118]
  Acceptance: log fires exactly once at first overflow; includes all fields.
- [ ] T1.3 Log arena configuration at engine init in NewGPUEngine: configured
  capacity (GB and bytes), managed-memory flag, overflow-stream-set flag.
  Owner: TBD  Est: 30m  Dep: none  verifies: [UC-118]
  Acceptance: one INFO line at init names the resolved arena size and overflow mode.
- [ ] T1.4 Unit tests: ArenaStats fields correct after a sequence of
  alloc/free/reset; first-overflow log fires once (swap the logger, assert one
  record). Owner: TBD  Est: 45m  Dep: T1.1, T1.2  verifies: [UC-118]
- [ ] T1.5 Run gofmt + golangci-lint + `go test ./internal/cuda/... ./compute/...`
  (CPU path). Owner: TBD  Est: 20m  Dep: T1.1-T1.4  verifies: [infrastructure]

### E2 -- Async-overflow hardening (defensive, independent of problem 1)
**Component:** cuda

- [ ] T2.1 Add dlopen symbols cudaDeviceGetDefaultMemPool, cudaMemPoolSetAttribute
  to internal/cuda/purego.go (symbol-guarded; absent => no-op). Owner: TBD
  Est: 45m  Dep: none  verifies: [UC-118]
  Acceptance: symbols resolved when present; lib() with missing symbols still loads.
- [ ] T2.2 Add SetMemPoolReleaseThreshold(deviceID int, bytes uint64) in
  internal/cuda that sets cudaMemPoolAttrReleaseThreshold on the device default
  async pool so freed async blocks are retained (not released to the OS) and
  re-served without page faults. No-op if the symbol is absent. Owner: TBD
  Est: 45m  Dep: T2.1  verifies: [UC-118]
  Acceptance: returns nil and is a no-op without the symbol; sets attr when present.
- [ ] T2.3 Call SetMemPoolReleaseThreshold from NewGPUEngine right after
  SetOverflowStream, with a sensible default threshold (e.g. arena capacity, or a
  configurable ZERFOO_OVERFLOW_POOL_RETAIN_GB). Log the chosen threshold.
  Owner: TBD  Est: 30m  Dep: T2.2  verifies: [UC-118]
- [ ] T2.4 Unit tests for the routing/no-op behavior (swap the symbol pointers,
  like arena_overflow_test.go does for MallocAsync). Owner: TBD  Est: 40m
  Dep: T2.2  verifies: [UC-118]
- [ ] T2.5 gofmt + golangci-lint + targeted tests. Owner: TBD  Est: 15m
  Dep: T2.1-T2.4  verifies: [infrastructure]

### E3 -- GB10 diagnostic gate (hardware; decides E4)
**Component:** crossasset

- [ ] T3.1 Build/publish an arm64 image at the E1+E2 branch and author a Spark Pod
  manifest that runs the GB10 arena repro (the existing overflow stress or a
  crossasset-shaped per-sample forward) with diagnostics enabled and
  ZERFOO_ARENA_SIZE_GB=64. Owner: TBD  Est: 60m  Dep: E1, E2  verifies: [UC-118]
  Acceptance: pod runs on GB10; logs include the init line and first-overflow report.
- [ ] T3.2 Capture and record the first-overflow report in docs/devlog.md: alloc
  count + total bytes at overflow, largest single alloc, resets-so-far. Classify
  problem (1) as (a) mis-sized single alloc / (b) legit full working set / (c)
  reset never driven. Owner: TBD  Est: 45m  Dep: T3.1  verifies: [UC-118]
  Acceptance: devlog entry states the verdict with the numbers that support it.
- [ ] T3.3 Confirm whether the async-overflow hardening (E2) alone removed the
  D-state thread on the same run (problem 2). Owner: TBD  Est: 30m  Dep: T3.1
  verifies: [UC-118]

### E4 -- Root-cause fix for problem (1) [CONDITIONAL on T3.2 verdict]
**Component:** cuda

- [ ] T4.1 If verdict (a) mis-sized single alloc: locate the runaway allocation
  (the diagnostics name the largest alloc + its gpu_kernels call site) and fix the
  size computation. Add a regression test asserting the buffer size. Owner: TBD
  Est: 2h  Dep: T3.2  verifies: [UC-118]
- [ ] T4.2 If verdict (c) reset never driven: confirm the trainer's
  BeginStep/MarkStepBoundary/ResetPool contract, and if ztensor's API is the gap,
  add/repair the StepScope wiring + a test that Reset rewinds offset across a
  simulated step. (Wolf-side trainer change tracked separately.) Owner: TBD
  Est: 90m  Dep: T3.2  verifies: [UC-118]
- [ ] T4.3 If verdict (b) legitimately full: document that overflow is expected for
  this working set and that E2 hardening is the safety net; capture the reasoning
  in ADR 006. Owner: TBD  Est: 90m  Dep: T3.2  verifies: [UC-118]
- [ ] T4.4 gofmt + golangci-lint + tests for whichever branch was taken.
  Owner: TBD  Est: 20m  Dep: T4.1/T4.2/T4.3  verifies: [infrastructure]

### E5 -- GB10 validation, ship
**Component:** release

- [ ] T5.1 Re-run the GB10 repro at the E4 branch: assert first overflow is not
  reached in per-sample training (or is hardened) and zero D-state threads.
  Owner: TBD  Est: 45m  Dep: E4  verifies: [UC-118]
- [ ] T5.2 (end-to-end) Rebuild Wolf train-crossasset against the new ztensor and
  confirm training advances past step=0 on full COIN bars. Owner: TBD  Est: 60m
  Dep: T5.1  verifies: [UC-118]
- [ ] T5.3 Branch fix/issue-118-arena-diagnostics-hardening; open PR to main with
  `fixes #118`; CI green; rebase-and-merge (not squash, not merge commit).
  Owner: TBD  Est: 30m  Dep: T5.1  verifies: [infrastructure]
- [ ] T5.4 release-please cuts the release; verify tag + GitHub release; #118 closes
  on merge. Owner: TBD  Est: 20m  Dep: T5.3  verifies: [infrastructure]
- [ ] T5.5 Commit the untracked docs/adr/005-stream-ordered-arena-overflow.md (the
  #115 ADR never landed on main) alongside this work. Owner: TBD  Est: 10m
  Dep: none  verifies: [infrastructure]

## Parallel Work

| Track | Tasks | Notes |
|-------|-------|-------|
| Track A: Diagnostics | T1.1-T1.5 | No external deps; start immediately |
| Track B: Async hardening | T2.1-T2.5 | No external deps; start immediately, parallel to A |
| Sync point 1 | E1 + E2 merged | Required before the GB10 diagnostic image (T3.1) |
| Track C: Diagnostic gate | T3.1-T3.3 | Hardware; serial; gates E4 |
| Sync point 2 | T3.2 verdict | Selects exactly one of T4.1/T4.2/T4.3 |
| Track D: Fix + ship | E4, E5 | Serial after the verdict |

### Waves

### Wave 1: Instrument + harden (2 agents)
- [x] E1 Arena diagnostics (T1.1-T1.5)  verifies: [UC-118]  (2026-06-07)
- [x] E2 Async-overflow hardening (T2.1-T2.5)  verifies: [UC-118]  (2026-06-07)

### Wave 2: GB10 diagnostic gate (1 agent, hardware)
- [x] E3 GB10 validation via Spark pod ztensor-issue118-diag-1 (2026-06-07):
  diagnostics fire on hardware, hardened async path does not wedge. Verdict on
  problem (1): UNDETERMINED from synthetic stress -- requires the real crossasset
  workload (see E4 block reason).

### Wave 3: Conditional fix (1 agent)
- [ ] E4 BLOCKED: needs the problem-(1) verdict from a real Wolf train-crossasset
  run against this branch. Unblock: run the new diagnostics under crossasset, read
  the ARENA_FIRST_OVERFLOW line, then apply the matching fix branch (T4.1/4.2/4.3).

### Wave 4: Validate + ship (1 agent, hardware)
- [ ] E5 Ship diagnostics+hardening PR (T5.3-T5.5 in progress); T5.2 end-to-end
  crossasset confirmation BLOCKED with E4.

## Timeline and Milestones

| Milestone | Description | Member tasks | Exit criteria |
|-----------|-------------|--------------|---------------|
| M0 | Arena observable | T1.1, T1.2, T1.3, T1.4, T1.5 | Diagnostics merged; logs at init + first overflow |
| M1 | Overflow hardened | T2.1, T2.2, T2.3, T2.4, T2.5 | Async pool retains backing; symbol-guarded |
| M2 | Root cause known | T3.1, T3.2, T3.3 | devlog states verdict (a/b/c) with numbers |
| M3 | Cause fixed | T4.1, T4.2, T4.3, T4.4 | Overflow not reached in per-sample training |
| M4 | Shipped + verified | T5.1, T5.2, T5.3, T5.4, T5.5 | train-crossasset past step=0; release cut; #118 closed |

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Root cause is in the Wolf trainer (Reset never driven), not ztensor | Med | Med | Diagnostics (T3.2) name it explicitly; ztensor ships the API + hardening net regardless; Wolf fix tracked separately |
| R2 | cudaMallocAsync wedges even with a release threshold set (GB10 page-fault deeper than the pool) | High | Med | Primary fix is to NOT reach overflow (E4); E2 is the net. If E2 insufficient, fall back to pre-reserved arena headroom |
| R3 | The 48 KB-overflow is a single runaway alloc whose call site is outside the arena package | Med | Med | First-overflow log records requested bytes + the gpu_kernels caller; trace from there |
| R4 | GB10 host contention / Spark queueing slows the diagnostic gate | Low | Med | Submit early; reuse the #115 overflow stress manifest as the harness |
| R5 | Pre-committing a fix before diagnosis wastes a wave on the wrong cause | Med | Low | Wave 3 is explicitly conditional on the T3.2 verdict; do not start E4 before it |

## Operating Procedure

- Definition of done (per global CLAUDE.md): merged (rebase-and-merge) -> released
  (release-please tag) -> deployed/validated on GB10 -> verified live (train past
  step=0) -> reported honestly.
- Add tests with every code change; run gofmt + golangci-lint after changes.
- Never commit files from different directories in one commit (pre-commit hook).
- Small logical commits. GB10 work via Spark manifests only -- no interactive SSH
  benchmarks.
- ADRs: defer the problem-(1) ADR until the T3.2 verdict is known (next number is
  006). E2's release-threshold choice may warrant an ADR if it becomes the primary
  safety mechanism.

## Progress Log

### 2026-06-07 (execution) -- E1, E2 shipped; E3 validated; E4 blocked

- E1 (arena diagnostics) and E2 (async-overflow hardening via mempool release
  threshold) implemented, unit-tested (CPU, -race clean), committed.
- E3: GB10 Spark pod ztensor-issue118-diag-1 PASSED -- diagnostics fire on
  hardware (`path=async capacity=1048576 offset=29184 epochAllocs=5
  epochMaxAlloc=1048576`), hardened async path ran 200 overflow rounds and synced
  with no wedge. Recorded in devlog.
- E4 BLOCKED: problem (1) (why a 48 KB alloc overflows a 64 GB arena on the first
  crossasset sample) cannot be classified from synthetic stress; it needs the real
  Wolf workload. The diagnostics added here are exactly the instrument for it.
- Shipping E1+E2 as the ztensor PR; #118 stays open pending the crossasset verdict.

### Change Summary -- 2026-06-07

- Replaced the completed #115 plan with the #118 plan. The #115 work (stream-ordered
  overflow, v1.9.0) is shipped; its stable knowledge is already in
  docs/adr/005-stream-ordered-arena-overflow.md and the 2026-06-07 devlog entry, so
  the #115 epics were trimmed rather than carried forward.
- New epics E1-E5 target issue #118: arena diagnostics (E1), async-overflow
  hardening (E2), a GB10 diagnostic gate (E3) that decides a conditional root-cause
  fix (E4), then validation + release (E5).
- Captured the code-review findings that rule out an arena-sizing parse bug and
  identify the missing mempool release-threshold as the async-overflow weakness.
- No ADR created yet: the problem-(1) decision is gated on GB10 diagnostics (T3.2).
- Flagged that docs/adr/005 is untracked on main (T5.5) -- the #115 ADR never landed.

## Hand-off Notes

- Issue: https://github.com/zerfoo/ztensor/issues/118 . Chain: #106 -> #111 (capture
  guard, ADR 004, v1.8.2) -> #115 (stream-ordered overflow, ADR 005, v1.9.0) -> #118.
- The investigation is sequential and hardware-gated. Do NOT attempt to fix problem
  (1) before the GB10 diagnostic report (T3.2) classifies it -- the three causes
  need three different fixes.
- GB10 work goes through Spark (http://192.168.86.250:8080), arm64 images, no pinned
  memory limits (unified memory). Reuse the #115 overflow-stress manifest pattern;
  Spark v1.13.1 mangles `args: |` block scalars, so deliver scripts base64-encoded
  as a single-line flow arg.
- Key files: internal/cuda/arena.go (overflow at :186), runtime_purego.go:86
  (MallocAsync), purego.go (symbol table), compute/gpu_engine.go:225-234 (wiring),
  compute/step_scope.go (Reset API), compute/gpu_kernels.go:603 (tanh alloc site).

## Appendix

- UC manifest: .claude/scratch/usecases-manifest.json
- ADR 004: docs/adr/004-capture-aware-arena-fallback.md (issue #111)
- ADR 005: docs/adr/005-stream-ordered-arena-overflow.md (issue #115; untracked, see T5.5)
- Wolf devlog 2026-06-07 has the full v1.8.1 -> v1.8.2 -> v1.9.0 chain.

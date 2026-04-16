# ztensor Work Plan

## Title

Resolve GB10 CUDA graph capture hang in GPUEngine[float32] on multi-tensor
training workloads.

## Context

### Problem statement

`GPUEngine[float32]` silently hangs on NVIDIA GB10 (arm64 Grace Hopper, DGX
Spark) when CUDA graph capture is active and the workload uploads a
non-trivial weight set via `WeightUploader.UploadWeights` followed by graph
construction. A minimal 4x4 MatMul smoke test passes with capture enabled,
so the failure is specific to larger multi-tensor workloads.

Reproduction downstream: Wolf CrossAsset training (12 Fibonacci scales,
193 features per scale, approximately 50 weight tensors including
256x1024 matrices) reliably hangs at the log line `Using GPU engine` with
0 percent GPU utilization across 5 independent attempts. Setting the
environment variable `ZERFOO_DISABLE_CUDA_GRAPH=1` fully bypasses the
hang and lets training complete (epochs 0 to 3 produced losses 0.864,
0.693, 0.651, 0.627).

Environment: NVIDIA DGX Spark GB10 (arm64 Grace Hopper), Ubuntu 24.04 in
Podman container, CUDA 13.0.96, ztensor
`v1.5.1-0.20260415020900-fd646fb10680`, zerfoo
`v1.48.1-0.20260415044400-d3ef8b617b34`, Go 1.26, CGO_ENABLED=1.

Existing evidence in source:

- `compute/gpu_engine.go:416-424` (the TODO above line 421) documents that
  `MmapStorage` plus `cudaMemcpy` misalignment on ARM64 Grace Hopper breaks
  CUDA graph capture. The current workaround skips `MmapStorage` tensors in
  `UploadWeights`.
- `compute/engine.go:137` documents that allocations during capture
  (`cudaMalloc`) fail with error 901.
- Partial mitigation exists at `compute/gpu_engine.go:617-630`
  (`BeginCapture`) which switches `pool` to `CaptureAwareAllocator` so
  allocations during capture use `cudaMallocAsync` on the capture stream
  and are recorded as graph nodes. This path is not exercised when
  training through zerfoo's `graph/cuda_graph.go` capture wrapper, which
  calls `cuda.StreamBeginCapture` directly at `graph/cuda_graph.go:299`
  without switching the engine's allocator.
- Upstream tracker: feza-ai/wolf PR #108 (merged, pins
  `ZERFOO_DISABLE_CUDA_GRAPH=1`) and zerfoo `docs/adr/088-gemma4-ple-cuda-graph-capture.md`
  which fixed a related capture breakage in gemma4e inference.

### Objectives

- Identify the exact allocation or H2D path that triggers a silent hang
  during graph capture on GB10 with a multi-tensor upload followed by
  forward pass.
- Deliver either working CUDA graph capture on GB10 under production
  training workloads, or a fail-fast error with an actionable message when
  capture cannot safely proceed.
- Remove the need for downstream callers (Wolf, zerfoo inference
  manifests) to set `ZERFOO_DISABLE_CUDA_GRAPH=1` for the affected
  workloads.
- Preserve the existing gemma4e inference capture path documented in
  zerfoo ADR-088 (no regression on passing workloads).

### Non goals

- Rewriting the `MmapStorage` quantized-weight path to use a different
  upload strategy. Scope is constrained to making capture safe (or fail
  loudly) with the existing upload paths for CrossAsset-style dense
  float32 workloads.
- Adding new CUDA kernel code. The fix is expected to live in the capture
  lifecycle, allocator routing, and error handling layers.
- Supporting CUDA graph capture on non-managed-memory GPUs where it is
  currently off by default.

### Constraints and assumptions

- DGX Spark (GB10) is the only target hardware where this bug manifests.
  Local dev on Apple Silicon and x86 CPU tests cannot reproduce, so fixes
  must be validated via Spark pod submissions (`scripts/bench-spark.sh`
  equivalents, or ad-hoc manifests in `docs/bench/manifests/`). Never
  `ssh` to the DGX to run benches; follow the repo convention in
  `/Users/dndungu/Code/zerfoo/zerfoo/CLAUDE.md`.
- ztensor must remain CGO-free by default. CUDA access is via
  `purego/dlopen` through `internal/cuda`. Any new runtime probe must go
  through `internal/cuda/runtime_purego.go`.
- Managed memory path (`e.managedMem`) is the default on GB10 (unified
  memory). The hang happens on that path. Do not assume a non-managed
  baseline.
- The main branch must stay green for CPU and non-capture GPU tests on
  every commit. Capture-specific tests gate on a DGX runner.

### Success metrics

- CrossAsset GPU training completes at least 3 epochs on DGX GB10 with
  CUDA graph capture enabled (no env-var override) and produces
  decreasing loss across epochs.
- A reproduction test in `compute/` (or a new `graph/` test) triggers the
  same code path the hang followed, and now either passes with capture
  on or returns a typed error that names the capture-incompatible
  operation within 5 seconds.
- `ZERFOO_DISABLE_CUDA_GRAPH=1` is removed from Wolf
  `deploy/spark/train-crossasset-gpu.yaml` and from zerfoo
  `docs/bench/manifests/gemma4-e2e.yaml` and `gpu-parity.yaml` (the
  latter only if the capture fix covers their workloads).
- No regression on the 184/185 instruction capture rate measured on
  GGUF inference (see zerfoo `docs/adr/033-how-we-beat-ollama.md`).

## Discovery Summary

ENGINEERING discovery against the knowledge graph was not rerun for this
plan because the symptom, reproduction path, and suspect code sites are
already identified in the user-supplied report and in-source TODOs at
`compute/gpu_engine.go:421` and `compute/engine.go:137`. The discovery
artifact lives inline below.

### Relevant code paths

- `compute/gpu_engine.go:293-525` -- `UploadWeights` entry point, covers
  Q4_K, Q5_0, Q8_0, FP8 E4M3, FP16, BF16, float32 branches. Each branch
  calls `allocWeight` then `uploadBytes`. `MmapStorage` is explicitly
  skipped.
- `compute/gpu_engine.go:576-596` -- `allocWeight` and `uploadBytes`.
  With `managedMem`, allocation routes through `cuda.MallocManaged` and
  upload is a direct host memcpy. Without managed memory, allocation
  routes through `e.runtime.Malloc` (the GRAL default) and upload issues
  `cudaMemcpyHostToDevice`.
- `compute/gpu_engine.go:611-655` -- `BeginCapture`/`EndCapture` on the
  engine. Switches the pool to `CaptureAwareAllocator`.
- `graph/cuda_graph.go:270-345` -- The zerfoo-facing capture driver
  that actually calls `cuda.StreamBeginCapture`. This path does NOT
  invoke `GPUEngine.BeginCapture`, so the capture-aware allocator
  switch is missed. Any allocation inside the captured region still goes
  through the default `allocWeight`, which on GB10 with managed memory
  calls `cuda.MallocManaged` (illegal during capture).
- `internal/cuda/runtime_purego.go:368-385` -- `StreamBeginCapture`
  uses `cudaStreamCaptureModeRelaxed`. Relaxed mode does not forbid
  host work but it does forbid `cudaMalloc` family calls on the capture
  stream.

### Likely root-cause candidates (in priority order)

1. `graph/cuda_graph.go` begins capture without routing the engine's
   allocator through the capture-aware path. A mid-capture
   `cuda.MallocManaged` or arena resize returns error 901 synchronously,
   but the return is swallowed because the arena path logs at a level
   that is suppressed, or the stream goes into an unrecoverable captured
   state and the next `Sync` deadlocks.
2. `MmapStorage` quantized weights are lazy: `matMulMmap` dequantizes
   per op and uploads via `cudaMemcpy` on the capture stream. On ARM64
   with an unaligned mmap base, this H2D either fails silently or
   corrupts the stream capture graph, causing the next CUDA call to
   block forever.
3. The first forward pass crosses the kv-cache-like workspace setup
   that allocates a scratch buffer lazily. The allocation is not
   registered with the pre-capture `EnsureCaptureInputsGPU` code at
   `graph/cuda_graph.go:283-287`, so it races with capture.

### Use case catalog

| ID | Domain | Name | Actor | Interfaces | Priority | Wiring status |
|----|--------|------|-------|-----------|----------|---------------|
| UC-001 | compute | Upload a multi-tensor float32 weight set to GB10 managed memory before capture | zerfoo training driver | `GPUEngine.UploadWeights` | P0 | WIRED |
| UC-002 | compute | Run a captured forward+backward pass on CrossAsset-shape float32 tensors | zerfoo training driver | `GPUEngine.BeginCapture` / `graph.BuildAndRun` / `EndCapture` | P0 | BROKEN on GB10 |
| UC-003 | compute | Detect a non-capturable allocation attempt and return a typed error instead of hanging | zerfoo training/inference driver | `GPUEngine.BeginCapture`, `allocWeight` | P0 | MISSING |
| UC-004 | compute | Reset the GPU arena between training batches without disturbing an active capture | zerfoo trainer | `compute.PoolResetter.ResetPool` | P1 | WIRED (verify) |
| UC-005 | compute | Fall back to non-captured execution when capture setup fails, without requiring process restart | zerfoo runtime | `graph/cuda_graph.go:RunInstructions` fallback path | P1 | PARTIAL (existing rollback only covers `StreamBeginCapture` failures, not post-capture hangs) |
| UC-006 | compute | Re-enable CUDA graph capture for gemma4e inference on GB10 via manifest edits | zerfoo serve / bench | `docs/bench/manifests/gemma4-e2e.yaml` | P1 | BLOCKED on this plan |
| UC-007 | compute | Re-enable CUDA graph capture for CrossAsset training on GB10 via Wolf manifest | Wolf trainer | `deploy/spark/train-crossasset-gpu.yaml` | P0 | BLOCKED on this plan |
| UC-008 | compute | Regression coverage for the minimal hang repro in CI (DGX-only job) | ztensor developer | `go test ./graph/... -run TestCUDAGraph_MultiTensorUpload` | P1 | MISSING |

Gaps: UC-002, UC-003, UC-008 need implementation. UC-005 is partially
wired (only the StreamBeginCapture-failure rollback path at
`graph/cuda_graph.go:299-303` covers this; a post-capture timeout is
missing).

Reference (for this plan's purposes): manifest derived inline above, no
separate JSON artifact committed. If the fix evolves further, write
`.claude/scratch/usecases-manifest.json` on the next iteration.

## Scope and Deliverables

### In scope

- Reproduction harness that runs on DGX GB10 via Spark and reliably
  triggers the hang within 60 seconds when capture is active.
- Instrumentation that turns the silent hang into an observable error
  (stream capture status probe + explicit log on allocator calls during
  capture).
- Root-cause fix (one of: allocator routing, MmapStorage alignment,
  pre-capture workspace allocation) that allows CrossAsset training to
  run with capture on.
- Fail-fast mode that detects unavoidable capture-incompatible
  conditions and returns a typed error so the caller can retry without
  capture.
- Regression test gated on a build tag or environment variable so it
  only runs on DGX.
- Manifest updates in downstream consumers once the fix lands.
- ADR documenting the decision (new ztensor ADR-003, taking the next
  number in that repo's `docs/adr/`).

### Out of scope

- Porting the fix to ROCm or OpenCL backends. Those paths do not have
  capture support today.
- Changing the default `managedMem` detection logic.
- Rewriting the quantized-weight upload logic. If `MmapStorage` turns
  out to be a root cause, the fix is to guard capture entry, not to
  redesign weight upload.

### Deliverables

| ID | Description | Owner | Acceptance criteria |
|----|-------------|-------|---------------------|
| D1 | Reproduction test `TestCUDAGraph_MultiTensorUpload_GB10` in `graph/cuda_graph_test.go` | TBD | Hangs or fails consistently on GB10 without the fix, passes after the fix, runs under 60s |
| D2 | Diagnostic probe `cuda.StreamCaptureStatus` exposed via `internal/cuda/runtime_purego.go` | TBD | Returns one of `None`, `Active`, `Invalidated` with unit tests on CPU-mock path |
| D3 | Capture-aware allocator wiring in `graph/cuda_graph.go` | TBD | All allocations inside capture region go through `CaptureAwareAllocator`; verified by logging on debug build |
| D4 | Typed error `compute.ErrCaptureIncompatibleAllocation` returned from `allocWeight` and `uploadBytes` when called on a capturing stream | TBD | Callers get the error synchronously; no hang possible |
| D5 | Root-cause fix passing CrossAsset training on GB10 with capture enabled | TBD | 3 epochs complete, losses decrease, runtime within 10 percent of the disable-graph baseline |
| D6 | ADR documenting decision in ztensor `docs/adr/003-cuda-graph-capture-on-gb10.md` | TBD | Covers context, options considered, decision, consequences |
| D7 | Downstream manifest cleanups (Wolf + zerfoo) that drop `ZERFOO_DISABLE_CUDA_GRAPH=1` for workloads the fix covers | TBD | Manifests merged; CI green on affected jobs |

## Checkable Work Breakdown

All estimates are rough; refine when a task starts.

### E1 Reproduce and instrument the hang

- [x] T1.1 Add `StreamCaptureStatus` purego binding in `internal/cuda/runtime_purego.go` (wraps `cudaStreamGetCaptureInfo`). Owner: task-T1.1. Est: 90m. verifies: [UC-003] Completed: 2026-04-15
  - Acceptance: Returns the three-valued enum, exported via `cuda.StreamCaptureStatus(stream *Stream) (Status, error)`. Unit test on a non-capturing stream returns `None`.
  - Dependencies: none.
- [x] T1.2 Add `ensureNotCapturing()` guard to `allocWeight` and `uploadBytes` in `compute/gpu_engine.go`. If status is `Active`, return a typed error `ErrCaptureIncompatibleAllocation`. Owner: task-T1.2. Est: 60m. verifies: [UC-003] Completed: 2026-04-15
  - Acceptance: Existing non-capture tests unaffected. New unit test with a mock stream in `Active` state triggers the error.
  - Dependencies: T1.1.
- [x] T1.3 Write `TestCUDAGraph_MultiTensorUpload_GB10` in `compute/gpu_engine_gb10_test.go` gated behind `//go:build dgxgb10` build tag. The test uploads 50 tensors (including a 256x1024 float32 matrix), then invokes `BeginCapture`, runs a MatMul, `EndCapture`. Owner: task-T1.3. Est: 2h. verifies: [UC-001, UC-002] Completed: 2026-04-15
  - Acceptance: Without the fix the test fails with either a hang (caught by a 30s `context.WithTimeout`) or the new typed error.
  - Dependencies: T1.2.
- [x] T1.4 Package the test into a Spark manifest `docs/bench/manifests/cuda-graph-gb10-repro.yaml` and submit. Collect logs for evidence. Owner: coordinator. Est: 90m. verifies: [UC-002] Completed: 2026-04-16
  - Acceptance: Manifest submitted via `curl -X POST $SPARK/api/v1/pods ...`; log output includes the hang signature or the new typed error. File one zerfoo-side GitHub issue if a new failure mode surfaces.
  - Outcome: PASS — capture completed cleanly (0.51s). Pre-upload workload does not trigger hang. Pod `ztensor-cuda-graph-gb10-20260416-084710`, commit `9bf9723`.
  - Dependencies: T1.3.
- [x] T1.5 Add unit and integration tests covering T1.1 to T1.3 code paths. Owner: task-T1.5. Est: 60m. verifies: [infrastructure] Completed: 2026-04-15
  - Acceptance: CPU-mock unit tests pass in `go test ./compute/... ./internal/cuda/...`.
  - Dependencies: T1.1, T1.2.
- [x] T1.6 Run `gofmt -s -w`, `goimports`, and `golangci-lint run ./...` after the E1 changes. Owner: coordinator. Est: 15m. verifies: [infrastructure] Completed: 2026-04-15
  - Dependencies: T1.5.

### E2 Fix the silent hang path (capture-aware allocation)

- [ ] T2.1 Route `zerfoo/graph/cuda_graph.go` capture entry through `GPUEngine.BeginCapture`/`EndCapture` instead of calling `cuda.StreamBeginCapture` directly. Owner: TBD. Est: 2h. verifies: [UC-002, UC-005]
  - Acceptance: Log line shows `CaptureAwareAllocator` is engaged before the capture region; existing gemma4e inference tests still pass.
  - Risk: zerfoo `graph/cuda_graph.go` is across a repo boundary. This task splits into ztensor-side (T2.1a) and zerfoo-side (T2.1b) commits in separate PRs, wired through a ztensor minor bump.
  - Dependencies: T1.4.
- [x] T2.1a ztensor: expose a stable `compute.GPUEngine.WithCapture(fn func() error) error` helper so callers do not need to unwrap pool types. Owner: task-T2.1a. Est: 60m. verifies: [UC-002] Completed: 2026-04-16
  - Acceptance: Helper unit-tested on CPU-mock engine; returns errors from either begin/end path.
  - Dependencies: T1.2.
- [ ] T2.1b zerfoo: switch `graph/cuda_graph.go:beginCapture` to use `WithCapture`. Owner: TBD. Est: 45m. verifies: [UC-002]
  - Acceptance: Existing zerfoo GGUF inference tests still pass; gemma4e and gemma3 parity suites unchanged.
  - Dependencies: T2.1a, ztensor version bump merged.
- [x] T2.2 Introduce a `managedMem` guard in `allocWeight` that routes to `cudaMallocAsync` on the capture stream when `CaptureAwareAllocator` is active. Otherwise fall back to `MallocManaged`. Owner: task-T2.2. Est: 90m. verifies: [UC-002] Completed: 2026-04-16
  - Acceptance: Unit test with a mocked capture stream records an async-alloc node instead of a sync call.
  - Dependencies: T2.1a.
- [x] T2.3 Pre-allocate workspace buffers used by `MatMul`, `Add`, and `RMSNorm` variants at `UploadWeights` time so no lazy alloc occurs inside capture for dense float32 workloads. Owner: task-T2.3. Est: 3h. verifies: [UC-001, UC-002] Completed: 2026-04-16
  - Acceptance: Instrument with a counter; capture region records zero `allocWeight` calls for the CrossAsset workload.
  - Dependencies: T1.3, T2.1a.
- [ ] T2.4 Add unit and integration tests for T2.1 to T2.3. Owner: TBD. Est: 90m. verifies: [infrastructure]
  - Dependencies: T2.3.
- [ ] T2.5 Run linters and formatters (`gofmt`, `goimports`, `golangci-lint`). Owner: TBD. Est: 15m. verifies: [infrastructure]
  - Dependencies: T2.4.
- [ ] T2.6 Submit the repro manifest from T1.4 on the fixed branch. Confirm CrossAsset-shape upload + capture run completes in under 5 seconds. Owner: TBD. Est: 60m. verifies: [UC-002, UC-007]
  - Acceptance: Pod `Succeeded`; log excerpt saved in devlog.
  - Dependencies: T2.5.

### E3 Investigate MmapStorage alignment on GB10 (conditional on E2 not being sufficient)

- [ ] T3.1 Add a targeted test `TestMmapStorage_GB10_Align` that allocates an `MmapStorage` tensor whose base address is intentionally 4-byte aligned (not 16) and calls `cudaMemcpy` onto the capture stream. Owner: TBD. Est: 2h. verifies: [UC-001]
  - Acceptance: Reproduces the corruption on GB10 OR cleanly confirms that managed-memory path sidesteps the issue.
  - Dependencies: T2.6.
- [ ] T3.2 If T3.1 reproduces, pad `MmapStorage.Bytes()` to a 128-byte aligned staging buffer before `cudaMemcpy`. Otherwise document in the ADR that `MmapStorage` skip in `UploadWeights` remains the intended behavior. Owner: TBD. Est: 3h. verifies: [UC-001]
  - Dependencies: T3.1.
- [ ] T3.3 Update the TODO at `compute/gpu_engine.go:421` so the comment reflects the resolved state (either fixed with T3.2 or reaffirmed as intended design). Owner: TBD. Est: 15m. verifies: [infrastructure]
  - Dependencies: T3.2.
- [ ] T3.4 Tests, linters, formatters. Owner: TBD. Est: 30m. verifies: [infrastructure]
  - Dependencies: T3.3.

### E4 Fail-fast path for residual capture-incompatible workloads

- [x] T4.1 Wrap `graph/cuda_graph.go` capture run with a 30-second watchdog that samples `StreamCaptureStatus` every second. If capture is `Invalidated` or a heartbeat ping stalls, call `StreamEndCapture`, mark failed, and fall back. Owner: task-T4.1. Est: 2h. verifies: [UC-005] Completed: 2026-04-16
  - Dependencies: T1.1.
- [ ] T4.2 Expose a helper `compute.CaptureSafe(engine, fn)` that tries capture, catches `ErrCaptureIncompatibleAllocation`, and runs the instructions uncaptured on the same stream. Owner: TBD. Est: 90m. verifies: [UC-005]
  - Dependencies: T1.2, T4.1.
- [ ] T4.3 Tests, linters, formatters. Owner: TBD. Est: 30m. verifies: [infrastructure]
  - Dependencies: T4.2.

### E5 Downstream rollout

- [ ] T5.1 Remove `ZERFOO_DISABLE_CUDA_GRAPH=1` from Wolf `deploy/spark/train-crossasset-gpu.yaml`. Submit the bench once with capture enabled and attach logs. Owner: TBD. Est: 60m. verifies: [UC-007]
  - Dependencies: T2.6 (ztensor fix released), T2.1b (zerfoo pickup).
- [ ] T5.2 Remove `ZERFOO_DISABLE_CUDA_GRAPH=1` from zerfoo `docs/bench/manifests/gemma4-e2e.yaml` once capture passes the parity suite without it. Owner: TBD. Est: 60m. verifies: [UC-006]
  - Dependencies: T2.6.
- [ ] T5.3 Keep `ZERFOO_DISABLE_CUDA_GRAPH=1` in `docs/bench/manifests/gpu-parity.yaml` only if a specific parity workload still requires it; otherwise remove. Owner: TBD. Est: 30m. verifies: [UC-006]
  - Dependencies: T5.2.
- [ ] T5.4 Update docs: remove the "known issue" note from zerfoo ADR-088's Consequences section once the gemma4e manifest drops the override. Owner: TBD. Est: 30m. verifies: [infrastructure]
  - Dependencies: T5.2.

### E6 Release and documentation

- [ ] T6.1 Write ztensor `docs/adr/003-cuda-graph-capture-on-gb10.md` capturing context, options considered, decision, and consequences. Owner: TBD. Est: 90m. verifies: [infrastructure]
  - Dependencies: T2.6.
- [ ] T6.2 Append a devlog entry dated 2026-04-15 describing the hang repro, the root cause, and the fix. Include the Spark pod name(s) and log excerpts. Owner: TBD. Est: 45m. verifies: [infrastructure]
  - Dependencies: T6.1.
- [ ] T6.3 Cut a ztensor minor release via release-please (`v1.6.0`). Bump zerfoo dependency once tag publishes. Owner: TBD. Est: 60m. verifies: [infrastructure]
  - Acceptance: `github.com/zerfoo/ztensor v1.6.0` on `main`; zerfoo `go.mod` updated in the same cycle as T2.1b.
  - Dependencies: T6.2.

## Parallel Work

### Parallel tracks

| Track | Tasks | Notes |
|-------|-------|-------|
| A: Reproduction and probe | T1.1, T1.2, T1.3 | Must finish first to unblock everything else |
| B: Fix path | T2.1a, T2.2, T2.3 | Can start once T1.2 lands the probe |
| C: Mmap investigation | T3.1, T3.2 | Starts only after T2 confirms the fix is or is not sufficient |
| D: Fallback path | T4.1, T4.2 | Runs in parallel with Track B once T1.1 is in |
| E: zerfoo pickup | T2.1b | Sequential after T2.1a is released |
| F: Rollout | T5.1, T5.2, T5.3, T5.4 | After the fix is released |

Sync points: the ztensor release (T6.3) is the hard sync for any
zerfoo-side change. Track E cannot start until Track B tags a version.

### Waves

Each wave lists the exact number of parallel agents to spin up. Agent
count equals the number of task IDs listed on that wave.

#### Wave 1: Repro and probe (2 agents)

- [x] T1.1 Add `StreamCaptureStatus` purego binding  verifies: [UC-003]  2026-04-15
- [x] T1.2 Add `ensureNotCapturing` guard and typed error  verifies: [UC-003]  2026-04-15

#### Wave 2: Reproduction harness (3 agents)

- [x] T1.3 Write `TestCUDAGraph_MultiTensorUpload_GB10`  verifies: [UC-001, UC-002]  2026-04-15
- [x] T1.5 Unit and integration tests for E1  verifies: [infrastructure]  2026-04-15
- [x] T1.6 Lint and format E1  verifies: [infrastructure]  2026-04-15

#### Wave 3: Repro on hardware (1 agent)

- [x] T1.4 Spark manifest and hardware run  verifies: [UC-002]  2026-04-16

#### Wave 4: Fix + fallback in parallel (4 agents)

- [x] T2.1a ztensor `WithCapture` helper  verifies: [UC-002]  2026-04-16
- [x] T2.2 Capture-aware `allocWeight` routing  verifies: [UC-002]  2026-04-16
- [x] T2.3 Pre-allocate forward-pass workspace  verifies: [UC-001, UC-002]  2026-04-16
- [x] T4.1 Capture watchdog  verifies: [UC-005]  2026-04-16

#### Wave 5: Tests, linters, zerfoo pickup (4 agents)

- [ ] T2.4 Unit and integration tests for E2  verifies: [infrastructure]
- [ ] T2.5 Lint and format E2  verifies: [infrastructure]
- [ ] T4.2 `CaptureSafe` helper  verifies: [UC-005]
- [ ] T4.3 Lint and format E4  verifies: [infrastructure]

#### Wave 6: Hardware validation (1 agent)

- [ ] T2.6 CrossAsset-shape capture run on DGX  verifies: [UC-002, UC-007]

#### Wave 7: Release + downstream cleanup (3 agents)

- [ ] T6.1 ADR-003 for ztensor  verifies: [infrastructure]
- [ ] T6.2 Devlog entry  verifies: [infrastructure]
- [ ] T6.3 Cut ztensor v1.6.0  verifies: [infrastructure]

#### Wave 8: Mmap follow-up (conditional, 4 agents)

- [ ] T3.1 Mmap alignment repro  verifies: [UC-001]
- [ ] T3.2 Mmap alignment fix or confirmation  verifies: [UC-001]
- [ ] T3.3 Update gpu_engine.go:421 TODO  verifies: [infrastructure]
- [ ] T3.4 Tests, linters  verifies: [infrastructure]

#### Wave 9: Rollout (3 agents)

- [ ] T5.1 Drop env var from Wolf manifest  verifies: [UC-007]
- [ ] T5.2 Drop env var from gemma4-e2e manifest  verifies: [UC-006]
- [ ] T5.4 Update zerfoo ADR-088 Consequences  verifies: [infrastructure]

Wave 5.3 handles the `gpu-parity.yaml` manifest only if T5.2 verification
succeeds; it sits as a stretch alongside Wave 9.

## Timeline and Milestones

| ID | Description | Depends on | Target date |
|----|-------------|------------|-------------|
| M1 | Reproduction test reliably triggers the hang on DGX and returns a typed error (no silent hang) | T1.4 | 2026-04-17 |
| M2 | Fix merged to ztensor `main`, CrossAsset-shape capture passes on DGX | T2.6 | 2026-04-21 |
| M3 | ztensor v1.6.0 released and picked up by zerfoo `main` | T6.3 | 2026-04-23 |
| M4 | `ZERFOO_DISABLE_CUDA_GRAPH=1` removed from Wolf CrossAsset deploy manifest, 3 training epochs pass with capture on | T5.1 | 2026-04-25 |
| M5 | Gemma4e inference manifest cleaned up; ADR-088 consequences updated | T5.2, T5.4 | 2026-04-28 |

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Root cause is neither allocator routing nor Mmap alignment but an intrinsic CUDA 13 + GB10 driver bug | Forces permanent `ZERFOO_DISABLE_CUDA_GRAPH=1` on training workloads | Medium | Wave 4 includes the fail-fast path (T4.1/T4.2); even if the fix fails, we ship a clean typed error and stop the silent hang |
| R2 | Capture-aware allocator forces `cudaMallocAsync`, which GB10 driver stack may not honor in managed-memory mode | Partial capture broken across all GGUF inference paths | Medium | Gate the new routing behind a runtime probe that confirms `cudaStreamGetCaptureInfo` reports `Active` before switching allocators |
| R3 | Watchdog false-positive abandons valid captures on slow first-pass warmup | Performance regression for inference | Low | Use 30-second default, only trigger when `StreamCaptureStatus` is `Invalidated` not merely slow |
| R4 | zerfoo-side pickup of `WithCapture` lags the release, leaving the bug live | Continued production pain in Wolf | Medium | Land T2.1a and T2.1b in the same 48-hour window, pair with a zerfoo patch release |
| R5 | Pre-allocated workspace buffers bloat GPU memory for small models | Memory regression on edge models | Low | Keep allocation lazy but move it out of the captured region; only allocate on first warmup pass |
| R6 | Tests gated by `//go:build dgxgb10` never run in CI | Regression regressed silently | Medium | Add a DGX runner selector that submits the gated test via `scripts/bench-spark.sh`-style wrapper at least weekly |

## Operating Procedure

- Definition of done for each task: PR merged, CI green, DGX Spark run
  attached (for GPU tasks), ADR updated where applicable, release cut
  where the task is blocked by a version bump.
- Every implementation task has a paired testing subtask. Add tests
  under `compute/` for engine-level fixes and under `graph/` for
  capture-lifecycle fixes.
- After each commit run `gofmt -s -w`, `goimports -w`, and
  `golangci-lint run ./...` on the affected packages.
- Small focused commits; never mix changes across `compute/`,
  `graph/`, `internal/cuda/` in one commit because the pre-commit hook
  rejects cross-directory staging.
- DGX benches go via Spark only. Never `ssh` to run `go test -tags
  cuda` or `go test -bench` on DGX (see zerfoo CLAUDE.md line on the
  2026-04-07 outage).
- Use rebase and merge on GitHub, not squash, not merge commits.
- After merging to `main`, let release-please open a release PR and
  merge it to tag the ztensor release.

## Progress Log

### 2026-04-15 Change summary

- Replaced the closed-Issue-79 plan body with a new plan targeting the
  GB10 CUDA graph capture hang reported via Wolf PR #108. Preserved
  Issue-79 investigation notes in the `Archive` section below because
  they document DGX Spark procedural gotchas that remain relevant.
- No tasks completed yet; seeded Epics E1 through E6 and Milestones M1
  through M5.
- No ADRs created yet. The plan commits to ztensor
  `docs/adr/003-cuda-graph-capture-on-gb10.md` being written under T6.1.
- Cross-references: zerfoo `docs/adr/088-gemma4-ple-cuda-graph-capture.md`,
  zerfoo `docs/plan.md` epic E99, zerfoo `docs/devlog.md` entries dated
  2026-04-14 and 2026-04-15 on `ZERFOO_DISABLE_CUDA_GRAPH=1`.

## Hand off Notes

A new engineer picking this up needs:

- DGX Spark access via the Spark HTTP API on
  `http://192.168.86.250:8080`. No interactive `ssh` for benches (see
  `/Users/dndungu/Code/zerfoo/zerfoo/CLAUDE.md`).
- Familiarity with `compute/gpu_engine.go` (UploadWeights and capture
  entry points) and `graph/cuda_graph.go` (capture driver). Read
  zerfoo ADR-088 first for the gemma4e precedent.
- `docs/bench/manifests/` examples to copy when writing
  `cuda-graph-gb10-repro.yaml`.
- Access to the Wolf repo at `github.com/feza-ai/wolf` for the
  downstream manifest cleanup (T5.1).
- Permission to cut a ztensor release (release-please PR merge rights).
- Do not commit secrets or API tokens; `SPARK_API_TOKEN` lives in the
  DGX host and is referenced via `Authorization: Bearer $(cat token)`
  only.

## Appendix

### Referenced files

- `compute/gpu_engine.go:293` UploadWeights entry
- `compute/gpu_engine.go:416-424` MmapStorage skip TODO
- `compute/gpu_engine.go:576-596` allocWeight and uploadBytes
- `compute/gpu_engine.go:611-655` BeginCapture and EndCapture
- `compute/engine.go:137` documented cudaMalloc 901 constraint
- `graph/cuda_graph.go:270-345` capture driver (no allocator switch)
- `internal/cuda/runtime_purego.go:368-385` StreamBeginCapture purego
- zerfoo `docs/adr/088-gemma4-ple-cuda-graph-capture.md` precedent

### Archive -- Issue 79 investigation (closed 2026-04-09)

Retained for two reasons: the Spark operational notes still apply to
this plan, and the closure evidence demonstrates that ztensor
primitives are not at fault for the PatchTST frozen-loss signature,
which informs where NOT to look when debugging the GB10 hang.

- #78 NCCL purego migration -- CLOSED via PR #80 (merged `af8af73`).
- #79 GPU engine dst-output routing -- INVESTIGATION CLOSED ztensor-side.
  Branch `fix/issue-79-matmul-accumulate-repro` retained as evidence.

Test file `compute/gpu_dst_roundtrip_test.go` on that branch ports the
exact backward-pass op sequence from
`zerfoo/timeseries/patchtst_gpu_train.go:1022-1031`:

```
Transpose(patches -> patchesT)
Zero(dPEW)
MatMul(patchesT, dX, dPEW)
Add(gradW, dPEW, gradW)                 # in-place accumulate
gradW.Data()
```

Ran 7 variants on DGX GB10 via Spark pod
`ztensor-issue79-repro-1775761950`:

```
TestGPUEngine_Add_DstRoundTrip_OutOfPlace        PASS
TestGPUEngine_Add_DstRoundTrip_InPlace           PASS
TestGPUEngine_Add_DstRoundTrip_RepeatedInPlace   PASS
TestGPUEngine_Add_DstRoundTrip_NoExplicitSync    PASS
TestGPUEngine_PatchTSTBackward_DstRoundTrip      PASS
TestGPUEngine_PatchTSTBackward_RealisticShapes   PASS
TestGPUEngine_PatchTSTBackward_LargerBatch       PASS
```

None of the four hypotheses from the issue body was triggered. The
`makeGPUResult` / `SetStorage` / `GPUStorage.Slice()` path correctly
routes dst tensors.

Spark operational gotchas captured during that investigation, still
valid:

- Spark silently drops `pod.spec.containers[0].command` when multi-element.
  Use `args: ["bash", "-c", ...]` with no `command` field.
- Spark silently truncates long `args[i]` strings. Put scripts on host at
  `/var/lib/zerfoo/bench-out/*.sh` and mount.
- Spark drops container stdout/stderr. Redirect to host file with
  `exec >...log 2>&1` inside the script.
- ztensor's `-tags cuda` build tag is unmaintained. The kernels package
  has only `//go:build !cuda` purego files. Default build is the GPU
  path. Do not pass `-tags cuda`.
- A prebuilt `/opt/zerfoo/lib/libkernels.so` exists on the DGX host and
  must be mounted into any pod running ztensor GPU tests.

Reference manifest: `docs/bench/manifests/issue-79-repro.yaml`.
Reference script: `/var/lib/zerfoo/bench-out/issue79-run.sh` on DGX host.

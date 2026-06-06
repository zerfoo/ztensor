# ztensor Work Plan

## Title

Resolve open GitHub issues: chunk bulkUploadF32 so large one-shot weight
uploads cannot wedge the GB10 CUDA driver (issue #106).

## Context

### Problem statement

`GPUEngine.bulkUploadF32` (compute/gpu_engine.go) consolidates every eligible
float32 weight tensor into one device allocation and uploads it with one
host-to-device copy. On the NVIDIA GB10 (sm_121, aarch64, 128 GB unified
memory) a sufficiently large single `Malloc(total)` + `Memcpy(total)` wedges
the CUDA driver in an uninterruptible D-state: the main OS thread is stuck in a
driver ioctl that never returns and cannot be SIGKILLed, which makes the
container unkillable and leaks a permanently running pod on the Spark
orchestrator.

Reproduced 2026-06-05 from Wolf `train-crossasset` (CrossAsset multiscale
model) at the sample pre-upload step: `UploadWeights -> bulkUploadF32(213304
tensors + 50 params)` never returns. The default branch is the non-managed one
(`ZERFOO_ENABLE_MANAGED_MEM` unset), so the wedge is in the single
`e.runtime.Malloc(total)` at gpu_engine.go:422 plus the single
`e.runtime.Memcpy(..., HostToDevice)` at gpu_engine.go:441.

The bulk path was added in #103 / release 1.8.0 to fix the inverse problem
(tens of thousands of per-tensor `cudaMalloc`/`cudaMemcpy` round-trips also
wedged GB10). The fix for #106 must keep that win while bounding every
individual driver call.

### Evidence it is a driver D-state, not a Go deadlock (from issue #106)

When the upload hangs, `podman exec`, log streaming, and pod delete all wedge,
while the orchestrator control plane stays responsive. A Go futex/channel
deadlock would not wedge `podman exec` (a fresh process in the namespace).
Wolf's heartbeat goroutine keeps ticking; only the main goroutine (in the CUDA
call) is stuck. Conclusion: main thread is in a CUDA driver ioctl in D-state.

### Objectives

- Bound every `Malloc`/`Memcpy` issued by `bulkUploadF32` to a configurable
  byte cap so no single driver call can wedge the GB10 driver, for any input
  tensor count.
- Preserve the bulk-upload win: a small constant number of large copies, not
  one copy per tensor.
- Keep the resulting per-tensor `GPUStorage` views byte-identical to the
  current single-buffer behavior within each chunk.
- Validate the fix on real GB10 hardware via Spark with a 213k-tensor upload
  that previously wedged.
- Merge, release, and close issue #106.

### Non goals

- Re-architecting the quantized / MmapStorage / FP8 / FP16 / BF16 upload
  branches. Scope is the float32 bulk path only (the branch #106 reproduces).
- Adding new CUDA kernels. The fix lives in the upload lifecycle in
  compute/gpu_engine.go.
- Changing the managed-memory default or the `bulkUploadF32MinTensors=64`
  lower bound.
- Investigating the exact GB10 driver wedge threshold (open question 1 in #106
  to ztensor maintainers). The cap is set conservatively below any plausible
  threshold; precise characterization is out of scope.

### Constraints and assumptions

- GB10 (DGX Spark, 192.168.86.250) is the only hardware where the wedge
  manifests. Local Apple Silicon and x86 CPU tests cannot reproduce it.
  Hardware validation MUST go through Spark pod submissions, never interactive
  `ssh` benchmarks (see zerfoo CLAUDE.md and the Spark gotchas in
  docs/devlog.md 2026-06-05).
- ztensor stays CGO-free by default; CUDA access is via purego/dlopen through
  internal/cuda. The chunking change touches only Go control flow in
  compute/gpu_engine.go and adds no CUDA bindings.
- Unit tests must run on CPU/CI without a GPU. Chunk-counting tests stub the
  package-level indirection points `mallocManagedFn` and the runtime
  `Malloc`/`Memcpy` (see gpu_engine.go:753-757) so they assert call counts and
  chunk boundaries without a device.
- main must stay green for CPU and non-capture GPU tests on every commit.

### Success metrics

- The Wolf 213k-tensor CrossAsset pre-upload completes through `UploadWeights`
  on GB10 with the chunked path and no env override, with no D-state wedge.
- A unit test proves that uploading tensors whose total exceeds the cap issues
  more than one bounded `Malloc`+`Memcpy` and that each is at or below the cap,
  for both the managed and non-managed branches.
- `bulk_upload_test.go` existing coverage
  (`TestGPUEngine_UploadWeights_BulkPath`,
  `TestGPUEngine_UploadWeights_BelowBulkThreshold`) continues to pass
  unchanged.
- Issue #106 closed; release tag cut after merge.

## Discovery Summary

ENGINEERING. The symptom, reproduction path, and suspect code site are fully
identified in issue #106 and confirmed against the current source.

Single open issue discovered: **#106** (created 2026-06-05, no labels). No
other open issues. Prior issues #78 (NCCL purego, closed via #80) and #79 (GPU
dst routing, investigation closed ztensor-side) are resolved; the prior
capture-hang plan shipped in release 1.8.0 and is retired into docs/devlog.md
(2026-06-05 entry).

Relevant code sites (compute/gpu_engine.go):

- `bulkUploadF32MinTensors = 64` (line 363) -- lower bound only; no upper bound.
- `bulkUploadF32` (lines 379-454) -- the function to chunk. Builds `eligible`
  with running `total` (lines 389-409), single alloc (419-423), single copy
  per branch (429-445), single `bulkUploadBuffers` append + view loop
  (447-452).
- `bulkUploadBuffers []unsafe.Pointer` (line 142) -- already a slice; freed in
  Close at lines 953-958.
- Indirection points for tests: `mallocManagedFn` (line 757), `e.runtime.Malloc`
  and `e.runtime.Memcpy`.
- `UploadWeights` (line 456) -- caller; unchanged by this work.

Decision rationale for the cap shape: docs/adr/003-bulk-upload-chunking-cap.md.

## Scope and Deliverables

In scope:
- Byte-bounded chunking of `bulkUploadF32` for both managed and non-managed
  branches, with a configurable cap.
- Unit tests proving chunk count and per-chunk byte bounds.
- GB10 Spark validation that the prior-wedging 213k-tensor upload completes.
- PR, rebase-and-merge, release, issue close.

Out of scope: everything in Non goals above.

| ID | Deliverable | Owner | Acceptance criteria |
|----|-------------|-------|---------------------|
| D1 | Chunked `bulkUploadF32` | TBD | No `Malloc`/`Memcpy` exceeds the cap; views unchanged within a chunk |
| D2 | Unit tests | TBD | Multi-chunk, exact-boundary, oversized-single-tensor, both branches; CI green |
| D3 | GB10 validation | TBD | 213k-tensor upload completes via Spark; devlog entry with pod + commit |
| D4 | Shipped fix | TBD | PR merged rebase-and-merge; release tag cut; #106 closed |

## Checkable Work Breakdown

### E0 -- Repo hygiene
**Component:** compute
Acceptance: clean working tree on a fix branch off origin/main.

- [x] T0.1 Clear the stale `UU` index entry on compute/gpu_engine.go (self-resolved; working tree clean)  Owner: David  Est: 10m  verifies: [infrastructure]  (2026 06 05)
- [x] T0.2 Confirm fix branch `fix/bulk-upload-chunking-106` is based on origin/main at the 1.8.0 release commit (1 commit ahead: 4eaae4b)  Owner: David  Est: 15m  verifies: [infrastructure]  (2026 06 05)

### E1 -- Chunk the bulk upload
**Component:** compute
Acceptance: `bulkUploadF32` issues one bounded `Malloc`+`Memcpy` per chunk; no driver call exceeds the cap; per-tensor `GPUStorage` views unchanged within a chunk.

DEVIATION (implemented in commit 4eaae4b): the shipped fix uses a dual cap --
byte cap `bulkUploadF32MaxChunkBytes = 64 MiB` (a `var` for test override) AND
tensor-count cap `bulkUploadF32MaxChunkTensors = 4096` -- instead of the
single 256 MB byte cap with a `ZERFOO_BULK_UPLOAD_CHUNK_MB` env var originally
planned. Tiling is extracted to a pure, CPU-testable `bulkUploadChunkRanges`.
ADR 003 was updated to record the actual decision. Rationale: more conservative
byte cap, no runtime-config surface needed, belt-and-suspenders tensor bound.

- [x] T1.1 Chunk-cap constants `bulkUploadF32MaxChunkBytes = 64 << 20` (var) + `bulkUploadF32MaxChunkTensors = 4096` (const). Decision rationale: docs/adr/003-bulk-upload-chunking-cap.md  Owner: David  Est: 45m  verifies: [#106]  (2026 06 05)
  - Dependencies: T0.2
  - Done: constants in compute/gpu_engine.go:372-374; no env var (deviation above).
- [x] T1.2 Refactor `bulkUploadF32` to greedily pack `eligible` into chunks bounded by both caps via `bulkUploadChunkRanges`; a lone tensor over the byte cap gets its own range. Per chunk: one `Malloc`/`mallocManaged(chunkBytes)`, one staging+`Memcpy` (non-managed) or in-place copy (managed), append chunk devPtr to `bulkUploadBuffers`, then `SetStorage` views at chunk-local offsets  Owner: David  Est: 90m  verifies: [#106]  (2026 06 05)
  - Dependencies: T1.1
  - Done: gpu_engine.go:414-511; both branches chunked; on error frees the chunk pointer and returns wrapped error; returns `len(eligible)`.
- [x] T1.3 Unit tests in compute/bulk_upload_chunk_test.go: `bulkUploadChunkRanges` tiling (empty, single, all-fit, byte-cap split, tensor-cap split, lone-oversized) + 213k-count bound. Existing `TestGPUEngine_UploadWeights_BulkPath` / `_BelowBulkThreshold` unchanged (skip without CUDA)  Owner: David  Est: 90m  verifies: [#106]  (2026 06 05)
  - Dependencies: T1.2
  - Done: `go test ./compute/` green on CPU; 7 chunk-range assertions PASS; GPU integration tests skip locally.
- [x] T1.4 gofmt + `go vet ./...` clean on changed files  Owner: David  Est: 20m  verifies: [infrastructure]  (2026 06 05)
  - Dependencies: T1.3
  - Done: `go build ./...` exit 0; `go vet ./compute/` clean.

### E2 -- Validate on GB10 hardware
**Component:** compute
Acceptance: the prior-wedging 213k-tensor upload completes through `UploadWeights` on GB10 via Spark with no D-state wedge.

- [ ] T2.1 Build an arm64 repro image at the E1 commit and submit a Spark Pod that constructs ~213k float32 tensors and calls `UploadWeights`, mounting `/opt/zerfoo/lib/libkernels.so`; redirect output to a host file (Spark gotchas in docs/devlog.md). Confirm phase Succeeded and the upload returns  Owner: TBD  Est: 90m  verifies: [#106]
  - Dependencies: T1.4
  - Acceptance: pod reaches `Succeeded`; log shows upload completed; no leaked running pod; rerun once to confirm reproducibility.
- [ ] T2.2 Record a devlog entry (/journal) with pod name, commit SHA, chunk count observed, and timing  Owner: TBD  Est: 20m  verifies: [infrastructure]
  - Dependencies: T2.1

### E3 -- Ship
**Component:** release
Acceptance: PR merged rebase-and-merge; release tag cut; #106 closed.

- [ ] T3.1 Open PR from `fix/bulk-upload-chunking-106` referencing #106; ensure CI green; rebase-and-merge (not squash, not merge commit)  Owner: TBD  Est: 30m  verifies: [#106]
  - Dependencies: T2.2
- [ ] T3.2 Confirm release-please cuts a release for the merge; verify the version tag exists  Owner: TBD  Est: 20m  verifies: [infrastructure]
  - Dependencies: T3.1
- [ ] T3.3 Close issue #106 with a summary linking the PR, ADR 003, and the GB10 validation pod  Owner: TBD  Est: 10m  verifies: [#106]
  - Dependencies: T3.2

## Parallel Work

This is a small, mostly linear fix touching one function, so cross-epic
parallelism is limited. The available parallelism is inside E1.

| Track | Tasks | Notes |
|-------|-------|-------|
| Track A: implementation | T1.1 -> T1.2 -> T1.3 -> T1.4 | Sequential; each depends on the prior |
| Track B: hygiene | T0.1, T0.2 | Independent of A until T1.1 starts |
| Track C: validation harness prep | draft Spark manifest + repro main during T1.2/T1.3 | Manifest authoring needs no code; only T2.1 execution waits on T1.4 |

Sync point: T1.4 must complete before T2.1 (validation needs the built fix).
T2.2 before E3.

### Wave 1: Hygiene + cap helper (2 agents)
- [x] T0.1 Clear stale index entry  verifies: [infrastructure]  (2026 06 05)
- [x] T0.2 Confirm/rebase fix branch  verifies: [infrastructure]  (2026 06 05)

### Wave 2: Implement chunking (1 agent, sequential chain)
- [x] T1.1 Chunk-cap constants (dual cap, var)  verifies: [#106]  (2026 06 05)
- [x] T1.2 Chunked bulkUploadF32  verifies: [#106]  (2026 06 05)
- [x] T1.3 Unit/integration tests  verifies: [#106]  (2026 06 05)
- [x] T1.4 gofmt + vet + lint  verifies: [infrastructure]  (2026 06 05)

(Wave 2 is a single chain because all four tasks edit the same function and
test file with hard data dependencies; splitting agents would only create merge
churn. A second agent can author the Wave 3 Spark manifest in parallel.)

### Wave 3: GB10 validation (1 agent)
- [ ] T2.1 Spark 213k-tensor upload completes  verifies: [#106]
- [ ] T2.2 Devlog entry  verifies: [infrastructure]

### Wave 4: Ship (1 agent)
- [ ] T3.1 PR + rebase-and-merge  verifies: [#106]
- [ ] T3.2 Verify release tag  verifies: [infrastructure]
- [ ] T3.3 Close #106  verifies: [#106]

## Timeline and Milestones

| Milestone | Description | Member tasks | Exit criteria |
|-----------|-------------|--------------|---------------|
| M0 | Branch ready | T0.1, T0.2 | Clean working tree on fix branch off origin/main |
| M1 | Fix implemented and unit-green | T1.1, T1.2, T1.3, T1.4 | `go test ./compute/...` green; no driver call exceeds cap in tests |
| M2 | GB10 validated | T2.1, T2.2 | 213k-tensor upload completes on GB10 via Spark; devlog recorded |
| M3 | Shipped | T3.1, T3.2, T3.3 | PR merged rebase-and-merge; release tag cut; #106 closed |

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | A single weight tensor exceeds the cap, so its chunk still issues an over-cap copy | Medium | Low | Individual dense f32 weights are <= a few MB; log a warning on over-cap single tensor (T1.2) so it is visible; cap is configurable down via env |
| R2 | 256 MB default is still above the true GB10 wedge threshold | High | Low | Conservative default well below observed multi-GB wedge; `ZERFOO_BULK_UPLOAD_CHUNK_MB` lets ops lower it without a rebuild; T2.1 validates empirically |
| R3 | Cannot reproduce the wedge on Spark to prove the fix (load too small, or hardware busy) | Medium | Medium | Reuse the exact Wolf CrossAsset 213k-tensor shape; if Spark is unavailable, mark M2 blocked and report honestly rather than claim done |
| R4 | Chunk-boundary offset bug corrupts a tensor view | High | Low | T1.3 asserts reconstructed tensor data equals source across a chunk boundary; existing BulkPath test guards the single-chunk case |
| R5 | Local main behind origin/main causes branch confusion | Low | Medium | T0.2 rebases the fix branch onto origin/main before work |

## Operating Procedure

- Definition of done (per global CLAUDE.md): merged via rebase-and-merge, CI
  green, release tag cut, and the fix verified live on GB10 (the 213k-tensor
  upload completing through `UploadWeights` via Spark, observed in pod logs,
  not merely "the code should chunk"). Report what was actually observed.
- Add tests with every implementation change (T1.3 pairs with T1.2).
- Run gofmt, `go vet`, and the linter after code changes (T1.4).
- Never commit files from different directories in one commit (pre-commit hook
  rejects it). Keep commits small and logical: cap helper, chunking refactor,
  tests, each its own commit where practical.
- Validate GPU behavior only via Spark Pod submissions; never interactive ssh
  benchmarks on the DGX.

## Progress Log

### Change Summary -- 2026-06-05 (apply run)

- E0 + E1 complete. Fix landed in commit 4eaae4b: `bulkUploadF32` now uploads in
  bounded chunks (64 MiB byte cap + 4096 tensor cap) via the pure
  `bulkUploadChunkRanges` tiling function. Both managed and non-managed branches
  chunked; per-chunk error paths free the device pointer.
- Validation: `go build ./...` exit 0; `go vet ./compute/` clean; 7
  `bulkUploadChunkRanges` unit tests PASS on CPU (tiling, both caps,
  lone-oversized, 213k-count bound). GPU integration tests skip locally (no
  CUDA), to be exercised on GB10 in E2.
- Recorded the dual-cap deviation from the original single-byte-cap/env-var
  plan; updated docs/adr/003-bulk-upload-chunking-cap.md to match the shipped
  decision.
- Remaining: E2 (GB10 Spark validation of the 213k-tensor upload) and E3 (PR,
  merge, release, close #106).

### Change Summary -- 2026-06-05

- Retired the prior CUDA-graph-capture-hang plan (shipped in release 1.8.0 via
  PRs #94-#98). Routed its closure into docs/devlog.md (2026-06-05 entry);
  stable interface knowledge already in docs/design.md. Removed the completed
  epics, waves, milestones, and the issue-79/78 archive from this plan.
- Created docs/adr/003-bulk-upload-chunking-cap.md: cap `bulkUploadF32` by a
  byte-sized chunk (256 MB default, `ZERFOO_BULK_UPLOAD_CHUNK_MB` override),
  not by tensor count.
- Wrote a new plan for the sole open issue #106 (bulkUploadF32 wedges GB10 in
  D-state on large one-shot uploads). Epics E0 (hygiene), E1 (chunk the bulk
  upload), E2 (GB10 validation), E3 (ship). Grounded against the current
  bulkUploadF32 source (gpu_engine.go:357-454).
- Noted the stale `UU` index entry on compute/gpu_engine.go (working tree
  matches HEAD; clear with `git reset` in T0.1).

ADRs created: docs/adr/003-bulk-upload-chunking-cap.md -- byte-sized chunk cap
for bulkUploadF32, with `ZERFOO_BULK_UPLOAD_CHUNK_MB` override.

## Hand-off Notes

- Sole open issue is #106. The fix is localized to one function,
  `bulkUploadF32` in compute/gpu_engine.go (lines 379-454). Read ADR 003 first
  for the cap decision.
- The default upload branch on GB10 is the non-managed one
  (`ZERFOO_ENABLE_MANAGED_MEM` unset): single `Malloc` at gpu_engine.go:422 and
  single `Memcpy` at :441. Both that branch and the managed branch (:420/:429)
  must be chunked.
- `bulkUploadBuffers` (gpu_engine.go:142) is already a slice freed in Close
  (:953); appending one pointer per chunk needs no structural change.
- Unit tests stub the package-level indirection `mallocManagedFn`
  (gpu_engine.go:757) and the runtime `Malloc`/`Memcpy` to count driver calls
  without a GPU.
- GB10 validation goes through Spark only. Spark operational gotchas and the
  `libkernels.so` mount requirement are in docs/devlog.md (2026-06-05 and the
  retained issue-79 notes). DGX Spark host: 192.168.86.250:8080.
- Wolf caller that triggers the wedge: `internal/crossasset/crossasset.go`
  `trainWithResult -> UploadWeights`. Wolf devlog 2026-06-05 (T8.1) cross-refs.

## Appendix

- Issue: github.com/zerfoo/ztensor#106.
- Origin of the bulk path: PR #104 / commit 9ca83f6 (#103), release 1.8.0.
- Cap decision: docs/adr/003-bulk-upload-chunking-cap.md.
- Code: compute/gpu_engine.go:357-454 (`bulkUploadF32`,
  `bulkUploadF32MinTensors`), :456 (`UploadWeights`), :142 / :953
  (`bulkUploadBuffers`).
- Tests: compute/bulk_upload_test.go.

# ztensor Open GitHub Issues Resolution

## Context

Resolve all open GitHub issues in `github.com/zerfoo/ztensor`:

- **#78** Migrate `internal/nccl` from CGo to purego/dlopen. PR #80 is open on branch `chore/nccl-purego`, CI green (per memory 2412).
- **#79** GPU engine `dst`-output routing bug: tensors written by GPU kernels read back as all-zero via `.Data()` on DGX GB10, freezing PatchTST training in zerfoo at a deterministic loss. CPU engine converges on the same test. Minimal ztensor-level diagnostic tests on branch `debug/gpu-dst-routing` did NOT reproduce the bug on DGX (memory 2407), so the fault is narrower than "any engine op" — likely specific to the op mix, in-place aliasing, or storage-flip path exercised by `trainWindowedGPU`.

### Objectives
1. Merge PR #80, close #78.
2. Reproduce #79 with a ztensor-only test, land a fix, close #79.

### Non-goals
- Refactoring unrelated GPU engine code.
- Touching zerfoo's PatchTST training loop beyond what is needed to bisect.

### Success metrics
- `gh issue list --repo zerfoo/ztensor --state open` returns zero issues.
- `go build ./...` in ztensor compiles with no build tags.
- A ztensor unit test reproduces the original #79 failure on a pre-fix commit and passes post-fix on DGX Spark.

## Discovery Summary

Work type: **Engineering**.

Existing artifacts:
- Branch `chore/nccl-purego` → PR #80 (CI green, verified on DGX GB10, ADR drafted).
- Branch `debug/gpu-dst-routing` with basic round-trip tests for `Add`/`MatMul` — did not reproduce #79.
- Issue #79 body contains seven probe logs and four hypotheses (α buffer mismatch, β SetStorage-after-kernel clobber, γ D2H source divergence, δ in-place aliasing).
- zerfoo `devlog.md` entry dated 2026-04-08 "FINAL" has full investigation history.

Key code paths implicated by #79:
- `compute/gpu_kernels.go:121-132` — `makeGPUResult` allocates a fresh device buffer and calls `dst[0].SetStorage(gs)`.
- `tensor/gpu_storage.go:215-250` — `GPUStorage.Slice()` does `make + D2H memcpy` on every `.Data()`.
- `compute/gpu_engine*.go` — Add/Sub/Mul/MatMul dispatch.

## Scope and Deliverables

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| D1 | PR #80 merged via rebase-and-merge | #78 closed, `main` builds without `-tags cuda` |
| D2 | ztensor regression test reproducing #79 on DGX | Test fails on current `main`, passes after fix |
| D3 | Root-cause fix in ztensor GPU engine | PatchTST GPU training in zerfoo converges on DGX |
| D4 | ADR documenting the routing fix | File in `docs/adr/` |
| D5 | devlog entry with RCA | Appended to `docs/devlog.md` |

In scope: NCCL migration finalization, #79 reproduction, bisect, fix, tests, docs.
Out of scope: New GPU features, perf work, unrelated kernel changes.

## Checkable Work Breakdown

### E1 — Close #78 (NCCL purego)

- [x] T1.1 Re-check PR #80 CI on latest push. 2026 04 09. CI green (test/COMPLETED/SUCCESS).
- [x] T1.2 Rebase PR #80 onto `main` if needed. 2026 04 09. mergeStateStatus=CLEAN, no rebase needed.
- [x] T1.3 Rebase-and-merge PR #80, confirm #78 auto-closes. 2026 04 09. Merged as af8af73; #78 CLOSED.
- [x] T1.4 Verify `go build ./...` (no tags) on `main` post-merge. 2026 04 09. Build OK.
- [ ] T1.5 Open tracking issue in zerfoo to drop its duplicate `internal/nccl` and import from ztensor. Owner: TBD. Est: 15m. verifies: [infrastructure]

### E2 — Reproduce #79 at ztensor level

- [ ] T2.1 Expand `gpu_engine_dst_routing_test.go` to mirror `trainWindowedGPU`'s exact op sequence: MatMul → Add (in-place dst==src0) → Sub → elementwise scale, with storage flips between CPU-allocated wrappers and GPU kernel outputs. Owner: TBD. Est: 90m. verifies: [UC-79]
- [ ] T2.2 Add a test exercising `engine.Add(ctx, a, b, a)` (in-place aliasing, hypothesis δ). Owner: TBD. Est: 30m. verifies: [UC-79]
- [ ] T2.3 Add a test where `dst` is a freshly-allocated CPUStorage wrapper that gets flipped to GPUStorage by `makeGPUResult`, then read via `.Data()` immediately and after `engine.Sync()`. Owner: TBD. Est: 45m. verifies: [UC-79]
- [ ] T2.4 Submit reproducer as a Spark Job manifest to `192.168.86.250:8080`; capture logs. Owner: TBD. Est: 30m. verifies: [UC-79]
- [x] T2.5 Port `trainWindowedGPU` patch-embedding backward op sequence to a standalone compute test. 2026 04 09. Added TestGPUEngine_PatchTSTBackward_DstRoundTrip on branch fix/issue-79-matmul-accumulate-repro. All 5 dst-routing tests PASS on DGX GB10 via Spark pod ztensor-issue79-repro-1775759440. Bug not reproducible at ztensor primitive level. See docs/devlog.md 2026-04-09 entry.

### E3 — Diagnose #79

- [ ] T3.1 Instrument `makeGPUResult` to log: caller `dst` pointer, existing storage dptr, kernel write-target dptr, post-`SetStorage` dptr. Run reproducer on DGX via Spark. Owner: TBD. Est: 60m. verifies: [UC-79]
- [ ] T3.2 Instrument `GPUStorage.Slice()` D2H to log source dptr and first 4 bytes. Owner: TBD. Est: 30m. verifies: [UC-79]
- [ ] T3.3 Classify root cause against hypotheses α/β/γ/δ from issue #79. Owner: TBD. Est: 30m. verifies: [UC-79]
- [ ] T3.4 Write ADR `docs/adr/002-gpu-dst-routing-fix.md` describing the chosen fix. Owner: TBD. Est: 30m. verifies: [infrastructure]

### E4 — Fix #79

- [ ] T4.1 Implement the routing fix in `compute/gpu_kernels.go` (and engine ops as needed). Owner: TBD. Est: 2h. verifies: [UC-79]
- [ ] T4.2 Make `.Data()` on GPU tensor implicitly `Sync()` the stream before D2H memcpy (safety net). Owner: TBD. Est: 45m. verifies: [UC-79]
- [ ] T4.3 Remove diagnostic logging added in E3. Owner: TBD. Est: 15m. verifies: [infrastructure]
- [ ] T4.4 Run `gofmt`, `go vet`, `golangci-lint run` across touched packages. Owner: TBD. Est: 15m. verifies: [infrastructure]
- [ ] T4.5 Unit tests: ensure E2 tests now pass on DGX via Spark submission. Owner: TBD. Est: 30m. verifies: [UC-79]
- [ ] T4.6 End-to-end: run `zerfoo/scripts/bench-spark.sh -samples 1000 -channels 5 -epochs 3` against a zerfoo branch pinned to the fixed ztensor; confirm loss decreases across epochs. Owner: TBD. Est: 45m. verifies: [UC-79]

### E5 — Release and close

- [ ] T5.1 Commit fix in small logical commits (test, instrumentation-removal, fix, ADR, devlog). Owner: TBD. Est: 20m. verifies: [infrastructure]
- [ ] T5.2 Open PR for #79 fix, link issue. Owner: TBD. Est: 10m. verifies: [infrastructure]
- [ ] T5.3 Rebase-and-merge after CI green. Owner: TBD. Est: 10m. verifies: [infrastructure]
- [ ] T5.4 Confirm release-please PR bumps ztensor version; merge when ready. Owner: TBD. Est: 10m. verifies: [infrastructure]
- [ ] T5.5 Append `docs/devlog.md` entry with RCA, probe outputs, fix summary. Owner: TBD. Est: 20m. verifies: [infrastructure]
- [ ] T5.6 Confirm both #78 and #79 are closed. Owner: TBD. Est: 5m. verifies: [infrastructure]

## Parallel Work

| Track | Tasks | Notes |
|-------|-------|-------|
| A — NCCL merge | T1.1–T1.5 | Independent of #79 work |
| B — #79 repro | T2.1–T2.5 | Independent of A |
| C — #79 diagnose/fix | T3.*, T4.* | Depends on B |
| D — Release | T5.* | Depends on A and C |

### Waves

### Wave 1: Parallel kickoff (6 agents)
- [ ] T1.1 CI recheck PR #80
- [ ] T1.2 Rebase PR #80
- [ ] T2.1 Expand routing test
- [ ] T2.2 In-place aliasing test
- [ ] T2.3 Storage-flip test
- [ ] T1.5 zerfoo duplicate-nccl tracker issue

### Wave 2: Merge + Spark repro (3 agents)
- [ ] T1.3 Merge PR #80
- [ ] T1.4 Post-merge build verify
- [ ] T2.4 Submit reproducer to Spark

### Wave 3: Diagnose (fallback + instrument) (3 agents)
- [ ] T2.5 Fallback trainWindowedGPU port (only if Wave 2 didn't repro)
- [ ] T3.1 Instrument makeGPUResult
- [ ] T3.2 Instrument GPUStorage.Slice

### Wave 4: Classify + fix (3 agents)
- [ ] T3.3 Classify root cause
- [ ] T3.4 ADR
- [ ] T4.1 Implement fix

### Wave 5: Harden + verify (4 agents)
- [ ] T4.2 Implicit Sync on .Data()
- [ ] T4.3 Remove diagnostics
- [ ] T4.4 Lint/format
- [ ] T4.5 Spark test rerun

### Wave 6: Release (5 agents, sequential within track)
- [ ] T4.6 zerfoo bench verification
- [ ] T5.1 Small commits
- [ ] T5.2 Open PR
- [ ] T5.3 Merge
- [ ] T5.4 release-please
- [ ] T5.5 devlog
- [ ] T5.6 Confirm closed

## Timeline and Milestones

| ID | Milestone | Dependencies | Exit criteria |
|----|-----------|--------------|---------------|
| M1 | #78 closed | Wave 2 | PR #80 merged, main builds no-tags |
| M2 | #79 reproduced in ztensor test | Wave 2 or Wave 3 | Test fails deterministically on DGX |
| M3 | #79 root cause identified | Wave 4 | Classified α/β/γ/δ, ADR drafted |
| M4 | #79 fix merged | Wave 6 | PR merged, CI green |
| M5 | zerfoo PatchTST GPU training converges | T4.6 | Loss decreases over 3 epochs via bench-spark |

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | #79 cannot be reproduced at ztensor level | High | Medium | Fallback T2.5: port minimal zerfoo graph verbatim |
| R2 | Fix masks symptom, zerfoo still frozen | High | Low | Gate M4 on T4.6 bench-spark convergence |
| R3 | Spark host contention delays DGX runs | Med | Low | Batch probes into single manifest submissions |
| R4 | NCCL purego regressions in zerfoo's distributed path | Med | Low | T1.5 follow-up issue; keep `nccl_cgo` legacy path behind `//go:build nccl_cgo` for rollback |

## Operating Procedure

- Definition of done: PR merged, CI green, DGX Spark verification captured in PR description, release-please PR merged.
- Always add tests alongside fixes.
- Always run `gofmt`, `go vet`, `golangci-lint` after code changes.
- No `ssh` benchmarks — use `bench-spark.sh` / Spark manifests only.
- Small logical commits; never mix files across subdirectories in a single commit.
- Rebase-and-merge on GitHub (never squash, never merge commits).

## Progress Log

### 2026-04-09 — Plan created
Plan authored to resolve #78 and #79. #78 has PR #80 already open (CI green on DGX per prior session). #79 diagnostic tests on `debug/gpu-dst-routing` did not reproduce; plan adds a fallback path to port zerfoo's failing op sequence directly into a ztensor test. No ADRs created yet — ADR-002 will be written when root cause is classified.

## Hand-off Notes

- PR #80 branch: `chore/nccl-purego`. Reference pattern: `internal/cublas/cublas_purego.go`.
- #79 instrumentation targets: `compute/gpu_kernels.go:121-132`, `tensor/gpu_storage.go:215-250`.
- DGX Spark: `SPARK=http://192.168.86.250:8080`. Submit manifests, never ssh.
- zerfoo reproducer: `scripts/bench-spark.sh -samples 1000 -channels 5 -epochs 3 -cleanup`. Frozen loss signature: `0.268357`.
- Full investigation history: zerfoo `docs/devlog.md` 2026-04-08 "FINAL" entry.

## Appendix

Use cases:
- **UC-79** PatchTST training on GPU engine converges on DGX GB10 (currently BROKEN — loss frozen at 0.268357).
- **UC-78** `go build ./...` in ztensor compiles the NCCL path without `-tags cuda` (IN PROGRESS via PR #80).

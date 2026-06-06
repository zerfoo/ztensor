# ztensor session updates

## 2026-06-05 -- Resolve open GitHub issues (#106)

Plan: docs/plan.md. Sole open issue: #106 (bulkUploadF32 wedges GB10 driver).

Status:
- E0 hygiene: DONE.
- E1 chunk bulkUploadF32: DONE (commit 4eaae4b). Dual cap 64 MiB + 4096 tensors.
- E2 GB10 validation: DONE. TestGPUEngine_UploadWeights_MultiChunk PASSED on
  GB10 (Spark pod ...guard-3c04539, exit-0 guard). 256 MiB -> 4x 64 MiB chunks,
  no wedge, cross-chunk views round-trip.
- PR #107: CI green; merging now.
- E3 ship: in progress (rebase-and-merge, release, close #106).

# ztensor session updates

## 2026-06-05 -- Resolve open GitHub issues (#106) -- COMPLETE

Sole open issue #106 (bulkUploadF32 wedges GB10 driver) is RESOLVED and SHIPPED.

- E1 fix: chunked bulkUploadF32 (64 MiB + 4096-tensor dual cap). ADR 003.
- E2 validation: TestGPUEngine_UploadWeights_MultiChunk PASSED on GB10 (Spark
  pod ...guard-3c04539). 256 MiB -> 4x 64 MiB chunks, no wedge, views round-trip.
- E3 ship: PR #107 rebase-merged; release-please cut v1.8.1 (PR #108);
  issue #106 auto-closed.

No open ztensor GitHub issues remain.

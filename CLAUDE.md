# ztensor — tensor, compute, and graph core

ztensor is the tensor/compute/autograd substrate for the Zerfoo ecosystem
(github.com/zerfoo/zerfoo and its consumers, including feza-ai/wolf).

## ztensor stays general-purpose

ztensor is general-purpose infrastructure. Wolf is its most demanding consumer
and best stress test — a **workload, not the spec**. Rules:

- **Contracts over special cases.** When a consumer exposes a bug, fix the
  general contract: the SaveForBackward/arena-pinning lifetime contract (ADR
  006), "dst-form ops write into dst's storage and never re-home it", poison
  mode for use-after-reset detection. Never branch on or hard-code a
  consumer's call pattern.
- **The arena's safety model is the contract, not the caller's discipline.**
  Buffer reuse must be safe under any reset cadence a caller chooses
  (per-sample, per-step, never). Tensors that must survive resets do so via
  the pinning/persistent-storage mechanisms, not via assumptions about who
  resets when.
- **Universal quality gates.** Every graph node and CUDA kernel — used by any
  consumer or none — must pass gradcheck/OpInfo (testing/gradcheck), the
  CPU-vs-GPU parity harness under arena-stress schedules (testing/parity),
  and the PyTorch-oracle gate (testing/oracle) before merge.
- **Track single-consumer deferrals as issues.** Known holes deferred because
  the current consumer set doesn't hit them (e.g. the legacy dst re-homing
  retained during CUDA graph capture) must have a tracking issue, not be
  silently accepted.
- **Neutral naming.** Test fixtures and hazard patterns are named for the
  pattern (e.g. accumulate-across-resets), not for a consumer.

## Conventions

- Rebase and merge; branch -> PR -> CI green -> rebase-merge.
- GPU work (benches, `-tags cuda` tests) runs on the DGX via Spark
  (`http://192.168.86.250:8080`), one GPU pod at a time — never via
  interactive SSH.
- `go test ./...` is CPU-only and must stay green everywhere; GPU-gated tests
  skip cleanly when `cuda.Available()` is false.

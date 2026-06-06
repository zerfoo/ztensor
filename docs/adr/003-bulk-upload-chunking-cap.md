# ADR 003: Bound bulkUploadF32 by a byte-sized chunk cap

## Status
Accepted

## Date
2026-06-05

## Context

`GPUEngine.bulkUploadF32` (compute/gpu_engine.go) was introduced in #103 /
release 1.8.0 to collapse many per-tensor `cudaMalloc` + `cudaMemcpy`
round-trips into one device allocation and one host-to-device copy. The
per-tensor pattern wedged the GB10 (sm_121, aarch64, unified memory) driver
when the tensor count reached the tens of thousands.

Issue #106 reports that the bulk path itself now wedges the GB10 driver in an
uninterruptible D-state when the consolidated buffer is large. Reproduced
2026-06-05 from Wolf `train-crossasset` uploading ~213k float32 tensors in one
shot: the single `Malloc(total)` + single `Memcpy(total, HostToDevice)` (the
non-managed branch, the default because `ZERFOO_ENABLE_MANAGED_MEM` is unset)
never returns. The main OS thread is stuck in a CUDA driver ioctl that cannot
be SIGKILLed, which makes the container unkillable and leaks a running pod on
the Spark orchestrator.

There is currently no upper bound on `total` or on the per-call tensor count
beyond `bulkUploadF32MinTensors = 64`, which is only a lower bound.

The issue asks the maintainers to choose the cap shape: bytes-based
(e.g. 256 MB) or tensor-count-based.

## Decision

Bound each chunk by **both** a byte cap and a tensor-count cap, whichever is
hit first. The byte cap is the primary control; the tensor-count cap is a
belt-and-suspenders bound.

- `bulkUploadF32MaxChunkBytes = 64 MiB` (64 << 20). Declared as a package `var`
  rather than a `const` so unit tests can lower it to force the multi-chunk
  path on CPU without a GPU.
- `bulkUploadF32MaxChunkTensors = 4096` (`const`).
- Chunk tiling is extracted into a pure function
  `bulkUploadChunkRanges(nelems, elemSize, maxBytes, maxTensors) [][2]int`
  that greedily packs eligible tensors into contiguous `[start,end)` ranges,
  each bounded by both caps. Every range holds at least one tensor, so a lone
  tensor whose size exceeds the byte cap still gets its own range rather than
  stalling. The ranges exactly tile the input with no gaps or overlaps, which
  makes the boundary math unit-testable independently of CUDA.
- `bulkUploadF32` iterates the ranges; per range it performs one bounded
  `Malloc` (or `mallocManaged`) plus one bounded `Memcpy` (or in-place copy on
  the managed branch), appends the chunk's device pointer to
  `bulkUploadBuffers` for release in `Close`, and sets each tensor's
  `GPUStorage` view at its chunk-local offset.
- No environment-variable override is exposed. The caps are conservative
  internal constants; tests override the `var` directly, and operators have no
  current need to tune them at runtime.

Rationale for bytes as the primary control: the wedge correlates with the size
of a single driver allocation/copy, not with the number of logical tensors. A
byte cap therefore predicts the wedge directly. The additional tensor-count cap
bounds per-chunk bookkeeping (host staging slice length, view-creation loop)
and guards against a pathological many-tiny-tensors input where the byte cap
alone would still pack hundreds of thousands of tensors into one chunk.

Rationale for 64 MiB over a larger value: the exact GB10 wedge threshold is
unknown (open question 1 in #106). 64 MiB is well below any multi-GB size that
was observed to wedge, keeps per-chunk staging allocations small, and still
collapses a multi-GB upload into a few dozen bounded driver round-trips rather
than the tens of thousands the per-tensor path would issue.

Rationale for a `var` instead of an env var: the only consumer that needs to
change the cap is the unit test that forces multi-chunk tiling; a package `var`
satisfies that without parsing, validation, or a runtime configuration surface.

## Consequences

Positive:
- No single driver call exceeds 64 MiB, so the GB10 wedge cannot recur
  regardless of how many tensors are uploaded.
- Preserves the bulk-upload win: a few dozen bounded copies instead of one copy
  per tensor. 213k float32 tensors totaling a few GB resolve to tens of driver
  round-trips, not tens of thousands.
- The resulting `GPUStorage` views are byte-identical to the single-buffer
  version within each chunk, so downstream tensor consumers are unaffected.
- `bulkUploadChunkRanges` is a pure function with no CUDA dependency, so the
  tiling logic (both caps, lone-oversized tensor, the 213k-count bound) is
  fully covered by CPU unit tests in `compute/bulk_upload_chunk_test.go`.

Negative:
- `bulkUploadBuffers` now holds several pointers instead of one, so `Close`
  frees a small list. It is already a slice; no structural change.
- A weight tensor larger than 64 MiB still issues one over-cap copy. In
  practice individual dense f32 weights are bounded (for example 256x1024 f32
  is 1 MB), so this path is rare; it is recorded in the plan risk register.
- The caps are heuristics, not measured thresholds. They are conservative; if a
  wedge is ever observed at 64 MiB the `var` can be lowered (requires a rebuild,
  acceptable given no operator has needed runtime tuning).

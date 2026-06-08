# ADR 005: Stream-ordered ArenaPool overflow on GB10

## Status
Accepted

## Date
2026-06-07

## Context

`ArenaPool` is a bump-pointer allocator over one pre-allocated CUDA region. When
the arena is exhausted, `ArenaPool.Alloc` falls back to its `MemPool`, which --
outside CUDA graph capture -- issues a synchronous `cudaMalloc`
(`internal/cuda/mempool.go`). The #111 capture guard (ADR 004) only refuses this
fallback while a graph capture is active; during normal (non-capture) training it
correctly does not fire, so the synchronous `cudaMalloc` runs.

Issue #115 showed this is fatal on the NVIDIA GB10 (sm_121, unified memory).
During Wolf `train-crossasset -gpu` (full COIN bars, v1.8.2,
`ZERFOO_ARENA_SIZE_GB=64`), training freezes at `step=0`: goroutine 1 is pinned
in the arena fallback `cudaMalloc` (forward `MatMul -> makeGPUResult` and backward
`LayerNormalization.Backward -> gpuSub -> gpuBroadcastOp -> ArenaPool.Alloc`), with
a D-state sibling thread in `folio_wait_bit_common -> filemap_fault` /
`do_swap_page`. On GB10 unified memory a synchronous `cudaMalloc` under memory
pressure faults and stalls indefinitely. Even tiny allocations (48 B, 12 KB,
16 KB) wedge once the arena is full.

A 64 GB arena still exhausts within a single forward+backward because a full-batch
crossasset step (~16,408 samples x 12 scales) retains every forward activation for
backprop, and the arena reclaims only between steps (`StepScope.Close ->
ResetPool -> arena.Reset`), never within a pass -- correctly, since the
activations are still referenced by the backward pass. So the overflow is not an
avoidable mis-sizing: for full-batch training, the arena legitimately overflows,
and any synchronous-`cudaMalloc` fallback wedges GB10.

`cuda.MallocAsync` / `cuda.FreeAsync` (wrapping `cudaMallocAsync` /
`cudaFreeAsync`, the CUDA stream-ordered pool allocator) already exist and already
work on GB10 -- the capture path uses them. The stream-ordered pool pre-reserves
and reuses device memory, avoiding the per-allocation page-fault thrash that
synchronous `cudaMalloc` triggers.

## Decision

Make the `ArenaPool` exhaustion fallback **stream-ordered** when a stream is
available, instead of a synchronous `cudaMalloc`.

1. Add an `overflowStream *Stream` to `cuda.ArenaPool`, set via
   `SetOverflowStream(s)`. The engine wires its own stream
   (`compute.GPUEngine` construction calls `arena.Inner().SetOverflowStream(e.stream)`),
   so overflow allocations are stream-ordered with the kernels that consume them
   (arena allocations are used by kernels launched on `e.stream`).
2. In `ArenaPool.Alloc`, after the #111 capture guard
   (`CaptureActive() && !fallback.IsCapturing()` -> `ErrCaptureUnsafeAlloc`,
   unchanged and taking precedence), when not capturing and `overflowStream != nil`,
   allocate via `MallocAsync(aligned, overflowStream)` and record the pointer so
   `Free` routes it to `FreeAsync(ptr, overflowStream)`. When `overflowStream` is
   nil (CPU/tests) or the fallback is capture-aware (engine-driven capture, which
   already routes through `captureStream` async), behavior is unchanged.
3. Route the async calls through swappable package-level indirections
   (`arenaMallocAsyncFn` / `arenaFreeAsyncFn`) so CPU unit tests can assert the
   routing without a GPU.

The #115 secondary question (why a 64 GB arena overflows) is answered by
documentation, not code: full-batch backprop retains activations and the arena
reclaims only between steps. `ZERFOO_ARENA_SIZE_GB` remains a tuning knob but is
not sufficient alone, because any synchronous fallback wedges GB10. Within-pass
reclamation is deliberately out of scope -- freeing still-referenced activations
would corrupt backprop.

## Consequences

Positive:
- The GB10 training freeze is eliminated at its mechanism: overflow allocations no
  longer issue a synchronous `cudaMalloc` that page-fault-thrashes under pressure.
- Stream-ordering preserves correctness: the async allocation is ordered with the
  consuming kernels on the same stream.
- The #111 capture guard is untouched and still takes precedence during
  graph-driven capture.
- The change is contained to `ArenaPool` plus a one-line engine wiring; the
  synchronous `MemPool` fallback remains for the no-stream case (CPU, tests).

Negative / trade-offs:
- `cudaMallocAsync` is valid in stream order: a pointer is safe for kernels
  launched on the same stream after the alloc, but not for cross-stream or host
  use before a sync. ztensor arena allocations are consumed on `e.stream`, so this
  holds; a future caller that uses an arena pointer off-stream would need an
  explicit sync.
- If a full-batch working set genuinely approaches GB10's 122 GB unified memory,
  the stream-ordered pool will also run out -- that is a true OOM, not a fallback
  mechanism problem, and the remedy is smaller batches, a segmented/growable
  arena, or within-pass lifetime reclamation (follow-up, not this ADR).
- Async free is also stream-ordered; pointers must be freed on the same stream.
  `Free` routes recorded async-fallback pointers to `FreeAsync` to honor this.

# ADR 004: Capture-aware ArenaPool exhaustion fallback

## Status
Accepted

## Date
2026-06-06

## Context

`ArenaPool` (internal/cuda/arena.go) is a bump-pointer allocator backed by one
large CUDA allocation. When the arena is exhausted, `ArenaPool.Alloc` falls back
to its `MemPool` (`a.fallback.Alloc -> MemPool.Alloc`, arena.go:147). `MemPool`
uses the capture-safe `cudaMallocAsync` path only when its `captureStream` is set
(via `SetCaptureStream`); otherwise it performs a plain synchronous `cudaMalloc`.

There are two CUDA-graph-capture entry points in the codebase:

1. `compute.GPUEngine.BeginCapture()` -- sets the pool into capture-aware mode
   (`SetCaptureStream`) before `cudaStreamBeginCapture`, so allocations during
   capture route through `cudaMallocAsync`. This path is safe.
2. `graph.(*Graph).Forward` -- calls `cuda.StreamBeginCapture(g.stream)` directly
   (graph/cuda_graph.go:406). It does NOT route through `GPUEngine.BeginCapture`,
   so the engine pool's `captureStream` is never set. During this capture,
   `MemPool.IsCapturing()` returns false and the fallback does a synchronous
   `cudaMalloc`.

On the NVIDIA GB10 (sm_121, aarch64, unified memory) a synchronous `cudaMalloc`
issued while a stream capture is active hangs the driver (the same failure class
as ztensor#93). Issue #111 pinned this with an out-of-band goroutine-1 capture:
during the first crossasset training forward pass, `LayerNormalization.Forward ->
ReduceSum -> gpuSum` allocates its output via `e.pool.Alloc`, the arena is
exhausted, and the fallback `cudaMalloc` hangs inside the graph-driven capture
region. This is the real root cause that issue #106 chased for weeks (the bulk
upload had already completed; the "UploadWeights never returns" log line was a
misattribution -- it was the last clean line before the silent captured forward
pass hung). The #106/#107 bulk-upload chunking is an unrelated defensive bound.

The arena already expresses capture-awareness intent: `resetFloor` exists to
"protect captured graph buffers" across `Reset()`. The exhaustion fallback path
was simply never covered.

## Decision

Make the `ArenaPool` exhaustion fallback capture-aware so it never issues a
synchronous `cudaMalloc` while a CUDA stream capture is active.

1. Track active captures in the `cuda` package with a process-level atomic
   counter. `StreamBeginCapture` increments it on success; `StreamEndCapture`
   (and the capture-watchdog force-end path) decrements it. Expose
   `cuda.CaptureActive() bool`.
2. In `ArenaPool.Alloc`, before taking the synchronous fallback, refuse when a
   capture is active and the fallback would be synchronous:
   `if CaptureActive() && !a.fallback.IsCapturing() { return nil, ErrCaptureUnsafeAlloc }`.
   When the fallback IS capture-aware (engine-driven `BeginCapture` path), the
   `cudaMallocAsync` fallback is graph-safe and is preserved unchanged.
3. Callers that allocate arena memory inside the captured forward pass
   (`gpuSum` and peers) already fall back to CPU on an allocation error, so the
   refused alloc degrades gracefully instead of hanging. If a caller instead
   propagates the error, the capture fails cleanly and the graph layer re-runs
   the region uncaptured -- preferable to an unkillable D-state wedge.

Pair this safety guard with the perf-correct mitigation: size the arena to cover
the forward+backward working set so the fallback does not fire during capture in
the first place (`NewArenaPool` capacity / arena sizing knob). The guard is the
backstop that prevents the hang; correct sizing keeps the captured graph intact
and avoids the per-step CPU fallback.

## Consequences

Positive:
- The GB10 D-state wedge from a synchronous `cudaMalloc` during graph capture is
  eliminated at its source, regardless of which capture entry point is used.
- The fix is localized to the `cuda` package (counter + guard) and requires no
  cross-package wiring between `graph` and `compute`.
- The engine-driven capture path (`GPUEngine.BeginCapture` with a capture-aware
  pool) is unaffected: its `cudaMallocAsync` fallback remains available.

Negative / trade-offs:
- A process-global capture counter assumes captures are not nested across
  unrelated streams in a way that should allow synchronous malloc on one while
  another captures. In practice ztensor captures one graph at a time on the
  engine stream, so the global signal is sufficient; a per-stream check via the
  existing `StreamCaptureStatus` is available if finer granularity is ever
  needed.
- If the arena is undersized, refusing the fallback forces a per-step CPU
  fallback (or a clean capture failure + uncaptured re-run), which is slower than
  a captured GPU step. The perf-correct arena sizing mitigates this; the guard's
  job is only to guarantee no hang.
- The counter must be decremented on every capture-exit path (normal end and
  watchdog force-end) or it will leak and wrongly suppress fallbacks after
  capture ends. Covered by routing all end paths through the decrement.

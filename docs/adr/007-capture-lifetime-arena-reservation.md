# ADR 007: Reserve the captured-graph arena footprint for the graph's lifetime

## Status
Accepted

## Date
2026-06-21

## Context
ztensor supports CUDA-graph capture-replay for training (compute.GraphCapturer,
graph/cuda_graph.go). A training step is captured ONCE and then replayed many
times (e.g. Wolf CrossAsset: `captures=1 replays=2559` across 40 epochs). The
captured graph executable freezes the device addresses of every buffer its
kernels touch (GraphInstantiate bakes kernel args + pointers).

ADR 006 added refcounted save-for-backward pinning so a node's saved activations
survive arena reuse between that node's Forward and Backward WITHIN a step. The
pins are released as soon as the owning node's Backward completes
(graph.releaseSaved). That is correct for eager execution.

It is NOT sufficient for capture-replay. During the capture step the Go
Forward/Backward code runs once: it allocates intermediates in the arena, pins
save-for-backward tensors, runs Backward, and UNPINS them. The captured graph's
kernels, however, keep referencing those exact device addresses on every replay.
After capture those addresses are unpinned, so the per-epoch `ResetPool()` and
intra-pass free-list reuse hand them out for unrelated tensors. The replayed
graph then reads/writes contaminated memory.

Observed on GB10 (sm_121), issue #167: GPU capture-on CrossAsset training has a
correct epoch 0, then the gradient collapses toward ~0 from the next step, loss
plateaus (~0.669), and fold-0 accuracy is 0.6047 vs the CPU baseline 0.7257 at a
byte-identical config. It is a fast WRONG answer. This is why the production
model is CPU-trained even though ADR 077 prefers GPU.

The existing `onCaptured` callback (graph/cuda_graph.go) only records the
plan's instruction OUTPUT slots (`capturedSlots`) so they can be restored on
replay. It does not reserve the transient arena memory the captured graph
relies on (save-for-backward intermediates, backward scratch). `SetResetFloor`
exists but is not raised to cover the full capture footprint.

## Decision
On successful capture, reserve the entire arena span the captured graph touched
for the lifetime of that graph: raise the arena reset floor to the arena's
current offset (the capture high-water mark) at capture completion.

- Add `ArenaPool.CurrentOffset() int` (internal/cuda/arena.go) returning the
  live bump offset.
- In the GPU engine capture path (compute/gpu_engine.go EndCapture, or the
  `onCaptured` hook wired through graph/cuda_graph.go), after EndCapture and
  before the first GraphLaunch, call `arena.SetResetFloor(arena.CurrentOffset())`.
- Consequence: `ResetPool()` (per-epoch, issue #118 growth bound) rewinds only
  the post-capture region; no address the captured graph uses is ever reissued
  until the graph executor is destroyed or re-captured (which resets the floor
  to the new high-water).

This is correct because the captured graph replays the identical kernel sequence
on the identical addresses every step; that memory must remain reserved for as
long as the graph can be replayed. No allocation happens during the replay phase
(inputs are staged by copying into pre-existing stable tensors), so the reserved
span is never reused mid-run.

The save-for-backward pins (ADR 006) remain the mechanism that protects
intermediates WITHIN an eager step and within the single capture step. ADR 007
adds the orthogonal guarantee for the replay lifetime.

## Consequences
Positive:
- GPU capture-on training matches the CPU/eager numerics (target: CrossAsset
  fold-0 0.7257, the ADR 006 Wolf-hazard gate), unblocking ADR 077 GPU training
  and the ~56x capture-replay speedup as the validated path.
- The fix is localized (one getter + one floor-raise at capture completion) and
  does not change eager-path behavior.

Negative / costs:
- The captured graph's arena footprint stays reserved for the graph's lifetime,
  so the arena cannot reclaim it between epochs. This raises steady-state arena
  residency by one step's high-water (tens of MB for CrossAsset; bounded and far
  under the ZERFOO_ARENA_SIZE_GB budget). Acceptable: that memory is needed every
  replay regardless.
- Re-capture (if the graph is ever rebuilt) must reset the floor to the new
  high-water to avoid leaking the old reservation; covered by setting the floor
  unconditionally at each capture completion.

## References
- ztensor #167 (the bug this fixes)
- ADR 006 save-for-backward arena pinning (the within-step mechanism)
- ADR 004 capture-aware arena fallback; ADR 005 stream-ordered arena overflow
- docs/plan-167-capture-arena-savefb.md (the plan this ADR governs)
- Wolf devlog 2026-06-18 T-PT.0 (first observation, 0.6047 vs 0.7257) and
  2026-06-21 (reconfirmed on v1.19.0, dropout-independent)

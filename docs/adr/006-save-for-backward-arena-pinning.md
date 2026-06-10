# ADR 006: Save-for-backward contract with arena pinning (and poison-on-reset debug mode)

## Status
Accepted

## Date
2026-06-10

## Context
ztensor's GPU arena allocator reuses buffers aggressively within a training step
(and callers such as Wolf reset the pool per sample). That reuse is only safe if
no live reference crosses a reuse boundary. In practice, autograd node
implementations cache forward intermediates in struct fields and read them in
Backward (LayerNorm cached variance, zerfoo#842; AdamW gradient buffer moved
into the arena by Fill, zerfoo#845; Wolf's QK-norm node cached its normalization
inverse). On GPU the arena overwrites those cached buffers before backward runs,
producing garbage gradients and NaN -- a bug class that has now recurred at
least three times and is the dominant reason GPU f32 training diverges while
CPU (no arena reuse, GC-kept tensors) trains cleanly.

PyTorch solves this structurally: ops declare saved tensors via
save_for_backward, and the autograd engine owns their lifetime -- the caching
allocator cannot recycle them. Nothing in ztensor expresses "this tensor must
survive until backward".

## Decision
1. Add an explicit **save-for-backward contract** to the ztensor graph: a
   SaveForBackward API that nodes call during Forward to register tensors their
   Backward will read. The graph engine records the saved set per node.
2. The arena gains **Pin/Unpin** (refcounted, per buffer). Saved tensors are
   pinned at registration; the graph unpins them when the owning node's
   Backward completes (or at step end via MarkStepBoundary for non-training
   forwards). ResetPool and intra-step reuse skip pinned buffers.
3. Caching forward intermediates in node struct fields and reading them in
   Backward is **deprecated**. Nodes must either (a) use SaveForBackward, or
   (b) recompute from the live `inputs ...` the graph already passes to
   Backward. Existing nodes are migrated; new nodes are reviewed against this
   rule.
4. Add a **poison-on-reset debug mode** (env ZTENSOR_ARENA_POISON=1): ResetPool
   and buffer-reuse fill freed regions with NaN sentinels, so any
   use-after-reset explodes deterministically at the corruption site instead of
   surfacing as a delayed, non-deterministic training NaN.

## Consequences
Positive: eliminates the live-tensor corruption class structurally instead of
one bug at a time; preserves the arena's performance for genuinely dead
buffers; poison mode turns days-long NaN hunts into a single failing run that
names the op; the contract mirrors a decade-hardened PyTorch design.

Negative: pinned tensors raise the arena's high-water mark (bounded by what
backward genuinely needs; acceptable on the GB10's 128 GiB unified memory, and
the overflow fallback path still applies); every existing Backward
implementation must be audited and migrated; a small bookkeeping cost
(refcounts) on the hot allocation path.

References: zerfoo#842, zerfoo#845, Wolf docs/adr/072 and devlog 2026-06-10
(the bug-class history); PyTorch SavedVariable / CUDACachingAllocator (the
pattern source); docs/plan-gpu-training-hardening.md in zerfoo (the umbrella
plan).

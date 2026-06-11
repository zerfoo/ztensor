package graph

import (
	"sync"

	"github.com/zerfoo/ztensor/tensor"
)

// Save-for-backward contract (ADR 006 decisions 1-3; zerfoo
// docs/plan-gpu-training-hardening.md T2.1, issue #128).
//
// Nodes that need a forward intermediate in Backward have exactly two
// sanctioned ways to keep it:
//
//  1. SaveForBackward (this file) -- for expensive intermediates. The node
//     registers the tensor during Forward; the graph records it in the
//     node's saved set and pins its storage if it is arena-backed
//     (tensor.PinnableStorage), so arena Reset / intra-pass reuse cannot
//     recycle the buffer before Backward consumes it. Non-arena storage
//     (CPU engines, bucketed pools) is a no-op: GC-owned memory cannot be
//     reclaimed behind a live reference.
//
//  2. Recompute from live inputs -- for cheap intermediates. Backward
//     already receives the node's live `inputs ...` from the graph memo;
//     recomputing (e.g. a mask, a scalar scale) avoids any pinned memory.
//
// Caching a forward intermediate in a node struct field and reading it in
// Backward WITHOUT either mechanism is deprecated: on GPU the arena
// overwrites the cached buffer before backward runs (zerfoo#842 LayerNorm
// variance, zerfoo#845 gradient buffer, Wolf QK-norm cached inverse).
// Run with ZTENSOR_ARENA_POISON=1 to make such bugs explode as
// deterministic NaNs at the corruption site.
//
// Lifetime: the graph unpins a node's saved set immediately after that
// node's Backward returns, and releases every remaining set at the end of
// Backward (nodes that never received a gradient) and at the start of the
// next Forward (forward-only inference loops). A pin held across an engine
// ResetPool raises the arena's rewind floor, so at most one pass of saved
// intermediates is retained between forward-only passes.
//
// Plumbing: nodes do not hold a graph handle, so Builder.Build hands every
// node implementing SaverAware a per-node Saver bound to its identity. A
// per-node handle (rather than a context value or a "current node" field on
// the graph) keeps saves correctly attributed under ParallelForward, where
// several nodes run Forward concurrently.

// Saver is the handle a node uses to register forward intermediates its
// Backward will read. Implementations are safe for concurrent use by
// multiple nodes (ParallelForward).
type Saver[T tensor.Numeric] interface {
	// SaveForBackward records tensors in the calling node's saved set and
	// pins arena-backed storage until the node's Backward completes.
	// Nil tensors are ignored. The node keeps its own references to the
	// tensors; the contract owns only the memory lifetime.
	SaveForBackward(ts ...*tensor.TensorNumeric[T])
}

// SaverAware is implemented by nodes that use the save-for-backward
// contract. Builder.Build calls SetSaver with a Saver bound to the node
// when the graph is constructed; nodes used outside a Graph receive no
// Saver and must tolerate a nil handle.
type SaverAware[T tensor.Numeric] interface {
	SetSaver(s Saver[T])
}

// nodeSaver binds a Saver to the node identity it was issued for, so a save
// made during a concurrent ParallelForward is attributed to the right node.
type nodeSaver[T tensor.Numeric] struct {
	g    *Graph[T]
	node Node[T]
}

func (s *nodeSaver[T]) SaveForBackward(ts ...*tensor.TensorNumeric[T]) {
	s.g.SaveForBackward(s.node, ts...)
}

// savedEntry records one saved tensor and, when the save pinned arena-backed
// storage, the exact PinnableStorage to unpin on release. Holding the
// asserted storage (not re-asserting at release time) keeps pin/unpin
// balanced even if the tensor's storage is swapped between save and release.
type savedEntry[T tensor.Numeric] struct {
	t      *tensor.TensorNumeric[T]
	pinned tensor.PinnableStorage // non-nil iff PinForBackward returned true
}

// savedSets is the per-node saved-tensor bookkeeping, guarded by its own
// mutex because ParallelForward runs node Forwards (and therefore saves)
// concurrently while the graph mutex is held by the executor.
type savedSets[T tensor.Numeric] struct {
	mu   sync.Mutex
	sets map[Node[T]][]savedEntry[T]
}

// SaveForBackward records tensors that node's Backward will read, pinning
// each tensor's storage if it is arena-backed. Nodes normally call this via
// the Saver handed to them by Builder.Build (see SaverAware); callers that
// orchestrate nodes manually may invoke it directly.
func (g *Graph[T]) SaveForBackward(node Node[T], ts ...*tensor.TensorNumeric[T]) {
	g.saved.mu.Lock()
	defer g.saved.mu.Unlock()
	if g.saved.sets == nil {
		g.saved.sets = make(map[Node[T]][]savedEntry[T])
	}
	for _, t := range ts {
		if t == nil {
			continue
		}
		e := savedEntry[T]{t: t}
		if p, ok := t.GetStorage().(tensor.PinnableStorage); ok && p.PinForBackward() {
			e.pinned = p
		}
		g.saved.sets[node] = append(g.saved.sets[node], e)
	}
}

// SavedForBackward returns the tensors currently saved for node, in save
// order. Mainly useful in tests and diagnostics; nodes are expected to keep
// their own references.
func (g *Graph[T]) SavedForBackward(node Node[T]) []*tensor.TensorNumeric[T] {
	g.saved.mu.Lock()
	defer g.saved.mu.Unlock()
	entries := g.saved.sets[node]
	if len(entries) == 0 {
		return nil
	}
	out := make([]*tensor.TensorNumeric[T], len(entries))
	for i, e := range entries {
		out[i] = e.t
	}
	return out
}

// releaseSaved unpins and drops node's saved set. Called immediately after
// the node's Backward returns (success or error): the set was consumed, or
// the step is aborting and the next Forward must not inherit stale pins.
func (g *Graph[T]) releaseSaved(node Node[T]) {
	g.saved.mu.Lock()
	defer g.saved.mu.Unlock()
	for _, e := range g.saved.sets[node] {
		if e.pinned != nil {
			e.pinned.UnpinForBackward()
		}
	}
	delete(g.saved.sets, node)
}

// releaseAllSaved unpins and drops every saved set. Called at the start of
// Forward (a forward-only pass never ran Backward, so its saves must not
// leak pins across inference loops) and at the end of Backward (nodes that
// never received a gradient).
func (g *Graph[T]) releaseAllSaved() {
	g.saved.mu.Lock()
	defer g.saved.mu.Unlock()
	for _, entries := range g.saved.sets {
		for _, e := range entries {
			if e.pinned != nil {
				e.pinned.UnpinForBackward()
			}
		}
	}
	g.saved.sets = nil
}

// wireSavers hands every SaverAware node a Saver bound to its identity.
// Called by Builder.Build once the graph exists.
func (g *Graph[T]) wireSavers() {
	for _, n := range g.nodes {
		if sa, ok := n.(SaverAware[T]); ok {
			sa.SetSaver(&nodeSaver[T]{g: g, node: n})
		}
	}
}

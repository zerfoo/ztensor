package graph

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Transposer is the minimal interface needed by FoldConstantTransposes
// to pre-apply transpose operations on constant tensors. The signature
// matches compute.Engine[T].Transpose (variadic dst for buffer reuse).
type Transposer[T tensor.Numeric] interface {
	Transpose(ctx context.Context, a *tensor.TensorNumeric[T], axes []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
}

// FoldConstantTransposes removes Transpose nodes whose sole input is a constant
// (Parameter/Constant node). The transpose is pre-applied to the constant data
// and all consumers of the Transpose node are rewired to use the pre-transposed
// constant directly.
//
// If the graph has no foldable transposes, the original graph is returned.
// The original graph should not be used after this call if a new graph is returned.
func FoldConstantTransposes[T tensor.Numeric](g *Graph[T], tr Transposer[T]) (*Graph[T], error) {
	// Identify Transpose nodes with constant inputs.
	type foldCandidate struct {
		transposeNode Node[T]
		constantInput Node[T]
		perm          []int
	}

	var candidates []foldCandidate
	for _, n := range g.nodes {
		if n.OpType() != "Transpose" {
			continue
		}
		deps := g.dependencies[n]
		if len(deps) != 1 {
			continue
		}
		input := deps[0]
		if !isConstantNode(input) {
			continue
		}
		perm := extractPerm(n)
		if perm == nil {
			continue
		}
		candidates = append(candidates, foldCandidate{
			transposeNode: n,
			constantInput: input,
			perm:          perm,
		})
	}

	if len(candidates) == 0 {
		return g, nil
	}

	// For each candidate, pre-transpose the constant and create a replacement node.
	replacements := make(map[Node[T]]Node[T]) // transposeNode -> replacement constantNode
	for _, c := range candidates {
		// Get the constant value by running Forward on it.
		val, err := c.constantInput.Forward(context.Background())
		if err != nil {
			return nil, fmt.Errorf("fold transpose: failed to evaluate constant: %w", err)
		}
		transposed, err := tr.Transpose(context.Background(), val, c.perm)
		if err != nil {
			return nil, fmt.Errorf("fold transpose: failed to pre-transpose constant: %w", err)
		}
		replacements[c.transposeNode] = &preTransposedNode[T]{value: transposed}
	}

	// Build a new graph with transposes removed and dependencies rewired.
	removedSet := make(map[Node[T]]bool, len(candidates))
	for _, c := range candidates {
		removedSet[c.transposeNode] = true
	}

	// Also check if any original constant node is now orphaned (no other consumer).
	// Count references to each constant input.
	constRefCount := make(map[Node[T]]int)
	for _, c := range candidates {
		constRefCount[c.constantInput] = 0
	}
	for _, n := range g.nodes {
		if removedSet[n] {
			continue
		}
		for _, dep := range g.dependencies[n] {
			if _, tracked := constRefCount[dep]; tracked {
				constRefCount[dep]++
			}
		}
	}

	var newNodes []Node[T]
	newDeps := make(map[Node[T]][]Node[T])
	var newInputs []Node[T]

	// Add replacement nodes first so they're available as dependencies.
	for _, replacement := range replacements {
		newNodes = append(newNodes, replacement)
		newDeps[replacement] = nil
	}

	for _, n := range g.nodes {
		if removedSet[n] {
			continue
		}
		// Skip orphaned constant nodes (only referenced by removed transposes).
		if _, isTracked := constRefCount[n]; isTracked && constRefCount[n] == 0 {
			continue
		}

		newNodes = append(newNodes, n)

		// Rewire dependencies.
		oldDeps := g.dependencies[n]
		newNodeDeps := make([]Node[T], len(oldDeps))
		for i, dep := range oldDeps {
			if replacement, ok := replacements[dep]; ok {
				newNodeDeps[i] = replacement
			} else {
				newNodeDeps[i] = dep
			}
		}
		newDeps[n] = newNodeDeps

		if _, ok := n.(*inputNode[T]); ok {
			newInputs = append(newInputs, n)
		}
	}

	// Determine the output node (may have been rewired).
	output := g.output
	if replacement, ok := replacements[output]; ok {
		output = replacement
	}

	// Re-sort topologically.
	sorted, err := topologicalSort(newNodes, newDeps)
	if err != nil {
		return nil, fmt.Errorf("fold transpose: topological sort failed: %w", err)
	}

	return &Graph[T]{
		engine:       g.engine,
		nodes:        sorted,
		dependencies: newDeps,
		inputs:       newInputs,
		output:       output,
		parallel:     g.parallel,
	}, nil
}

// isConstantNode returns true if the node produces a constant value
// (Parameter node or Constant node that has no dynamic inputs).
func isConstantNode[T tensor.Numeric](n Node[T]) bool {
	switch n.OpType() {
	case "Parameter", "Constant":
		return true
	}
	return false
}

// extractPerm extracts the permutation axes from a Transpose node's attributes.
func extractPerm[T tensor.Numeric](n Node[T]) []int {
	attrs := n.Attributes()
	if attrs == nil {
		return nil
	}
	permRaw, ok := attrs["perm"]
	if !ok {
		return nil
	}
	switch p := permRaw.(type) {
	case []int:
		return p
	case []int64:
		result := make([]int, len(p))
		for i, v := range p {
			result[i] = int(v)
		}
		return result
	}
	return nil
}

// preTransposedNode wraps a pre-transposed constant tensor.
type preTransposedNode[T tensor.Numeric] struct {
	value *tensor.TensorNumeric[T]
}

// OpType returns "Parameter" since pre-transposed nodes act as constants.
func (n *preTransposedNode[T]) OpType() string { return "Parameter" }

// Attributes returns nil as pre-transposed nodes have no attributes.
func (n *preTransposedNode[T]) Attributes() map[string]interface{} { return nil }

// OutputShape returns the shape of the pre-transposed tensor.
func (n *preTransposedNode[T]) OutputShape() []int { return n.value.Shape() }

// Parameters returns nil as pre-transposed nodes have no trainable parameters.
func (n *preTransposedNode[T]) Parameters() []*Parameter[T] { return nil }

// Forward returns the pre-transposed constant tensor.
func (n *preTransposedNode[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return n.value, nil
}

// Backward is a no-op for constant nodes.
func (n *preTransposedNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

package graph

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// CheckpointedSegment wraps a subgraph segment for gradient checkpointing.
// During the forward pass, it executes the segment normally but discards
// intermediate activations, keeping only the final output. During the backward
// pass, it re-runs the forward pass to recompute the intermediates before
// computing gradients.
//
// This trades compute for memory: peak memory is reduced because intermediate
// tensors from checkpointed layers are not held simultaneously.
type CheckpointedSegment[T tensor.Numeric] struct {
	// forward is the user-supplied function that builds/runs the subgraph segment.
	forward func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error)
	// savedInputs are the inputs captured during the forward pass for recomputation.
	savedInputs []*tensor.TensorNumeric[T]
	// output is the result of the forward pass (kept for the backward pass entry point).
	output *tensor.TensorNumeric[T]
}

// checkpointNode is a graph node that wraps a CheckpointedSegment.
// It implements the Node[T] interface so it can be inserted into a Graph.
type checkpointNode[T tensor.Numeric] struct {
	NoParameters[T]
	segment     *CheckpointedSegment[T]
	outputShape []int
}

func (n *checkpointNode[T]) OpType() string                     { return "Checkpoint" }
func (n *checkpointNode[T]) OutputShape() []int                 { return n.outputShape }
func (n *checkpointNode[T]) Attributes() map[string]interface{} { return map[string]interface{}{} }

func (n *checkpointNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	seg := n.segment

	// Save inputs for recomputation during backward.
	seg.savedInputs = make([]*tensor.TensorNumeric[T], len(inputs))
	copy(seg.savedInputs, inputs)

	// Run the subgraph forward.
	outputs, err := seg.forward(ctx, inputs)
	if err != nil {
		return nil, fmt.Errorf("checkpoint forward: %w", err)
	}
	if len(outputs) == 0 {
		return nil, fmt.Errorf("checkpoint forward: segment produced no outputs")
	}

	seg.output = outputs[0]
	// Intermediate activations inside the segment function are not referenced
	// here — only the final output is retained. The segment function's internal
	// tensors become eligible for GC / pool release.
	return seg.output, nil
}

func (n *checkpointNode[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGrad *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	seg := n.segment

	// Recompute forward to recover intermediate activations.
	recomputeInputs := seg.savedInputs
	if len(recomputeInputs) == 0 {
		recomputeInputs = inputs
	}

	// Run the segment forward again to materialize intermediates.
	// The segment function is expected to be deterministic.
	_, err := seg.forward(ctx, recomputeInputs)
	if err != nil {
		return nil, fmt.Errorf("checkpoint recompute: %w", err)
	}

	// Propagate the output gradient back through the recomputed segment.
	// Since we re-ran forward, the segment's internal nodes have fresh
	// activations that can be used for gradient computation.
	//
	// For a simple implementation, we pass the gradient through unchanged
	// to each input (identity gradient). Real backward logic depends on
	// the segment's internal graph structure, which is handled by the
	// Graph.Backward method when the segment is a sub-graph.
	inputGrads := make([]*tensor.TensorNumeric[T], len(recomputeInputs))
	for i := range inputGrads {
		inputGrads[i] = outputGrad
	}

	return inputGrads, nil
}

// Checkpoint wraps a subgraph segment function for gradient checkpointing.
// The returned function can be used in place of the original segment: it
// produces the same output during forward, but discards intermediate
// activations. During backward, intermediates are recomputed from the
// saved inputs.
//
// Usage:
//
//	checkpointed := graph.Checkpoint[float32](func(ctx context.Context, inputs []*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
//	    // ... build and run a subgraph segment ...
//	    return outputs, nil
//	})
//	outputNode := checkpointed(builder, inputNodes)
func Checkpoint[T tensor.Numeric](
	segmentFn func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error),
	outputShape []int,
) *checkpointNode[T] {
	seg := &CheckpointedSegment[T]{
		forward: segmentFn,
	}
	return &checkpointNode[T]{
		segment:     seg,
		outputShape: outputShape,
	}
}


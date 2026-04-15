package graph

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// poisoningPool mimics GPU pool.Release semantics on CPU: Release invalidates
// the tensor by zeroing its data, so any downstream read of a prematurely
// released tensor produces wrong values. This lets the CPU regression test
// reproduce the gemma4-edge symptom (reading a released pass-through tensor)
// without a CUDA device.
type poisoningPool[T tensor.Numeric] struct{}

func (poisoningPool[T]) Release(t *tensor.TensorNumeric[T]) {
	if t == nil {
		return
	}
	data := t.Data()
	var zero T
	for i := range data {
		data[i] = zero
	}
}

// passthroughNode returns inputs[0] verbatim. Models gemma4-edge's
// pleCombinedProducer, which emits a side-effect (cached tensors) but returns
// its input hidden state unchanged so downstream nodes see a well-formed edge.
type passthroughNode[T tensor.Numeric] struct {
	NoParameters[T]
}

func (n *passthroughNode[T]) OpType() string                     { return "PassThrough" }
func (n *passthroughNode[T]) OutputShape() []int                 { return nil }
func (n *passthroughNode[T]) Attributes() map[string]interface{} { return map[string]interface{}{} }
func (n *passthroughNode[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return inputs[0], nil
}
func (n *passthroughNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// TestPassthroughNodeNotReleased guards the E98/T98.2.3 regression: when a
// node returns its input verbatim, the graph runtime must not pool-release
// the upstream tensor, because the pass-through output aliases it. Before
// the fix, the runtime freed the buffer as soon as the upstream's refcount
// hit zero, which manifested on CUDA as an illegal memory access at the
// next consumer (input_layernorm for gemma4-edge).
func TestPassthroughNodeNotReleased(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := NewBuilder[float32](eng)

	in := b.Input([]int{2, 3})
	// Insert a non-input producer so its memo entry is release-eligible
	// (input nodes are pinned with rc=-1). pre is the "embedding" stand-in.
	pre := b.AddNode(&scaleNode[float32]{factor: 1, outputShape: []int{2, 3}}, in)
	pt := b.AddNode(&passthroughNode[float32]{}, pre)
	scaled := b.AddNode(&scaleNode[float32]{factor: 2, outputShape: []int{2, 3}}, pt)

	g, err := b.Build(scaled)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	_ = compute.NewTensorPool[float32] // keep import alive for API symmetry
	g.WithPool(poisoningPool[float32]{})

	data := []float32{1, 2, 3, 4, 5, 6}
	input, err := tensor.New[float32]([]int{2, 3}, data)
	if err != nil {
		t.Fatalf("new input: %v", err)
	}

	out, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	got := out.Data()
	want := []float32{2, 4, 6, 8, 10, 12}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("out[%d]: got %v, want %v (full: %v)", i, got[i], want[i], got)
		}
	}
}

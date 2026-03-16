package graph

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// trackingPool counts releases for testing.
type trackingPool struct {
	released int
	tensors  []*tensor.TensorNumeric[float32]
}

func (p *trackingPool) Release(t *tensor.TensorNumeric[float32]) {
	p.released++
	p.tensors = append(p.tensors, t)
}

// addNode adds two inputs element-wise (simplified: returns first input).
type addTestNode struct {
	id int // ensure unique pointer identity for each instance
}

func (n *addTestNode) OpType() string                    { return "Add" }
func (n *addTestNode) Attributes() map[string]any        { return nil }
func (n *addTestNode) OutputShape() []int                { return nil }
func (n *addTestNode) Parameters() []*Parameter[float32] { return nil }
func (n *addTestNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	// Create a new tensor (simulates intermediate allocation).
	data := make([]float32, inputs[0].Size())
	for i := range data {
		v0, _ := inputs[0].At(i)
		data[i] = v0
		if len(inputs) > 1 {
			v1, _ := inputs[1].At(i)
			data[i] += v1
		}
	}
	return tensor.New(inputs[0].Shape(), data)
}
func (n *addTestNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func TestGraph_ForwardWithPool_ReleasesIntermediates(t *testing.T) {
	// Build: Input -> Add1 -> Add2 -> Output
	// Add1's output should be released after Add2 consumes it.
	builder := NewBuilder[float32](nil)
	input := builder.Input([]int{2})
	add1 := &addTestNode{id: 1}
	add2 := &addTestNode{id: 2}
	builder.AddNode(add1, input)
	builder.AddNode(add2, add1)

	g, err := builder.Build(add2)
	if err != nil {
		t.Fatal(err)
	}

	pool := &trackingPool{}
	g.WithPool(pool)

	in, _ := tensor.New([]int{2}, []float32{1, 2})
	out, err := g.Forward(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}

	// add1's intermediate should have been released.
	if pool.released != 1 {
		t.Fatalf("expected 1 release, got %d", pool.released)
	}

	// Output should still be valid.
	if out == nil {
		t.Fatal("output is nil")
	}
	if out.Data()[0] != 1 || out.Data()[1] != 2 {
		t.Fatalf("unexpected output: %v", out.Data())
	}
}

func TestGraph_ForwardWithPool_ProtectsOutput(t *testing.T) {
	// Single node graph: Input -> Add -> Output
	// Add's output IS the graph output, so it must NOT be released.
	builder := NewBuilder[float32](nil)
	input := builder.Input([]int{3})
	add := &addTestNode{id: 0}
	builder.AddNode(add, input)

	g, err := builder.Build(add)
	if err != nil {
		t.Fatal(err)
	}

	pool := &trackingPool{}
	g.WithPool(pool)

	in, _ := tensor.New([]int{3}, []float32{1, 2, 3})
	out, err := g.Forward(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}

	// No releases expected: the only intermediate is the output itself.
	if pool.released != 0 {
		t.Fatalf("expected 0 releases (output protected), got %d", pool.released)
	}

	if out.Data()[0] != 1 {
		t.Fatalf("unexpected output: %v", out.Data())
	}
}

func TestGraph_ForwardWithPool_MultipleConsumers(t *testing.T) {
	// Input -> Add1 -> Add2
	//                -> Add3
	//          Add2, Add3 -> Add4 (output)
	// Add1's output feeds both Add2 and Add3, so it should only be released
	// after both have consumed it (refcount=2).
	builder := NewBuilder[float32](nil)
	input := builder.Input([]int{2})
	add1 := &addTestNode{id: 1}
	add2 := &addTestNode{id: 2}
	add3 := &addTestNode{id: 3}
	add4 := &addTestNode{id: 4}
	builder.AddNode(add1, input)
	builder.AddNode(add2, add1)
	builder.AddNode(add3, add1)
	builder.AddNode(add4, add2, add3)

	g, err := builder.Build(add4)
	if err != nil {
		t.Fatal(err)
	}

	pool := &trackingPool{}
	g.WithPool(pool)

	in, _ := tensor.New([]int{2}, []float32{5, 10})
	_, err = g.Forward(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}

	// Intermediates: add1, add2, add3 should be released (3 total).
	// add4 is output, protected.
	if pool.released != 3 {
		t.Fatalf("expected 3 releases, got %d", pool.released)
	}
}

func TestGraph_ForwardWithoutPool_NoReleases(t *testing.T) {
	// Without pool, no releases should happen.
	builder := NewBuilder[float32](nil)
	input := builder.Input([]int{2})
	add1 := &addTestNode{id: 1}
	add2 := &addTestNode{id: 2}
	builder.AddNode(add1, input)
	builder.AddNode(add2, add1)

	g, err := builder.Build(add2)
	if err != nil {
		t.Fatal(err)
	}

	// No pool set.
	in, _ := tensor.New([]int{2}, []float32{1, 2})
	_, err = g.Forward(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	// If we got here without panic, no releases happened.
}

func TestGraph_ForwardWithPool_ProtectsConstants(t *testing.T) {
	// Parameter -> Add -> Output
	// The parameter node should never be released.
	constVal, _ := tensor.New([]int{2}, []float32{10, 20})
	paramNode := &preTransposedNode[float32]{value: constVal}
	add := &addTestNode{id: 0}

	builder := NewBuilder[float32](nil)
	input := builder.Input([]int{2})
	builder.AddNode(paramNode)
	builder.AddNode(add, input, paramNode)

	g, err := builder.Build(add)
	if err != nil {
		t.Fatal(err)
	}

	pool := &trackingPool{}
	g.WithPool(pool)

	in, _ := tensor.New([]int{2}, []float32{1, 2})
	out, err := g.Forward(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}

	// paramNode is a constant — should NOT be released.
	// Input is protected. Output is protected. No releases expected.
	if pool.released != 0 {
		t.Fatalf("expected 0 releases (constants protected), got %d", pool.released)
	}

	// Output should be the sum.
	if out.Data()[0] != 11 || out.Data()[1] != 22 {
		t.Fatalf("unexpected output: %v", out.Data())
	}
}

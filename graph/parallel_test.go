package graph

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// trackingNode records when Forward was called for concurrency testing.
type trackingNode struct {
	op        string
	shape     []int
	delay     time.Duration
	mu        sync.Mutex
	startAt   time.Time
	endAt     time.Time
	callOrder int32
	counter   *atomic.Int32
}

func newTrackingNode(op string, shape []int, delay time.Duration, counter *atomic.Int32) *trackingNode {
	return &trackingNode{op: op, shape: shape, delay: delay, counter: counter}
}

func (n *trackingNode) OpType() string                     { return n.op }
func (n *trackingNode) Attributes() map[string]interface{} { return nil }
func (n *trackingNode) OutputShape() []int                 { return n.shape }
func (n *trackingNode) Parameters() []*Parameter[float32]  { return nil }

func (n *trackingNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	n.mu.Lock()
	n.startAt = time.Now()
	n.callOrder = n.counter.Add(1)
	n.mu.Unlock()

	if n.delay > 0 {
		time.Sleep(n.delay)
	}

	// Sum all inputs element-wise, or create a ones tensor.
	if len(inputs) == 0 || inputs[0] == nil {
		return tensor.New[float32](n.shape, make([]float32, product(n.shape)))
	}
	result := inputs[0]
	for i := 1; i < len(inputs); i++ {
		// Simple add: just sum data arrays.
		d1 := result.Data()
		d2 := inputs[i].Data()
		out := make([]float32, len(d1))
		for j := range d1 {
			out[j] = d1[j] + d2[j]
		}
		var err error
		result, err = tensor.New[float32](result.Shape(), out)
		if err != nil {
			return nil, err
		}
	}
	n.mu.Lock()
	n.endAt = time.Now()
	n.mu.Unlock()
	return result, nil
}

func (n *trackingNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func product(dims []int) int {
	p := 1
	for _, d := range dims {
		p *= d
	}
	return p
}

func TestParallelForward_DiamondGraph(t *testing.T) {
	// Diamond: Input -> B, C (parallel) -> D
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	builder := NewBuilder[float32](engine)

	counter := &atomic.Int32{}
	input := builder.Input([]int{2})
	nodeB := newTrackingNode("B", []int{2}, 50*time.Millisecond, counter)
	nodeC := newTrackingNode("C", []int{2}, 50*time.Millisecond, counter)
	nodeD := newTrackingNode("D", []int{2}, 0, counter)

	builder.AddNode(nodeB, input)
	builder.AddNode(nodeC, input)
	builder.AddNode(nodeD, nodeB, nodeC)

	g, err := builder.Build(nodeD)
	if err != nil {
		t.Fatal(err)
	}

	inputData, _ := tensor.New[float32]([]int{2}, []float32{1, 2})

	result, err := ParallelForward(context.Background(), g, inputData)
	if err != nil {
		t.Fatal(err)
	}

	// D = B(input) + C(input) = input + input = [2, 4]
	got := result.Data()
	if got[0] != 2 || got[1] != 4 {
		t.Errorf("got %v, want [2 4]", got)
	}

	// B and C should have overlapping execution (both started before either finished).
	if nodeB.startAt.After(nodeC.endAt) || nodeC.startAt.After(nodeB.endAt) {
		t.Error("B and C did not execute in parallel")
	}
}

func TestParallelForward_LinearGraph(t *testing.T) {
	// Linear: Input -> A -> B -> C (no parallelism possible)
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	builder := NewBuilder[float32](engine)

	counter := &atomic.Int32{}
	input := builder.Input([]int{3})
	nodeA := newTrackingNode("A", []int{3}, 0, counter)
	nodeB := newTrackingNode("B", []int{3}, 0, counter)
	nodeC := newTrackingNode("C", []int{3}, 0, counter)

	builder.AddNode(nodeA, input)
	builder.AddNode(nodeB, nodeA)
	builder.AddNode(nodeC, nodeB)

	g, err := builder.Build(nodeC)
	if err != nil {
		t.Fatal(err)
	}

	inputData, _ := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
	result, err := ParallelForward(context.Background(), g, inputData)
	if err != nil {
		t.Fatal(err)
	}

	got := result.Data()
	if got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Errorf("got %v, want [1 2 3]", got)
	}

	// Must execute in order: A before B before C.
	if nodeA.callOrder >= nodeB.callOrder || nodeB.callOrder >= nodeC.callOrder {
		t.Errorf("execution order wrong: A=%d, B=%d, C=%d", nodeA.callOrder, nodeB.callOrder, nodeC.callOrder)
	}
}

func TestParallelForward_ContextCancellation(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	builder := NewBuilder[float32](engine)

	counter := &atomic.Int32{}
	input := builder.Input([]int{1})
	slow := newTrackingNode("slow", []int{1}, 500*time.Millisecond, counter)
	builder.AddNode(slow, input)

	g, err := builder.Build(slow)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	inputData, _ := tensor.New[float32]([]int{1}, []float32{1})
	_, err = ParallelForward(ctx, g, inputData)
	if err == nil {
		t.Error("expected error from canceled context")
	}
}

func TestParallelForward_MatchesSequential(t *testing.T) {
	// Verify parallel gives same result as sequential Forward.
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	builder := NewBuilder[float32](engine)

	counter := &atomic.Int32{}
	input := builder.Input([]int{4})
	a := newTrackingNode("A", []int{4}, 0, counter)
	b := newTrackingNode("B", []int{4}, 0, counter)
	c := newTrackingNode("C", []int{4}, 0, counter)

	builder.AddNode(a, input)
	builder.AddNode(b, input)
	builder.AddNode(c, a, b)

	g, err := builder.Build(c)
	if err != nil {
		t.Fatal(err)
	}

	inputData, _ := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})

	seqResult, err := g.Forward(context.Background(), inputData)
	if err != nil {
		t.Fatal(err)
	}

	parResult, err := ParallelForward(context.Background(), g, inputData)
	if err != nil {
		t.Fatal(err)
	}

	seqData := seqResult.Data()
	parData := parResult.Data()
	for i := range seqData {
		if seqData[i] != parData[i] {
			t.Errorf("index %d: sequential=%v, parallel=%v", i, seqData[i], parData[i])
		}
	}
}

func TestWithParallel_ForwardDelegates(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	builder := NewBuilder[float32](engine)

	counter := &atomic.Int32{}
	input := builder.Input([]int{2})
	nodeA := newTrackingNode("A", []int{2}, 50*time.Millisecond, counter)
	nodeB := newTrackingNode("B", []int{2}, 50*time.Millisecond, counter)
	nodeC := newTrackingNode("C", []int{2}, 0, counter)

	builder.AddNode(nodeA, input)
	builder.AddNode(nodeB, input)
	builder.AddNode(nodeC, nodeA, nodeB)

	g, err := builder.Build(nodeC)
	if err != nil {
		t.Fatal(err)
	}

	// Enable parallel mode.
	g.WithParallel(true)

	inputData, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	result, err := g.Forward(context.Background(), inputData)
	if err != nil {
		t.Fatal(err)
	}

	got := result.Data()
	if got[0] != 2 || got[1] != 4 {
		t.Errorf("got %v, want [2 4]", got)
	}

	// A and B should run in parallel.
	if nodeA.startAt.After(nodeB.endAt) || nodeB.startAt.After(nodeA.endAt) {
		t.Error("A and B did not execute in parallel with WithParallel(true)")
	}
}

func BenchmarkForward_4Branch(b *testing.B) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	builder := NewBuilder[float32](engine)

	counter := &atomic.Int32{}
	input := builder.Input([]int{64})
	branches := make([]Node[float32], 4)
	for i := range branches {
		n := newTrackingNode("branch", []int{64}, time.Millisecond, counter)
		builder.AddNode(n, input)
		branches[i] = n
	}
	merge := newTrackingNode("merge", []int{64}, 0, counter)
	builder.AddNode(merge, branches...)

	g, err := builder.Build(merge)
	if err != nil {
		b.Fatal(err)
	}

	inputData, _ := tensor.New[float32]([]int{64}, make([]float32, 64))

	b.Run("sequential", func(b *testing.B) {
		for range b.N {
			_, _ = g.Forward(context.Background(), inputData)
		}
	})

	b.Run("parallel", func(b *testing.B) {
		for range b.N {
			_, _ = ParallelForward(context.Background(), g, inputData)
		}
	})
}

package graph

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// activationTestNode applies element-wise scaling for testing.
type activationTestNode struct {
	NoParameters[float32]
	factor      float32
	outputShape []int
}

func (n *activationTestNode) OpType() string                     { return "Activation" }
func (n *activationTestNode) OutputShape() []int                 { return n.outputShape }
func (n *activationTestNode) Attributes() map[string]interface{} { return map[string]interface{}{} }

func (n *activationTestNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	src := inputs[0].Data()
	out := make([]float32, len(src))
	for i, v := range src {
		out[i] = v * n.factor
	}
	return tensor.New[float32](inputs[0].Shape(), out)
}

func (n *activationTestNode) Backward(_ context.Context, _ types.BackwardMode, outputGrad *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	src := outputGrad.Data()
	out := make([]float32, len(src))
	for i, v := range src {
		out[i] = v * n.factor
	}
	t, err := tensor.New[float32](outputGrad.Shape(), out)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[float32]{t}, nil
}

// TestGradientCheckpointing verifies gradient checkpointing correctness
// and memory reduction.
//
// The test builds two equivalent computation graphs:
//  1. Non-checkpointed: 8 sequential activation nodes (each a separate memo entry)
//  2. Checkpointed: pairs of 2 activation nodes wrapped in 4 checkpoint segments
//
// This gives us:
//   - No-CP: 1 input + 8 activation nodes = 9 memo entries
//   - CP: 1 input + 4 checkpoint nodes = 5 memo entries (60% of non-CP)
//   - Memory reduction = (9-5)/9 = 44% >= 40%
//
// Forward outputs must be identical (same computation).
// The checkpoint segments recompute during backward, verified via call counting.
func TestGradientCheckpointing(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	const (
		batchSize = 2
		dim       = 16
		numNodes  = 8 // total activation nodes
	)

	// Use distinct factors per node so the computation is unique.
	factors := []float32{0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4}

	// --- Run 1: No checkpointing (8 separate nodes) ---
	b1 := NewBuilder[float32](engine)
	in1 := b1.Input([]int{batchSize, dim})
	var prev1 Node[float32] = in1
	for i := 0; i < numNodes; i++ {
		act := &activationTestNode{factor: factors[i], outputShape: []int{batchSize, dim}}
		b1.AddNode(act, prev1)
		prev1 = act
	}
	g1, err := b1.Build(prev1)
	if err != nil {
		t.Fatalf("Build (no-cp): %v", err)
	}

	inputData := make([]float32, batchSize*dim)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}
	inTensor1, _ := tensor.New[float32]([]int{batchSize, dim}, inputData)
	out1, err := g1.Forward(ctx, inTensor1)
	if err != nil {
		t.Fatalf("Forward (no-cp): %v", err)
	}

	peakNoCP := len(g1.memo)

	// Run backward on non-checkpointed graph.
	gradData := make([]float32, len(out1.Data()))
	for i := range gradData {
		gradData[i] = 1.0
	}
	initGrad1, _ := tensor.New[float32](out1.Shape(), gradData)
	if err := g1.Backward(ctx, types.FullBackprop, initGrad1); err != nil {
		t.Fatalf("Backward (no-cp): %v", err)
	}

	// --- Run 2: Checkpointed (4 checkpoint nodes, each wrapping 2 activations) ---
	var totalFwdCalls int

	b2 := NewBuilder[float32](engine)
	in2 := b2.Input([]int{batchSize, dim})
	var prev2 Node[float32] = in2

	for seg := 0; seg < 4; seg++ {
		f1 := factors[seg*2]
		f2 := factors[seg*2+1]
		cpNode := Checkpoint[float32](
			func(ctx context.Context, inputs []*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
				totalFwdCalls++
				src := inputs[0].Data()
				// Apply two successive activations.
				mid := make([]float32, len(src))
				for i, v := range src {
					mid[i] = v * f1
				}
				out := make([]float32, len(mid))
				for i, v := range mid {
					out[i] = v * f2
				}
				result, err := tensor.New[float32](inputs[0].Shape(), out)
				if err != nil {
					return nil, err
				}
				return []*tensor.TensorNumeric[float32]{result}, nil
			},
			[]int{batchSize, dim},
		)
		b2.AddNode(cpNode, prev2)
		prev2 = cpNode
	}
	g2, err := b2.Build(prev2)
	if err != nil {
		t.Fatalf("Build (cp): %v", err)
	}

	inTensor2, _ := tensor.New[float32]([]int{batchSize, dim}, inputData)
	out2, err := g2.Forward(ctx, inTensor2)
	if err != nil {
		t.Fatalf("Forward (cp): %v", err)
	}

	peakCP := len(g2.memo)

	if totalFwdCalls != 4 {
		t.Errorf("expected 4 forward calls during forward pass, got %d", totalFwdCalls)
	}

	// Verify forward outputs are identical.
	d1, d2 := out1.Data(), out2.Data()
	if len(d1) != len(d2) {
		t.Fatalf("output length mismatch: %d vs %d", len(d1), len(d2))
	}
	for i := range d1 {
		if math.Abs(float64(d1[i]-d2[i])) > 1e-5 {
			t.Errorf("output[%d] mismatch: no-cp=%g, cp=%g", i, d1[i], d2[i])
			break
		}
	}

	// Run backward on checkpointed graph.
	initGrad2, _ := tensor.New[float32](out2.Shape(), gradData)
	if err := g2.Backward(ctx, types.FullBackprop, initGrad2); err != nil {
		t.Fatalf("Backward (cp): %v", err)
	}

	// After backward, each checkpoint segment should have been called a second time
	// (recompute). 4 segments * 2 calls each = 8 total.
	if totalFwdCalls != 8 {
		t.Errorf("expected 8 total forward calls (4 forward + 4 recompute), got %d", totalFwdCalls)
	}

	// --- Memory reduction ---
	// No-CP: 1 input + 8 activation = 9 memo entries
	// CP: 1 input + 4 checkpoint = 5 memo entries
	// Reduction = (9-5)/9 = 44.4%
	t.Logf("Memo entries: no-checkpoint=%d, checkpoint=%d", peakNoCP, peakCP)

	if peakNoCP == 0 {
		t.Fatal("no-checkpoint memo count is 0; something is wrong")
	}

	reduction := float64(peakNoCP-peakCP) / float64(peakNoCP)
	t.Logf("Memory reduction: %.1f%%", reduction*100)

	if reduction < 0.40 {
		t.Errorf("expected >= 40%% memory reduction, got %.1f%% (no-cp=%d, cp=%d)",
			reduction*100, peakNoCP, peakCP)
	}
}

// TestCheckpointForwardCorrectness verifies that a checkpointed node
// produces the same output as running the segment directly.
func TestCheckpointForwardCorrectness(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	const (
		batch = 2
		dim   = 8
	)

	segmentFn := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
		src := inputs[0].Data()
		out := make([]float32, len(src))
		for i, v := range src {
			out[i] = v*2.0 + 1.0
		}
		result, err := tensor.New[float32](inputs[0].Shape(), out)
		if err != nil {
			return nil, err
		}
		return []*tensor.TensorNumeric[float32]{result}, nil
	}

	cpNode := Checkpoint[float32](segmentFn, []int{batch, dim})

	b := NewBuilder[float32](engine)
	input := b.Input([]int{batch, dim})
	b.AddNode(cpNode, input)

	g, err := b.Build(cpNode)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	inputData := make([]float32, batch*dim)
	for i := range inputData {
		inputData[i] = float32(i) * 0.5
	}
	inputTensor, _ := tensor.New[float32]([]int{batch, dim}, inputData)

	out, err := g.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	data := out.Data()
	for i, v := range data {
		expected := inputData[i]*2.0 + 1.0
		if math.Abs(float64(v-expected)) > 1e-6 {
			t.Errorf("data[%d] = %g, want %g", i, v, expected)
		}
	}
}

// TestCheckpointRecomputesDuringBackward verifies that the segment function
// is called again during backward (recomputation).
func TestCheckpointRecomputesDuringBackward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	const dim = 4
	var callCount int

	segmentFn := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
		callCount++
		return []*tensor.TensorNumeric[float32]{inputs[0]}, nil
	}

	cpNode := Checkpoint[float32](segmentFn, []int{1, dim})

	b := NewBuilder[float32](engine)
	input := b.Input([]int{1, dim})
	b.AddNode(cpNode, input)

	g, err := b.Build(cpNode)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	inputTensor, _ := tensor.New[float32]([]int{1, dim}, []float32{1, 2, 3, 4})
	_, err = g.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if callCount != 1 {
		t.Fatalf("expected 1 forward call, got %d", callCount)
	}

	initGrad, _ := tensor.New[float32]([]int{1, dim}, []float32{1, 1, 1, 1})
	err = g.Backward(ctx, types.FullBackprop, initGrad)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if callCount != 2 {
		t.Errorf("expected 2 total forward calls (1 forward + 1 recompute), got %d", callCount)
	}
}

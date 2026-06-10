package gradcheck

import (
	"github.com/zerfoo/ztensor/graph"
)

// Registry returns the OpInfo table for every graph.Node implementation
// shipped (as engine-op wrappers) with ztensor's gradcheck harness. ztensor's
// graph package itself contains no public op nodes (only the unexported
// inputNode/checkpointNode plumbing), so the registry covers the
// compute.Engine operations that training graphs are built from; zerfoo's
// layer nodes register against the same harness in T1.6.
//
// Domain notes (mirroring PyTorch OpInfo):
//   - log/sqrt/rsqrt/div(denominator)/pow(base) sample positive-only inputs;
//   - relu/leaky-relu sample away from the kink at zero;
//   - reduce-max relies on continuous sampling to avoid ties (the
//     non-differentiable point).
func Registry() []OpInfo {
	return []OpInfo{
		// Elementwise binary.
		{
			Name: "Add", Seed: 1,
			Make:        func(e engineT) (graph.Node[float64], error) { return newAddNode(e), nil },
			InputShapes: [][]int{{2, 3}, {2, 3}},
		},
		{
			Name: "Sub", Seed: 2,
			Make:        func(e engineT) (graph.Node[float64], error) { return newSubNode(e), nil },
			InputShapes: [][]int{{2, 3}, {2, 3}},
		},
		{
			Name: "Mul", Seed: 3,
			Make:        func(e engineT) (graph.Node[float64], error) { return newMulNode(e), nil },
			InputShapes: [][]int{{2, 3}, {2, 3}},
		},
		{
			Name: "Div", Seed: 4,
			Make:        func(e engineT) (graph.Node[float64], error) { return newDivNode(e), nil },
			InputShapes: [][]int{{2, 3}, {2, 3}},
			Domains:     []Sampler{DomainDefault, DomainPositive}, // denominator away from zero
		},
		{
			Name: "Pow", Seed: 5,
			Make:        func(e engineT) (graph.Node[float64], error) { return newPowNode(e), nil },
			InputShapes: [][]int{{2, 3}, {2, 3}},
			Domains:     []Sampler{DomainPositive, DomainDefault}, // base must be positive
		},

		// Elementwise unary / activations.
		{
			Name: "Tanh", Seed: 6,
			Make:        func(e engineT) (graph.Node[float64], error) { return newTanhNode(e), nil },
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "Sigmoid", Seed: 7,
			Make:        func(e engineT) (graph.Node[float64], error) { return newSigmoidNode(e), nil },
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "ReLU", Seed: 8,
			Make:        func(e engineT) (graph.Node[float64], error) { return newReLUNode(e), nil },
			InputShapes: [][]int{{2, 3}},
			Domains:     []Sampler{DomainAwayFromZero}, // kink at 0
		},
		{
			Name: "LeakyReLU", Seed: 9,
			Make:        func(e engineT) (graph.Node[float64], error) { return newLeakyReLUNode(e, 0.1), nil },
			InputShapes: [][]int{{2, 3}},
			Domains:     []Sampler{DomainAwayFromZero}, // kink at 0
		},
		{
			Name: "Exp", Seed: 10,
			Make:        func(e engineT) (graph.Node[float64], error) { return newExpNode(e), nil },
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "Log", Seed: 11,
			Make:        func(e engineT) (graph.Node[float64], error) { return newLogNode(e), nil },
			InputShapes: [][]int{{2, 3}},
			Domains:     []Sampler{DomainPositive},
		},
		{
			Name: "Sqrt", Seed: 12,
			Make:        func(e engineT) (graph.Node[float64], error) { return newSqrtNode(e), nil },
			InputShapes: [][]int{{2, 3}},
			Domains:     []Sampler{DomainPositive},
		},
		{
			Name: "Rsqrt", Seed: 13,
			Make:        func(e engineT) (graph.Node[float64], error) { return newRsqrtNode(e), nil },
			InputShapes: [][]int{{2, 3}},
			Domains:     []Sampler{DomainPositive},
		},
		{
			Name: "Sin", Seed: 14,
			Make:        func(e engineT) (graph.Node[float64], error) { return newSinNode(e), nil },
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "Cos", Seed: 15,
			Make:        func(e engineT) (graph.Node[float64], error) { return newCosNode(e), nil },
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "AddScalar", Seed: 16,
			Make:        func(e engineT) (graph.Node[float64], error) { return newAddScalarNode(e, 0.7), nil },
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "MulScalar", Seed: 17,
			Make:        func(e engineT) (graph.Node[float64], error) { return newMulScalarNode(e, -1.3), nil },
			InputShapes: [][]int{{2, 3}},
		},

		// MatMul-like and shape ops.
		{
			Name: "MatMul", Seed: 18,
			Make:        func(e engineT) (graph.Node[float64], error) { return newMatMulNode(e), nil },
			InputShapes: [][]int{{2, 3}, {3, 4}},
		},
		{
			Name: "Transpose", Seed: 19,
			Make:        func(e engineT) (graph.Node[float64], error) { return newTransposeNode(e), nil },
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "Reshape", Seed: 20,
			Make:        func(e engineT) (graph.Node[float64], error) { return newReshapeNode(e, []int{3, 2}), nil },
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "HadamardTransform", Seed: 21,
			Make:        func(e engineT) (graph.Node[float64], error) { return newHadamardNode(e), nil },
			InputShapes: [][]int{{2, 4}}, // dim must be a power of two
		},

		// Softmax and reductions.
		{
			Name: "Softmax", Seed: 22,
			Make:        func(e engineT) (graph.Node[float64], error) { return newSoftmaxNode(e, 1), nil },
			InputShapes: [][]int{{2, 4}},
		},
		{
			Name: "ReduceSum", Seed: 23,
			Make:        func(e engineT) (graph.Node[float64], error) { return newReduceSumNode(e, 1), nil },
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "ReduceMean", Seed: 24,
			Make:        func(e engineT) (graph.Node[float64], error) { return newReduceMeanNode(e, 1), nil },
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "ReduceMax", Seed: 25,
			Make:        func(e engineT) (graph.Node[float64], error) { return newReduceMaxNode(e), nil },
			InputShapes: [][]int{{2, 3}}, // continuous sampling avoids ties
		},

		// LayerNorm-like (trainable parameters; exercises the param path).
		{
			Name: "LayerNorm", Seed: 26,
			Make:        func(e engineT) (graph.Node[float64], error) { return newLayerNormNode(e, 4) },
			InputShapes: [][]int{{3, 4}},
		},
	}
}

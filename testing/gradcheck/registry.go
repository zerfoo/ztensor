package gradcheck

import (
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// NewRegistryNode constructs the registry op named name on the given engine
// at any float precision. It is the single source of truth for constructor
// arguments (activation slopes, axes, reshape targets, layernorm width), so
// the float64 gradcheck harness and the float32 PyTorch-oracle harness
// (testing/oracle) exercise byte-identical op configurations.
func NewRegistryNode[T tensor.Float](name string, e compute.Engine[T]) (graph.Node[T], error) {
	switch name {
	case "Add":
		return newAddNode(e), nil
	case "Sub":
		return newSubNode(e), nil
	case "Mul":
		return newMulNode(e), nil
	case "Div":
		return newDivNode(e), nil
	case "Pow":
		return newPowNode(e), nil
	case "Tanh":
		return newTanhNode(e), nil
	case "Sigmoid":
		return newSigmoidNode(e), nil
	case "ReLU":
		return newReLUNode(e), nil
	case "LeakyReLU":
		return newLeakyReLUNode(e, 0.1), nil
	case "Exp":
		return newExpNode(e), nil
	case "Log":
		return newLogNode(e), nil
	case "Sqrt":
		return newSqrtNode(e), nil
	case "Rsqrt":
		return newRsqrtNode(e), nil
	case "Sin":
		return newSinNode(e), nil
	case "Cos":
		return newCosNode(e), nil
	case "AddScalar":
		return newAddScalarNode(e, 0.7), nil
	case "MulScalar":
		return newMulScalarNode(e, -1.3), nil
	case "MatMul":
		return newMatMulNode(e), nil
	case "Transpose":
		return newTransposeNode(e), nil
	case "Reshape":
		return newReshapeNode(e, []int{3, 2}), nil
	case "HadamardTransform":
		return newHadamardNode(e), nil
	case "Softmax":
		return newSoftmaxNode(e, 1), nil
	case "ReduceSum":
		return newReduceSumNode(e, 1), nil
	case "ReduceMean":
		return newReduceMeanNode(e, 1), nil
	case "ReduceMax":
		return newReduceMaxNode(e), nil
	case "LayerNorm":
		return newLayerNormNode(e, 4)
	case "GroupNorm":
		return newGroupNormNode(e, 4, 2)
	default:
		return nil, fmt.Errorf("gradcheck: no registry op named %q", name)
	}
}

// registryMake adapts NewRegistryNode to the OpInfo.Make signature.
func registryMake(name string) func(e engineT) (graph.Node[float64], error) {
	return func(e engineT) (graph.Node[float64], error) {
		return NewRegistryNode[float64](name, e)
	}
}

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
			Make:        registryMake("Add"),
			InputShapes: [][]int{{2, 3}, {2, 3}},
		},
		{
			Name: "Sub", Seed: 2,
			Make:        registryMake("Sub"),
			InputShapes: [][]int{{2, 3}, {2, 3}},
		},
		{
			Name: "Mul", Seed: 3,
			Make:        registryMake("Mul"),
			InputShapes: [][]int{{2, 3}, {2, 3}},
		},
		{
			Name: "Div", Seed: 4,
			Make:        registryMake("Div"),
			InputShapes: [][]int{{2, 3}, {2, 3}},
			Domains:     []Sampler{DomainDefault, DomainPositive}, // denominator away from zero
		},
		{
			Name: "Pow", Seed: 5,
			Make:        registryMake("Pow"),
			InputShapes: [][]int{{2, 3}, {2, 3}},
			Domains:     []Sampler{DomainPositive, DomainDefault}, // base must be positive
		},

		// Elementwise unary / activations.
		{
			Name: "Tanh", Seed: 6,
			Make:        registryMake("Tanh"),
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "Sigmoid", Seed: 7,
			Make:        registryMake("Sigmoid"),
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "ReLU", Seed: 8,
			Make:        registryMake("ReLU"),
			InputShapes: [][]int{{2, 3}},
			Domains:     []Sampler{DomainAwayFromZero}, // kink at 0
		},
		{
			Name: "LeakyReLU", Seed: 9,
			Make:        registryMake("LeakyReLU"),
			InputShapes: [][]int{{2, 3}},
			Domains:     []Sampler{DomainAwayFromZero}, // kink at 0
		},
		{
			Name: "Exp", Seed: 10,
			Make:        registryMake("Exp"),
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "Log", Seed: 11,
			Make:        registryMake("Log"),
			InputShapes: [][]int{{2, 3}},
			Domains:     []Sampler{DomainPositive},
		},
		{
			Name: "Sqrt", Seed: 12,
			Make:        registryMake("Sqrt"),
			InputShapes: [][]int{{2, 3}},
			Domains:     []Sampler{DomainPositive},
		},
		{
			Name: "Rsqrt", Seed: 13,
			Make:        registryMake("Rsqrt"),
			InputShapes: [][]int{{2, 3}},
			Domains:     []Sampler{DomainPositive},
		},
		{
			Name: "Sin", Seed: 14,
			Make:        registryMake("Sin"),
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "Cos", Seed: 15,
			Make:        registryMake("Cos"),
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "AddScalar", Seed: 16,
			Make:        registryMake("AddScalar"),
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "MulScalar", Seed: 17,
			Make:        registryMake("MulScalar"),
			InputShapes: [][]int{{2, 3}},
		},

		// MatMul-like and shape ops.
		{
			Name: "MatMul", Seed: 18,
			Make:        registryMake("MatMul"),
			InputShapes: [][]int{{2, 3}, {3, 4}},
		},
		{
			Name: "Transpose", Seed: 19,
			Make:        registryMake("Transpose"),
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "Reshape", Seed: 20,
			Make:        registryMake("Reshape"),
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "HadamardTransform", Seed: 21,
			Make:        registryMake("HadamardTransform"),
			InputShapes: [][]int{{2, 4}}, // dim must be a power of two
		},

		// Softmax and reductions.
		{
			Name: "Softmax", Seed: 22,
			Make:        registryMake("Softmax"),
			InputShapes: [][]int{{2, 4}},
		},
		{
			Name: "ReduceSum", Seed: 23,
			Make:        registryMake("ReduceSum"),
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "ReduceMean", Seed: 24,
			Make:        registryMake("ReduceMean"),
			InputShapes: [][]int{{2, 3}},
		},
		{
			Name: "ReduceMax", Seed: 25,
			Make:        registryMake("ReduceMax"),
			InputShapes: [][]int{{2, 3}}, // continuous sampling avoids ties
		},

		// LayerNorm-like (trainable parameters; exercises the param path).
		{
			Name: "LayerNorm", Seed: 26,
			Make:        registryMake("LayerNorm"),
			InputShapes: [][]int{{3, 4}},
		},
		// GroupNorm (per-group normalize + per-channel affine); dim=4, groups=2
		// so the [3,4] input reshapes to [6,2] groups. Canonical VAE/UNet norm
		// (E127/T127.1.0a -- extends the oracle to the GroupNorm op class).
		{
			Name: "GroupNorm", Seed: 27,
			Make:        registryMake("GroupNorm"),
			InputShapes: [][]int{{3, 4}},
		},
	}
}

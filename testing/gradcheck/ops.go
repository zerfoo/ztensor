package gradcheck

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// tn abbreviates the tensor type used throughout the op wrappers. The op
// constructors are generic over tensor.Float so the same definitions serve
// both the float64 gradcheck harness and the float32 PyTorch-oracle harness
// (testing/oracle) -- one source of truth per op.
type tn[T tensor.Float] = *tensor.TensorNumeric[T]

// t64 abbreviates the float64 tensor type used by the checker itself.
type t64 = tn[float64]

// opNode is a graph.Node wrapper around compute.Engine operations: Forward
// runs an engine op, Backward applies the analytic gradient (itself built
// from engine ops where possible, so the engine kernels are what gradcheck
// exercises). Like production nodes, it caches its forward output for use in
// Backward -- which is exactly why the checker constructs a fresh instance
// per evaluation.
//
// The cached output is registered through the save-for-backward contract
// (graph.SaverAware, ztensor ADR 006) when a Saver is wired: on arena-backed
// engines the cache would otherwise be recycled by a ResetPool between
// Forward and Backward (the zerfoo#842 bug class). Harnesses that never
// reset an arena between the two passes (gradcheck itself, testing/oracle)
// simply leave the Saver nil.
type opNode[T tensor.Float] struct {
	graph.NoParameters[T]
	opType string
	fwd    func(ctx context.Context, inputs []tn[T]) (tn[T], error)
	bwd    func(ctx context.Context, g tn[T], inputs []tn[T], out tn[T]) ([]tn[T], error)
	out    tn[T] // cached forward output, saved for backward
	saver  graph.Saver[T]
	// extraSaves, when set, returns forward intermediates (beyond the output)
	// that Backward reads via closure capture. They are registered with the
	// Saver so they survive an arena Reset between Forward and Backward (the
	// testing/parity reset-between-fwd-bwd schedule); without this they are
	// arena-freed and Backward reads garbage (max_abs=+Inf). Evaluated after
	// fwd has populated the captured variables.
	extraSaves func() []tn[T]
}

func (n *opNode[T]) OpType() string                     { return n.opType }
func (n *opNode[T]) Attributes() map[string]interface{} { return nil }
func (n *opNode[T]) SetSaver(s graph.Saver[T])          { n.saver = s }
func (n *opNode[T]) OutputShape() []int {
	if n.out == nil {
		return nil
	}
	return n.out.Shape()
}

func (n *opNode[T]) Forward(ctx context.Context, inputs ...tn[T]) (tn[T], error) {
	y, err := n.fwd(ctx, inputs)
	if err != nil {
		return nil, err
	}
	n.out = y
	if n.saver != nil {
		n.saver.SaveForBackward(y)
		if n.extraSaves != nil {
			n.saver.SaveForBackward(n.extraSaves()...)
		}
	}
	return y, nil
}

func (n *opNode[T]) Backward(ctx context.Context, _ types.BackwardMode, g tn[T], inputs ...tn[T]) ([]tn[T], error) {
	if n.out == nil {
		return nil, errors.New(n.opType + ": Backward called before Forward")
	}
	return n.bwd(ctx, g, inputs, n.out)
}

type engineT = compute.Engine[float64]

// The op wrappers participate in the save-for-backward contract so harnesses
// that reset an arena between Forward and Backward (testing/parity) can pin
// their cached intermediates.
var (
	_ graph.SaverAware[float32] = (*opNode[float32])(nil)
	_ graph.SaverAware[float32] = (*layerNormNode[float32])(nil)
)

// unary builds an opNode for a single-input op.
func unary[T tensor.Float](
	name string,
	fwd func(ctx context.Context, x tn[T]) (tn[T], error),
	bwd func(ctx context.Context, g, x, y tn[T]) (tn[T], error),
) *opNode[T] {
	return &opNode[T]{
		opType: name,
		fwd: func(ctx context.Context, in []tn[T]) (tn[T], error) {
			if len(in) != 1 {
				return nil, fmt.Errorf("%s: want 1 input, got %d", name, len(in))
			}
			return fwd(ctx, in[0])
		},
		bwd: func(ctx context.Context, g tn[T], in []tn[T], out tn[T]) ([]tn[T], error) {
			dx, err := bwd(ctx, g, in[0], out)
			if err != nil {
				return nil, err
			}
			return []tn[T]{dx}, nil
		},
	}
}

// binary builds an opNode for a two-input op.
func binary[T tensor.Float](
	name string,
	fwd func(ctx context.Context, a, b tn[T]) (tn[T], error),
	bwd func(ctx context.Context, g, a, b, y tn[T]) (tn[T], tn[T], error),
) *opNode[T] {
	return &opNode[T]{
		opType: name,
		fwd: func(ctx context.Context, in []tn[T]) (tn[T], error) {
			if len(in) != 2 {
				return nil, fmt.Errorf("%s: want 2 inputs, got %d", name, len(in))
			}
			return fwd(ctx, in[0], in[1])
		},
		bwd: func(ctx context.Context, g tn[T], in []tn[T], out tn[T]) ([]tn[T], error) {
			da, db, err := bwd(ctx, g, in[0], in[1], out)
			if err != nil {
				return nil, err
			}
			return []tn[T]{da, db}, nil
		},
	}
}

// --- elementwise binary ops -------------------------------------------------

func newAddNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return binary("Add",
		func(ctx context.Context, a, b tn[T]) (tn[T], error) { return e.Add(ctx, a, b) },
		func(_ context.Context, g, _, _, _ tn[T]) (tn[T], tn[T], error) {
			return g.Copy(), g.Copy(), nil
		})
}

func newSubNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return binary("Sub",
		func(ctx context.Context, a, b tn[T]) (tn[T], error) { return e.Sub(ctx, a, b) },
		func(ctx context.Context, g, _, _, _ tn[T]) (tn[T], tn[T], error) {
			db, err := e.MulScalar(ctx, g, -1)
			if err != nil {
				return nil, nil, err
			}
			return g.Copy(), db, nil
		})
}

func newMulNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return binary("Mul",
		func(ctx context.Context, a, b tn[T]) (tn[T], error) { return e.Mul(ctx, a, b) },
		func(ctx context.Context, g, a, b, _ tn[T]) (tn[T], tn[T], error) {
			da, err := e.Mul(ctx, g, b)
			if err != nil {
				return nil, nil, err
			}
			db, err := e.Mul(ctx, g, a)
			if err != nil {
				return nil, nil, err
			}
			return da, db, nil
		})
}

func newDivNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return binary("Div",
		func(ctx context.Context, a, b tn[T]) (tn[T], error) { return e.Div(ctx, a, b) },
		func(ctx context.Context, g, _, b, y tn[T]) (tn[T], tn[T], error) {
			da, err := e.Div(ctx, g, b)
			if err != nil {
				return nil, nil, err
			}
			yb, err := e.Div(ctx, y, b)
			if err != nil {
				return nil, nil, err
			}
			gyb, err := e.Mul(ctx, g, yb)
			if err != nil {
				return nil, nil, err
			}
			db, err := e.MulScalar(ctx, gyb, -1)
			if err != nil {
				return nil, nil, err
			}
			return da, db, nil
		})
}

func newPowNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return binary("Pow",
		func(ctx context.Context, a, b tn[T]) (tn[T], error) { return e.Pow(ctx, a, b) },
		func(ctx context.Context, g, a, b, y tn[T]) (tn[T], tn[T], error) {
			// dA = g * b * a^(b-1) = g * b * y/a (positive-base domain).
			ya, err := e.Div(ctx, y, a)
			if err != nil {
				return nil, nil, err
			}
			bya, err := e.Mul(ctx, b, ya)
			if err != nil {
				return nil, nil, err
			}
			da, err := e.Mul(ctx, g, bya)
			if err != nil {
				return nil, nil, err
			}
			// dB = g * y * ln(a).
			lna, err := e.Log(ctx, a)
			if err != nil {
				return nil, nil, err
			}
			ylna, err := e.Mul(ctx, y, lna)
			if err != nil {
				return nil, nil, err
			}
			db, err := e.Mul(ctx, g, ylna)
			if err != nil {
				return nil, nil, err
			}
			return da, db, nil
		})
}

// --- elementwise unary ops --------------------------------------------------

func newTanhNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return unary("Tanh",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.Tanh(ctx, x) },
		// TanhPrime is the engine's fused dtanh kernel: upstream * (1 - tanh(x)^2).
		func(ctx context.Context, g, x, _ tn[T]) (tn[T], error) { return e.TanhPrime(ctx, x, g) })
}

func newSigmoidNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	ops := e.Ops()
	return unary("Sigmoid",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.UnaryOp(ctx, x, ops.Sigmoid) },
		func(ctx context.Context, g, x, _ tn[T]) (tn[T], error) {
			sg, err := e.UnaryOp(ctx, x, ops.SigmoidGrad)
			if err != nil {
				return nil, err
			}
			return e.Mul(ctx, g, sg)
		})
}

func newReLUNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	ops := e.Ops()
	return unary("ReLU",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.UnaryOp(ctx, x, ops.ReLU) },
		func(ctx context.Context, g, x, _ tn[T]) (tn[T], error) {
			rg, err := e.UnaryOp(ctx, x, ops.ReLUGrad)
			if err != nil {
				return nil, err
			}
			return e.Mul(ctx, g, rg)
		})
}

func newLeakyReLUNode[T tensor.Float](e compute.Engine[T], alpha float64) *opNode[T] {
	ops := e.Ops()
	return unary("LeakyReLU",
		func(ctx context.Context, x tn[T]) (tn[T], error) {
			return e.UnaryOp(ctx, x, func(v T) T { return ops.LeakyReLU(v, alpha) })
		},
		func(ctx context.Context, g, x, _ tn[T]) (tn[T], error) {
			lg, err := e.UnaryOp(ctx, x, func(v T) T { return ops.LeakyReLUGrad(v, alpha) })
			if err != nil {
				return nil, err
			}
			return e.Mul(ctx, g, lg)
		})
}

func newExpNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return unary("Exp",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.Exp(ctx, x) },
		func(ctx context.Context, g, _, y tn[T]) (tn[T], error) { return e.Mul(ctx, g, y) })
}

func newLogNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return unary("Log",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.Log(ctx, x) },
		func(ctx context.Context, g, x, _ tn[T]) (tn[T], error) { return e.Div(ctx, g, x) })
}

func newSqrtNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return unary("Sqrt",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.Sqrt(ctx, x) },
		func(ctx context.Context, g, _, y tn[T]) (tn[T], error) {
			gy, err := e.Div(ctx, g, y)
			if err != nil {
				return nil, err
			}
			return e.MulScalar(ctx, gy, 0.5)
		})
}

func newRsqrtNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return unary("Rsqrt",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.Rsqrt(ctx, x) },
		func(ctx context.Context, g, x, y tn[T]) (tn[T], error) {
			// d/dx x^(-1/2) = -0.5 * x^(-3/2) = -0.5 * y / x.
			yx, err := e.Div(ctx, y, x)
			if err != nil {
				return nil, err
			}
			gyx, err := e.Mul(ctx, g, yx)
			if err != nil {
				return nil, err
			}
			return e.MulScalar(ctx, gyx, -0.5)
		})
}

func newSinNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return unary("Sin",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.Sin(ctx, x) },
		func(ctx context.Context, g, x, _ tn[T]) (tn[T], error) {
			c, err := e.Cos(ctx, x)
			if err != nil {
				return nil, err
			}
			return e.Mul(ctx, g, c)
		})
}

func newCosNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return unary("Cos",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.Cos(ctx, x) },
		func(ctx context.Context, g, x, _ tn[T]) (tn[T], error) {
			s, err := e.Sin(ctx, x)
			if err != nil {
				return nil, err
			}
			gs, err := e.Mul(ctx, g, s)
			if err != nil {
				return nil, err
			}
			return e.MulScalar(ctx, gs, -1)
		})
}

func newAddScalarNode[T tensor.Float](e compute.Engine[T], c float64) *opNode[T] {
	return unary("AddScalar",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.AddScalar(ctx, x, T(c)) },
		func(_ context.Context, g, _, _ tn[T]) (tn[T], error) { return g.Copy(), nil })
}

func newMulScalarNode[T tensor.Float](e compute.Engine[T], c float64) *opNode[T] {
	return unary("MulScalar",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.MulScalar(ctx, x, T(c)) },
		func(ctx context.Context, g, _, _ tn[T]) (tn[T], error) { return e.MulScalar(ctx, g, T(c)) })
}

// newDropoutNode builds an inverted-dropout op with a fixed drop probability p
// and seed in training mode. The mask is deterministic (Philox keyed by (seed,
// offset)), so with p and seed held constant the op is a fixed element-wise
// linear map y = x * mask/(1-p): finite differences and the analytic backward
// (the same masked scale applied to the upstream gradient) agree exactly. The
// engine must implement the Dropouter capability (CPU and GPU both do); the
// gradcheck/oracle harnesses run on the CPU engine.
func newDropoutNode[T tensor.Float](e compute.Engine[T], p float64, seed uint64) *opNode[T] {
	d, ok := e.(compute.Dropouter[T])
	return unary("Dropout",
		func(ctx context.Context, x tn[T]) (tn[T], error) {
			if !ok {
				return nil, fmt.Errorf("Dropout: engine %T does not implement Dropouter", e)
			}
			return d.Dropout(ctx, x, p, seed, true)
		},
		func(ctx context.Context, g, _, _ tn[T]) (tn[T], error) {
			return d.DropoutBackward(ctx, g, p, seed, true)
		})
}

// --- matmul-like and shape ops ----------------------------------------------

func newMatMulNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return binary("MatMul",
		func(ctx context.Context, a, b tn[T]) (tn[T], error) { return e.MatMul(ctx, a, b) },
		func(ctx context.Context, g, a, b, _ tn[T]) (tn[T], tn[T], error) {
			// dA = g @ B^T. Use the fused MatMulTransposeB kernel when the
			// engine provides it (optional interface); otherwise transpose.
			var da tn[T]
			var err error
			if tb, ok := e.(compute.TransposeBMatMuler[T]); ok {
				da, err = tb.MatMulTransposeB(ctx, g, b)
			} else {
				var bt tn[T]
				bt, err = e.Transpose(ctx, b, []int{1, 0})
				if err == nil {
					da, err = e.MatMul(ctx, g, bt)
				}
			}
			if err != nil {
				return nil, nil, err
			}
			// dB = A^T @ g.
			at, err := e.Transpose(ctx, a, []int{1, 0})
			if err != nil {
				return nil, nil, err
			}
			db, err := e.MatMul(ctx, at, g)
			if err != nil {
				return nil, nil, err
			}
			return da, db, nil
		})
}

func newTransposeNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return unary("Transpose",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.Transpose(ctx, x, []int{1, 0}) },
		func(ctx context.Context, g, _, _ tn[T]) (tn[T], error) { return e.Transpose(ctx, g, []int{1, 0}) })
}

func newReshapeNode[T tensor.Float](e compute.Engine[T], shape []int) *opNode[T] {
	return unary("Reshape",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.Reshape(ctx, x, shape) },
		func(ctx context.Context, g, x, _ tn[T]) (tn[T], error) { return e.Reshape(ctx, g, x.Shape()) })
}

func newHadamardNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return unary("HadamardTransform",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.HadamardTransform(ctx, x) },
		// The normalized Walsh-Hadamard matrix is symmetric, so dX = H^T g = H g.
		func(ctx context.Context, g, _, _ tn[T]) (tn[T], error) { return e.HadamardTransform(ctx, g) })
}

// --- softmax and reductions ---------------------------------------------------

func newSoftmaxNode[T tensor.Float](e compute.Engine[T], axis int) *opNode[T] {
	return unary("Softmax",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.Softmax(ctx, x, axis) },
		func(ctx context.Context, g, _, y tn[T]) (tn[T], error) {
			// dX = y * (g - sum(g*y, axis)).
			gy, err := e.Mul(ctx, g, y)
			if err != nil {
				return nil, err
			}
			s, err := e.ReduceSum(ctx, gy, axis, true)
			if err != nil {
				return nil, err
			}
			gms, err := e.Sub(ctx, g, s)
			if err != nil {
				return nil, err
			}
			return e.Mul(ctx, y, gms)
		})
}

func newReduceSumNode[T tensor.Float](e compute.Engine[T], axis int) *opNode[T] {
	return unary("ReduceSum",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.ReduceSum(ctx, x, axis, true) },
		func(ctx context.Context, g, x, _ tn[T]) (tn[T], error) {
			return e.Repeat(ctx, g, axis, x.Shape()[axis])
		})
}

func newReduceMeanNode[T tensor.Float](e compute.Engine[T], axis int) *opNode[T] {
	return unary("ReduceMean",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.ReduceMean(ctx, x, axis, true) },
		func(ctx context.Context, g, x, _ tn[T]) (tn[T], error) {
			n := x.Shape()[axis]
			r, err := e.Repeat(ctx, g, axis, n)
			if err != nil {
				return nil, err
			}
			return e.DivScalar(ctx, r, T(n))
		})
}

// newReduceMaxNode reduces along axis 1 of a 2D input with keepDims=true.
// The gradient flows only to the (unique) argmax per row; the OpInfo domain
// relies on continuous sampling to avoid ties (the non-differentiable point),
// as PyTorch's OpInfo does for max-like ops.
func newReduceMaxNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	return unary("ReduceMax",
		func(ctx context.Context, x tn[T]) (tn[T], error) { return e.ReduceMax(ctx, x, 1, true) },
		func(_ context.Context, g, x, _ tn[T]) (tn[T], error) {
			shape := x.Shape()
			if len(shape) != 2 {
				return nil, errors.New("ReduceMax wrapper: input must be 2D")
			}
			rows, cols := shape[0], shape[1]
			xd := x.Data()
			dx := make([]T, rows*cols)
			for i := 0; i < rows; i++ {
				arg := 0
				for j := 1; j < cols; j++ {
					if xd[i*cols+j] > xd[i*cols+arg] {
						arg = j
					}
				}
				dx[i*cols+arg] = g.Data()[i]
			}
			return tensor.New[T]([]int{rows, cols}, dx)
		})
}

// --- layernorm-like (with trainable parameters) -------------------------------

// layerNormNode normalizes the last axis of a 2D input and applies a
// trainable elementwise affine (gamma, beta), mirroring the production
// LayerNorm whose cached-statistics GPU bug motivated this harness. It caches
// xhat and the inverse stddev in Forward and consumes them in Backward,
// registering both through the save-for-backward contract when a Saver is
// wired (graph.SaverAware, ztensor ADR 006) -- the exact migration the
// pre-fix LayerNorm needed.
type layerNormNode[T tensor.Float] struct {
	engine compute.Engine[T]
	gamma  *graph.Parameter[T]
	beta   *graph.Parameter[T]
	eps    float64

	xhat  tn[T]
	inv   tn[T]
	saver graph.Saver[T]
}

func newTensorOf[T tensor.Float](shape []int, data []T) (tn[T], error) {
	return tensor.New[T](shape, data)
}

func newLayerNormNode[T tensor.Float](e compute.Engine[T], dim int) (*layerNormNode[T], error) {
	gammaData := make([]T, dim)
	betaData := make([]T, dim)
	for i := 0; i < dim; i++ {
		// Deterministic, non-uniform initial values so parameter gradients
		// are structurally informative.
		gammaData[i] = T(0.8 + 0.1*float64(i))
		betaData[i] = T(-0.2 + 0.15*float64(i))
	}
	gv, err := newTensorOf([]int{1, dim}, gammaData)
	if err != nil {
		return nil, err
	}
	bv, err := newTensorOf([]int{1, dim}, betaData)
	if err != nil {
		return nil, err
	}
	gamma, err := graph.NewParameter[T]("gamma", gv, newTensorOf[T])
	if err != nil {
		return nil, err
	}
	beta, err := graph.NewParameter[T]("beta", bv, newTensorOf[T])
	if err != nil {
		return nil, err
	}
	return &layerNormNode[T]{engine: e, gamma: gamma, beta: beta, eps: 1e-5}, nil
}

func (n *layerNormNode[T]) OpType() string { return "LayerNorm" }
func (n *layerNormNode[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"epsilon": n.eps}
}
func (n *layerNormNode[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{n.gamma, n.beta}
}
func (n *layerNormNode[T]) SetSaver(s graph.Saver[T]) { n.saver = s }

func (n *layerNormNode[T]) OutputShape() []int {
	if n.xhat == nil {
		return nil
	}
	return n.xhat.Shape()
}

func (n *layerNormNode[T]) Forward(ctx context.Context, inputs ...tn[T]) (tn[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("LayerNorm: want 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	e := n.engine
	mean, err := e.ReduceMean(ctx, x, 1, true)
	if err != nil {
		return nil, err
	}
	xc, err := e.Sub(ctx, x, mean)
	if err != nil {
		return nil, err
	}
	sq, err := e.Mul(ctx, xc, xc)
	if err != nil {
		return nil, err
	}
	variance, err := e.ReduceMean(ctx, sq, 1, true)
	if err != nil {
		return nil, err
	}
	veps, err := e.AddScalar(ctx, variance, T(n.eps))
	if err != nil {
		return nil, err
	}
	inv, err := e.Rsqrt(ctx, veps)
	if err != nil {
		return nil, err
	}
	xhat, err := e.Mul(ctx, xc, inv)
	if err != nil {
		return nil, err
	}
	n.xhat = xhat
	n.inv = inv
	if n.saver != nil {
		n.saver.SaveForBackward(xhat, inv)
	}
	scaled, err := e.Mul(ctx, xhat, n.gamma.Value)
	if err != nil {
		return nil, err
	}
	return e.Add(ctx, scaled, n.beta.Value)
}

func (n *layerNormNode[T]) Backward(ctx context.Context, _ types.BackwardMode, g tn[T], _ ...tn[T]) ([]tn[T], error) {
	if n.xhat == nil || n.inv == nil {
		return nil, errors.New("LayerNorm: Backward called before Forward")
	}
	e := n.engine
	// dGamma = sum_batch(g * xhat); dBeta = sum_batch(g).
	gx, err := e.Mul(ctx, g, n.xhat)
	if err != nil {
		return nil, err
	}
	dgamma, err := e.ReduceSum(ctx, gx, 0, true)
	if err != nil {
		return nil, err
	}
	if err := n.gamma.AddGradient(dgamma); err != nil {
		return nil, err
	}
	dbeta, err := e.ReduceSum(ctx, g, 0, true)
	if err != nil {
		return nil, err
	}
	if err := n.beta.AddGradient(dbeta); err != nil {
		return nil, err
	}
	// dX = inv * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)),
	// means over the normalized axis.
	dxhat, err := e.Mul(ctx, g, n.gamma.Value)
	if err != nil {
		return nil, err
	}
	m1, err := e.ReduceMean(ctx, dxhat, 1, true)
	if err != nil {
		return nil, err
	}
	dxx, err := e.Mul(ctx, dxhat, n.xhat)
	if err != nil {
		return nil, err
	}
	m2, err := e.ReduceMean(ctx, dxx, 1, true)
	if err != nil {
		return nil, err
	}
	t1, err := e.Sub(ctx, dxhat, m1)
	if err != nil {
		return nil, err
	}
	xm2, err := e.Mul(ctx, n.xhat, m2)
	if err != nil {
		return nil, err
	}
	t2, err := e.Sub(ctx, t1, xm2)
	if err != nil {
		return nil, err
	}
	dx, err := e.Mul(ctx, t2, n.inv)
	if err != nil {
		return nil, err
	}
	return []tn[T]{dx}, nil
}

// --- groupnorm (per-group normalize + per-channel affine) ---------------------

// groupNormNode normalizes a 2D input [N, C] in `groups` channel-groups and
// then applies a trainable per-channel affine (gamma, beta of shape [1, C]).
// It reshapes [N, C] -> [N*groups, C/groups], normalizes the last axis exactly
// like layerNormNode (so the per-group statistics fall out of the same engine
// reduce/elementwise ops), and reshapes back -- needing NO new engine kernel.
// This is the canonical convolutional-VAE/UNet normalization
// (torch.nn.functional.group_norm); landing it here extends the ADR-091
// PyTorch-oracle harness to the GroupNorm op class (E127/T127.1.0a) and
// unlocks the whole Stable-Diffusion-family VAE/UNet primitive set.
type groupNormNode[T tensor.Float] struct {
	engine compute.Engine[T]
	gamma  *graph.Parameter[T]
	beta   *graph.Parameter[T]
	groups int
	eps    float64

	xhatR tn[T] // normalized input in grouped shape [N*groups, C/groups]
	inv   tn[T] // inverse stddev per group, [N*groups, 1]
	nRows int   // N
	chans int   // C
	saver graph.Saver[T]
}

func newGroupNormNode[T tensor.Float](e compute.Engine[T], dim, groups int) (*groupNormNode[T], error) {
	if groups <= 0 || dim%groups != 0 {
		return nil, fmt.Errorf("GroupNorm: dim %d not divisible by groups %d", dim, groups)
	}
	gammaData := make([]T, dim)
	betaData := make([]T, dim)
	for i := 0; i < dim; i++ {
		// Deterministic, non-uniform initial values so parameter gradients
		// are structurally informative (mirrors newLayerNormNode).
		gammaData[i] = T(0.8 + 0.1*float64(i))
		betaData[i] = T(-0.2 + 0.15*float64(i))
	}
	gv, err := newTensorOf([]int{1, dim}, gammaData)
	if err != nil {
		return nil, err
	}
	bv, err := newTensorOf([]int{1, dim}, betaData)
	if err != nil {
		return nil, err
	}
	gamma, err := graph.NewParameter[T]("gamma", gv, newTensorOf[T])
	if err != nil {
		return nil, err
	}
	beta, err := graph.NewParameter[T]("beta", bv, newTensorOf[T])
	if err != nil {
		return nil, err
	}
	return &groupNormNode[T]{engine: e, gamma: gamma, beta: beta, groups: groups, eps: 1e-5}, nil
}

func (n *groupNormNode[T]) OpType() string { return "GroupNorm" }
func (n *groupNormNode[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"epsilon": n.eps, "groups": n.groups}
}
func (n *groupNormNode[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{n.gamma, n.beta}
}
func (n *groupNormNode[T]) SetSaver(s graph.Saver[T]) { n.saver = s }

func (n *groupNormNode[T]) OutputShape() []int {
	if n.xhatR == nil {
		return nil
	}
	return []int{n.nRows, n.chans}
}

func (n *groupNormNode[T]) Forward(ctx context.Context, inputs ...tn[T]) (tn[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("GroupNorm: want 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("GroupNorm: want 2D input [N, C], got %v", shape)
	}
	n.nRows, n.chans = shape[0], shape[1]
	if n.chans%n.groups != 0 {
		return nil, fmt.Errorf("GroupNorm: C %d not divisible by groups %d", n.chans, n.groups)
	}
	gw := n.chans / n.groups
	e := n.engine
	// Reshape [N, C] -> [N*groups, C/groups]; normalize the last axis.
	xr, err := e.Reshape(ctx, x, []int{n.nRows * n.groups, gw})
	if err != nil {
		return nil, err
	}
	mean, err := e.ReduceMean(ctx, xr, 1, true)
	if err != nil {
		return nil, err
	}
	xc, err := e.Sub(ctx, xr, mean)
	if err != nil {
		return nil, err
	}
	sq, err := e.Mul(ctx, xc, xc)
	if err != nil {
		return nil, err
	}
	variance, err := e.ReduceMean(ctx, sq, 1, true)
	if err != nil {
		return nil, err
	}
	veps, err := e.AddScalar(ctx, variance, T(n.eps))
	if err != nil {
		return nil, err
	}
	inv, err := e.Rsqrt(ctx, veps)
	if err != nil {
		return nil, err
	}
	xhatR, err := e.Mul(ctx, xc, inv)
	if err != nil {
		return nil, err
	}
	n.xhatR = xhatR
	n.inv = inv
	if n.saver != nil {
		n.saver.SaveForBackward(xhatR, inv)
	}
	// Reshape back to [N, C] and apply the per-channel affine.
	xhat, err := e.Reshape(ctx, xhatR, []int{n.nRows, n.chans})
	if err != nil {
		return nil, err
	}
	scaled, err := e.Mul(ctx, xhat, n.gamma.Value)
	if err != nil {
		return nil, err
	}
	return e.Add(ctx, scaled, n.beta.Value)
}

func (n *groupNormNode[T]) Backward(ctx context.Context, _ types.BackwardMode, g tn[T], _ ...tn[T]) ([]tn[T], error) {
	if n.xhatR == nil || n.inv == nil {
		return nil, errors.New("GroupNorm: Backward called before Forward")
	}
	e := n.engine
	gw := n.chans / n.groups
	// xhat in [N, C] form for the per-channel parameter gradients.
	xhat, err := e.Reshape(ctx, n.xhatR, []int{n.nRows, n.chans})
	if err != nil {
		return nil, err
	}
	// dGamma = sum_batch(g * xhat); dBeta = sum_batch(g). Both [1, C].
	gx, err := e.Mul(ctx, g, xhat)
	if err != nil {
		return nil, err
	}
	dgamma, err := e.ReduceSum(ctx, gx, 0, true)
	if err != nil {
		return nil, err
	}
	if err := n.gamma.AddGradient(dgamma); err != nil {
		return nil, err
	}
	dbeta, err := e.ReduceSum(ctx, g, 0, true)
	if err != nil {
		return nil, err
	}
	if err := n.beta.AddGradient(dbeta); err != nil {
		return nil, err
	}
	// dxhat = g * gamma in [N, C]; reshape to grouped form so the per-group
	// normalization backward (means over the C/groups axis) reuses the exact
	// layerNorm dX formula.
	dxhat, err := e.Mul(ctx, g, n.gamma.Value)
	if err != nil {
		return nil, err
	}
	dxhatR, err := e.Reshape(ctx, dxhat, []int{n.nRows * n.groups, gw})
	if err != nil {
		return nil, err
	}
	m1, err := e.ReduceMean(ctx, dxhatR, 1, true)
	if err != nil {
		return nil, err
	}
	dxx, err := e.Mul(ctx, dxhatR, n.xhatR)
	if err != nil {
		return nil, err
	}
	m2, err := e.ReduceMean(ctx, dxx, 1, true)
	if err != nil {
		return nil, err
	}
	t1, err := e.Sub(ctx, dxhatR, m1)
	if err != nil {
		return nil, err
	}
	xm2, err := e.Mul(ctx, n.xhatR, m2)
	if err != nil {
		return nil, err
	}
	t2, err := e.Sub(ctx, t1, xm2)
	if err != nil {
		return nil, err
	}
	dxR, err := e.Mul(ctx, t2, n.inv)
	if err != nil {
		return nil, err
	}
	dx, err := e.Reshape(ctx, dxR, []int{n.nRows, n.chans})
	if err != nil {
		return nil, err
	}
	return []tn[T]{dx}, nil
}

// --- cross-attention (scaled dot-product attention) ---------------------------

// newCrossAttentionNode builds a single-head scaled-dot-product-attention op
// over three 2D inputs Q[Lq,d], K[Lk,d], V[Lk,d] -> [Lq,d]:
//
//	scores = (Q @ K^T) / sqrt(d);  A = softmax(scores, axis=1);  out = A @ V
//
// This is the packaged cross-attention primitive (separate Q vs K/V
// projections) that E127 composes for text<->stream and audio<->video
// coupling. Landing it extends the ADR-091 oracle to the cross-attention op
// class (T127.1.0a). torch oracle:
// torch.nn.functional.scaled_dot_product_attention(x0, x1, x2) (default scale
// 1/sqrt(E), E = query last dim = d -- matches). No trainable parameters; the
// three inputs are Q, K, V and the gradient flows to all three.
func newCrossAttentionNode[T tensor.Float](e compute.Engine[T]) *opNode[T] {
	// Intermediates captured across Forward/Backward via closure. Backward reads
	// attn (softmax weights) and the Q/K/V inputs; under the testing/parity
	// reset-between-fwd-bwd schedule the arena is Reset between the passes, so
	// these must be pinned via the Saver (extraSaves below) -- otherwise they are
	// arena-freed and Backward reads garbage (max_abs=+Inf). gradcheck itself
	// never resets between passes, so it is unaffected either way.
	var (
		attn  tn[T] // softmax weights A [Lq, Lk]
		qIn   tn[T]
		kIn   tn[T]
		vIn   tn[T]
		scale float64
	)
	return &opNode[T]{
		opType: "CrossAttention",
		extraSaves: func() []tn[T] {
			return []tn[T]{attn, qIn, kIn, vIn}
		},
		fwd: func(ctx context.Context, in []tn[T]) (tn[T], error) {
			if len(in) != 3 {
				return nil, fmt.Errorf("CrossAttention: want 3 inputs (Q,K,V), got %d", len(in))
			}
			q, k, v := in[0], in[1], in[2]
			qs := q.Shape()
			scale = 1.0 / math.Sqrt(float64(qs[len(qs)-1]))
			kt, err := e.Transpose(ctx, k, []int{1, 0})
			if err != nil {
				return nil, err
			}
			scores, err := e.MatMul(ctx, q, kt)
			if err != nil {
				return nil, err
			}
			scaled, err := e.MulScalar(ctx, scores, T(scale))
			if err != nil {
				return nil, err
			}
			a, err := e.Softmax(ctx, scaled, 1)
			if err != nil {
				return nil, err
			}
			attn, qIn, kIn, vIn = a, q, k, v
			return e.MatMul(ctx, a, v)
		},
		bwd: func(ctx context.Context, g tn[T], _ []tn[T], _ tn[T]) ([]tn[T], error) {
			if attn == nil {
				return nil, errors.New("CrossAttention: Backward called before Forward")
			}
			// dV = A^T @ g.
			at, err := e.Transpose(ctx, attn, []int{1, 0})
			if err != nil {
				return nil, err
			}
			dV, err := e.MatMul(ctx, at, g)
			if err != nil {
				return nil, err
			}
			// dA = g @ V^T.
			vt, err := e.Transpose(ctx, vIn, []int{1, 0})
			if err != nil {
				return nil, err
			}
			dA, err := e.MatMul(ctx, g, vt)
			if err != nil {
				return nil, err
			}
			// Softmax backward over rows: dScaled = A * (dA - sum(dA*A, axis=1)).
			dAA, err := e.Mul(ctx, dA, attn)
			if err != nil {
				return nil, err
			}
			s, err := e.ReduceSum(ctx, dAA, 1, true)
			if err != nil {
				return nil, err
			}
			dAm, err := e.Sub(ctx, dA, s)
			if err != nil {
				return nil, err
			}
			dScaled, err := e.Mul(ctx, attn, dAm)
			if err != nil {
				return nil, err
			}
			dScores, err := e.MulScalar(ctx, dScaled, T(scale))
			if err != nil {
				return nil, err
			}
			// scores = Q @ K^T  =>  dQ = dScores @ K ; dK = dScores^T @ Q.
			dQ, err := e.MatMul(ctx, dScores, kIn)
			if err != nil {
				return nil, err
			}
			dsT, err := e.Transpose(ctx, dScores, []int{1, 0})
			if err != nil {
				return nil, err
			}
			dK, err := e.MatMul(ctx, dsT, qIn)
			if err != nil {
				return nil, err
			}
			return []tn[T]{dQ, dK, dV}, nil
		},
	}
}

// --- adaLN (adaptive layer-norm modulation) -----------------------------------

// adaLNNode applies the AdaLN affine modulation used by DiT-family diffusion
// models: from a conditioning vector c it projects per-channel scale and shift
// and applies out = x * (1 + scale) + shift, where scale = c @ Ws and
// shift = c @ Wsh. Inputs are x0 = x [N, C] (the pre-normalized activations)
// and x1 = c [N, cond]; Ws, Wsh are [cond, C]. This is the modulation core of
// AdaLN-Zero (the zero-init projection is an initialization detail mapped in
// the arch builder, ADR-092; the op math is identical). Landing it extends the
// ADR-091 oracle to the AdaLN op class (E127/T127.1.0a) and unlocks every
// AdaLN-DiT model. torch oracle: x0 * (1 + x1 @ Ws) + (x1 @ Wsh).
type adaLNNode[T tensor.Float] struct {
	engine compute.Engine[T]
	ws     *graph.Parameter[T]
	wsh    *graph.Parameter[T]

	onePlusScale tn[T]
	xIn          tn[T]
	cIn          tn[T]
	saver        graph.Saver[T]
}

func newAdaLNNode[T tensor.Float](e compute.Engine[T], dim, cond int) (*adaLNNode[T], error) {
	mk := func(name string, scale float64) (*graph.Parameter[T], error) {
		data := make([]T, cond*dim)
		for i := range data {
			// Deterministic, non-uniform, small values so parameter gradients
			// are structurally informative (mirrors newLayerNormNode).
			data[i] = T(scale * (0.05 + 0.03*float64(i%7)))
		}
		v, err := newTensorOf([]int{cond, dim}, data)
		if err != nil {
			return nil, err
		}
		return graph.NewParameter[T](name, v, newTensorOf[T])
	}
	ws, err := mk("Ws", 1.0)
	if err != nil {
		return nil, err
	}
	wsh, err := mk("Wsh", -1.0)
	if err != nil {
		return nil, err
	}
	return &adaLNNode[T]{engine: e, ws: ws, wsh: wsh}, nil
}

func (n *adaLNNode[T]) OpType() string                     { return "AdaLN" }
func (n *adaLNNode[T]) Attributes() map[string]interface{} { return nil }
func (n *adaLNNode[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{n.ws, n.wsh}
}
func (n *adaLNNode[T]) SetSaver(s graph.Saver[T]) { n.saver = s }

func (n *adaLNNode[T]) OutputShape() []int {
	if n.onePlusScale == nil {
		return nil
	}
	return n.onePlusScale.Shape()
}

func (n *adaLNNode[T]) Forward(ctx context.Context, inputs ...tn[T]) (tn[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("AdaLN: want 2 inputs (x, c), got %d", len(inputs))
	}
	x, c := inputs[0], inputs[1]
	e := n.engine
	scale, err := e.MatMul(ctx, c, n.ws.Value)
	if err != nil {
		return nil, err
	}
	shift, err := e.MatMul(ctx, c, n.wsh.Value)
	if err != nil {
		return nil, err
	}
	onePlusScale, err := e.AddScalar(ctx, scale, T(1))
	if err != nil {
		return nil, err
	}
	n.onePlusScale = onePlusScale
	n.xIn = x
	n.cIn = c
	if n.saver != nil {
		n.saver.SaveForBackward(onePlusScale, x, c)
	}
	xs, err := e.Mul(ctx, x, onePlusScale)
	if err != nil {
		return nil, err
	}
	return e.Add(ctx, xs, shift)
}

func (n *adaLNNode[T]) Backward(ctx context.Context, _ types.BackwardMode, g tn[T], _ ...tn[T]) ([]tn[T], error) {
	if n.onePlusScale == nil {
		return nil, errors.New("AdaLN: Backward called before Forward")
	}
	e := n.engine
	// out = x*(1+scale) + shift.
	// dx = g * (1 + scale).
	dx, err := e.Mul(ctx, g, n.onePlusScale)
	if err != nil {
		return nil, err
	}
	// dscale = g * x ; dWs = c^T @ dscale.
	dscale, err := e.Mul(ctx, g, n.xIn)
	if err != nil {
		return nil, err
	}
	cT, err := e.Transpose(ctx, n.cIn, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dWs, err := e.MatMul(ctx, cT, dscale)
	if err != nil {
		return nil, err
	}
	if err := n.ws.AddGradient(dWs); err != nil {
		return nil, err
	}
	// dshift = g ; dWsh = c^T @ g.
	dWsh, err := e.MatMul(ctx, cT, g)
	if err != nil {
		return nil, err
	}
	if err := n.wsh.AddGradient(dWsh); err != nil {
		return nil, err
	}
	// dc = dscale @ Ws^T + g @ Wsh^T.
	wsT, err := e.Transpose(ctx, n.ws.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dcScale, err := e.MatMul(ctx, dscale, wsT)
	if err != nil {
		return nil, err
	}
	wshT, err := e.Transpose(ctx, n.wsh.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dcShift, err := e.MatMul(ctx, g, wshT)
	if err != nil {
		return nil, err
	}
	dc, err := e.Add(ctx, dcScale, dcShift)
	if err != nil {
		return nil, err
	}
	return []tn[T]{dx, dc}, nil
}

// --- timestep sinusoidal embedding --------------------------------------------

// timestepEmbedNode is the sinusoidal frequency embedding at the head of every
// diffusion-DiT timestep embedder: from scalar timesteps t [N, 1] it produces
// [N, 2H] = concat(sin(t @ freqs), cos(t @ freqs)) over a learned (here: leaf)
// frequency row freqs [1, H]. The downstream MLP is a plain Linear (already
// covered by MatMul), so this op isolates the sinusoidal piece. Landing it
// extends the ADR-091 oracle to the timestep-embedding op class
// (E127/T127.1.0a). torch oracle:
// torch.cat([torch.sin(x0 @ freqs), torch.cos(x0 @ freqs)], dim=1).
type timestepEmbedNode[T tensor.Float] struct {
	engine compute.Engine[T]
	freqs  *graph.Parameter[T] // [1, H]
	half   int                 // H

	sinv  tn[T] // sin(arg) [N, H]
	cosv  tn[T] // cos(arg) [N, H]
	tIn   tn[T]
	saver graph.Saver[T]
}

func newTimestepEmbedNode[T tensor.Float](e compute.Engine[T], half int) (*timestepEmbedNode[T], error) {
	data := make([]T, half)
	for j := 0; j < half; j++ {
		// Deterministic, moderate frequencies so sin/cos curvature is exercised
		// across the input domain (well-conditioned central differences).
		data[j] = T(0.5 + 0.4*float64(j))
	}
	v, err := newTensorOf([]int{1, half}, data)
	if err != nil {
		return nil, err
	}
	freqs, err := graph.NewParameter[T]("freqs", v, newTensorOf[T])
	if err != nil {
		return nil, err
	}
	return &timestepEmbedNode[T]{engine: e, freqs: freqs, half: half}, nil
}

func (n *timestepEmbedNode[T]) OpType() string                     { return "TimestepEmbed" }
func (n *timestepEmbedNode[T]) Attributes() map[string]interface{} { return nil }
func (n *timestepEmbedNode[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{n.freqs}
}
func (n *timestepEmbedNode[T]) SetSaver(s graph.Saver[T]) { n.saver = s }

func (n *timestepEmbedNode[T]) OutputShape() []int {
	if n.sinv == nil {
		return nil
	}
	s := n.sinv.Shape()
	return []int{s[0], 2 * s[1]}
}

func (n *timestepEmbedNode[T]) Forward(ctx context.Context, inputs ...tn[T]) (tn[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TimestepEmbed: want 1 input (t), got %d", len(inputs))
	}
	t := inputs[0]
	e := n.engine
	arg, err := e.MatMul(ctx, t, n.freqs.Value) // [N,1] @ [1,H] -> [N,H]
	if err != nil {
		return nil, err
	}
	sinv, err := e.Sin(ctx, arg)
	if err != nil {
		return nil, err
	}
	cosv, err := e.Cos(ctx, arg)
	if err != nil {
		return nil, err
	}
	n.sinv, n.cosv, n.tIn = sinv, cosv, t
	if n.saver != nil {
		n.saver.SaveForBackward(sinv, cosv, t)
	}
	return e.Concat(ctx, []tn[T]{sinv, cosv}, 1)
}

func (n *timestepEmbedNode[T]) Backward(ctx context.Context, _ types.BackwardMode, g tn[T], _ ...tn[T]) ([]tn[T], error) {
	if n.sinv == nil {
		return nil, errors.New("TimestepEmbed: Backward called before Forward")
	}
	e := n.engine
	// Split the [N,2H] upstream gradient into the sin and cos halves.
	parts, err := e.Split(ctx, g, 2, 1)
	if err != nil {
		return nil, err
	}
	gs, gc := parts[0], parts[1]
	// arg = t @ freqs;  d/darg[sin] = cos(arg), d/darg[cos] = -sin(arg).
	// darg = gs*cos(arg) - gc*sin(arg).
	a1, err := e.Mul(ctx, gs, n.cosv)
	if err != nil {
		return nil, err
	}
	a2, err := e.Mul(ctx, gc, n.sinv)
	if err != nil {
		return nil, err
	}
	darg, err := e.Sub(ctx, a1, a2)
	if err != nil {
		return nil, err
	}
	// dt = darg @ freqs^T  ([N,H] @ [H,1] -> [N,1]).
	fT, err := e.Transpose(ctx, n.freqs.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dt, err := e.MatMul(ctx, darg, fT)
	if err != nil {
		return nil, err
	}
	// dfreqs = t^T @ darg  ([1,N] @ [N,H] -> [1,H]).
	tT, err := e.Transpose(ctx, n.tIn, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dfreqs, err := e.MatMul(ctx, tT, darg)
	if err != nil {
		return nil, err
	}
	if err := n.freqs.AddGradient(dfreqs); err != nil {
		return nil, err
	}
	return []tn[T]{dt}, nil
}

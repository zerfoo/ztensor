package gradcheck

import (
	"context"
	"errors"
	"fmt"

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

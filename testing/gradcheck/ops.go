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

// t64 abbreviates the tensor type used throughout the checker.
type t64 = *tensor.TensorNumeric[float64]

// opNode is a graph.Node wrapper around compute.Engine operations: Forward
// runs an engine op, Backward applies the analytic gradient (itself built
// from engine ops where possible, so the engine kernels are what gradcheck
// exercises). Like production nodes, it caches its forward output for use in
// Backward -- which is exactly why the checker constructs a fresh instance
// per evaluation.
type opNode struct {
	graph.NoParameters[float64]
	opType string
	fwd    func(ctx context.Context, inputs []t64) (t64, error)
	bwd    func(ctx context.Context, g t64, inputs []t64, out t64) ([]t64, error)
	out    t64 // cached forward output
}

func (n *opNode) OpType() string                     { return n.opType }
func (n *opNode) Attributes() map[string]interface{} { return nil }
func (n *opNode) OutputShape() []int {
	if n.out == nil {
		return nil
	}
	return n.out.Shape()
}

func (n *opNode) Forward(ctx context.Context, inputs ...t64) (t64, error) {
	y, err := n.fwd(ctx, inputs)
	if err != nil {
		return nil, err
	}
	n.out = y
	return y, nil
}

func (n *opNode) Backward(ctx context.Context, _ types.BackwardMode, g t64, inputs ...t64) ([]t64, error) {
	if n.out == nil {
		return nil, errors.New(n.opType + ": Backward called before Forward")
	}
	return n.bwd(ctx, g, inputs, n.out)
}

type engineT = compute.Engine[float64]

// unary builds an opNode for a single-input op.
func unary(
	name string,
	fwd func(ctx context.Context, x t64) (t64, error),
	bwd func(ctx context.Context, g, x, y t64) (t64, error),
) *opNode {
	return &opNode{
		opType: name,
		fwd: func(ctx context.Context, in []t64) (t64, error) {
			if len(in) != 1 {
				return nil, fmt.Errorf("%s: want 1 input, got %d", name, len(in))
			}
			return fwd(ctx, in[0])
		},
		bwd: func(ctx context.Context, g t64, in []t64, out t64) ([]t64, error) {
			dx, err := bwd(ctx, g, in[0], out)
			if err != nil {
				return nil, err
			}
			return []t64{dx}, nil
		},
	}
}

// binary builds an opNode for a two-input op.
func binary(
	name string,
	fwd func(ctx context.Context, a, b t64) (t64, error),
	bwd func(ctx context.Context, g, a, b, y t64) (t64, t64, error),
) *opNode {
	return &opNode{
		opType: name,
		fwd: func(ctx context.Context, in []t64) (t64, error) {
			if len(in) != 2 {
				return nil, fmt.Errorf("%s: want 2 inputs, got %d", name, len(in))
			}
			return fwd(ctx, in[0], in[1])
		},
		bwd: func(ctx context.Context, g t64, in []t64, out t64) ([]t64, error) {
			da, db, err := bwd(ctx, g, in[0], in[1], out)
			if err != nil {
				return nil, err
			}
			return []t64{da, db}, nil
		},
	}
}

// --- elementwise binary ops -------------------------------------------------

func newAddNode(e engineT) *opNode {
	return binary("Add",
		func(ctx context.Context, a, b t64) (t64, error) { return e.Add(ctx, a, b) },
		func(_ context.Context, g, _, _, _ t64) (t64, t64, error) {
			return g.Copy(), g.Copy(), nil
		})
}

func newSubNode(e engineT) *opNode {
	return binary("Sub",
		func(ctx context.Context, a, b t64) (t64, error) { return e.Sub(ctx, a, b) },
		func(ctx context.Context, g, _, _, _ t64) (t64, t64, error) {
			db, err := e.MulScalar(ctx, g, -1)
			if err != nil {
				return nil, nil, err
			}
			return g.Copy(), db, nil
		})
}

func newMulNode(e engineT) *opNode {
	return binary("Mul",
		func(ctx context.Context, a, b t64) (t64, error) { return e.Mul(ctx, a, b) },
		func(ctx context.Context, g, a, b, _ t64) (t64, t64, error) {
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

func newDivNode(e engineT) *opNode {
	return binary("Div",
		func(ctx context.Context, a, b t64) (t64, error) { return e.Div(ctx, a, b) },
		func(ctx context.Context, g, _, b, y t64) (t64, t64, error) {
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

func newPowNode(e engineT) *opNode {
	return binary("Pow",
		func(ctx context.Context, a, b t64) (t64, error) { return e.Pow(ctx, a, b) },
		func(ctx context.Context, g, a, b, y t64) (t64, t64, error) {
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

func newTanhNode(e engineT) *opNode {
	return unary("Tanh",
		func(ctx context.Context, x t64) (t64, error) { return e.Tanh(ctx, x) },
		// TanhPrime is the engine's fused dtanh kernel: upstream * (1 - tanh(x)^2).
		func(ctx context.Context, g, x, _ t64) (t64, error) { return e.TanhPrime(ctx, x, g) })
}

func newSigmoidNode(e engineT) *opNode {
	ops := e.Ops()
	return unary("Sigmoid",
		func(ctx context.Context, x t64) (t64, error) { return e.UnaryOp(ctx, x, ops.Sigmoid) },
		func(ctx context.Context, g, x, _ t64) (t64, error) {
			sg, err := e.UnaryOp(ctx, x, ops.SigmoidGrad)
			if err != nil {
				return nil, err
			}
			return e.Mul(ctx, g, sg)
		})
}

func newReLUNode(e engineT) *opNode {
	ops := e.Ops()
	return unary("ReLU",
		func(ctx context.Context, x t64) (t64, error) { return e.UnaryOp(ctx, x, ops.ReLU) },
		func(ctx context.Context, g, x, _ t64) (t64, error) {
			rg, err := e.UnaryOp(ctx, x, ops.ReLUGrad)
			if err != nil {
				return nil, err
			}
			return e.Mul(ctx, g, rg)
		})
}

func newLeakyReLUNode(e engineT, alpha float64) *opNode {
	ops := e.Ops()
	return unary("LeakyReLU",
		func(ctx context.Context, x t64) (t64, error) {
			return e.UnaryOp(ctx, x, func(v float64) float64 { return ops.LeakyReLU(v, alpha) })
		},
		func(ctx context.Context, g, x, _ t64) (t64, error) {
			lg, err := e.UnaryOp(ctx, x, func(v float64) float64 { return ops.LeakyReLUGrad(v, alpha) })
			if err != nil {
				return nil, err
			}
			return e.Mul(ctx, g, lg)
		})
}

func newExpNode(e engineT) *opNode {
	return unary("Exp",
		func(ctx context.Context, x t64) (t64, error) { return e.Exp(ctx, x) },
		func(ctx context.Context, g, _, y t64) (t64, error) { return e.Mul(ctx, g, y) })
}

func newLogNode(e engineT) *opNode {
	return unary("Log",
		func(ctx context.Context, x t64) (t64, error) { return e.Log(ctx, x) },
		func(ctx context.Context, g, x, _ t64) (t64, error) { return e.Div(ctx, g, x) })
}

func newSqrtNode(e engineT) *opNode {
	return unary("Sqrt",
		func(ctx context.Context, x t64) (t64, error) { return e.Sqrt(ctx, x) },
		func(ctx context.Context, g, _, y t64) (t64, error) {
			gy, err := e.Div(ctx, g, y)
			if err != nil {
				return nil, err
			}
			return e.MulScalar(ctx, gy, 0.5)
		})
}

func newRsqrtNode(e engineT) *opNode {
	return unary("Rsqrt",
		func(ctx context.Context, x t64) (t64, error) { return e.Rsqrt(ctx, x) },
		func(ctx context.Context, g, x, y t64) (t64, error) {
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

func newSinNode(e engineT) *opNode {
	return unary("Sin",
		func(ctx context.Context, x t64) (t64, error) { return e.Sin(ctx, x) },
		func(ctx context.Context, g, x, _ t64) (t64, error) {
			c, err := e.Cos(ctx, x)
			if err != nil {
				return nil, err
			}
			return e.Mul(ctx, g, c)
		})
}

func newCosNode(e engineT) *opNode {
	return unary("Cos",
		func(ctx context.Context, x t64) (t64, error) { return e.Cos(ctx, x) },
		func(ctx context.Context, g, x, _ t64) (t64, error) {
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

func newAddScalarNode(e engineT, c float64) *opNode {
	return unary("AddScalar",
		func(ctx context.Context, x t64) (t64, error) { return e.AddScalar(ctx, x, c) },
		func(_ context.Context, g, _, _ t64) (t64, error) { return g.Copy(), nil })
}

func newMulScalarNode(e engineT, c float64) *opNode {
	return unary("MulScalar",
		func(ctx context.Context, x t64) (t64, error) { return e.MulScalar(ctx, x, c) },
		func(ctx context.Context, g, _, _ t64) (t64, error) { return e.MulScalar(ctx, g, c) })
}

// --- matmul-like and shape ops ----------------------------------------------

func newMatMulNode(e engineT) *opNode {
	return binary("MatMul",
		func(ctx context.Context, a, b t64) (t64, error) { return e.MatMul(ctx, a, b) },
		func(ctx context.Context, g, a, b, _ t64) (t64, t64, error) {
			// dA = g @ B^T. Use the fused MatMulTransposeB kernel when the
			// engine provides it (optional interface); otherwise transpose.
			var da t64
			var err error
			if tb, ok := e.(compute.TransposeBMatMuler[float64]); ok {
				da, err = tb.MatMulTransposeB(ctx, g, b)
			} else {
				var bt t64
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

func newTransposeNode(e engineT) *opNode {
	return unary("Transpose",
		func(ctx context.Context, x t64) (t64, error) { return e.Transpose(ctx, x, []int{1, 0}) },
		func(ctx context.Context, g, _, _ t64) (t64, error) { return e.Transpose(ctx, g, []int{1, 0}) })
}

func newReshapeNode(e engineT, shape []int) *opNode {
	return unary("Reshape",
		func(ctx context.Context, x t64) (t64, error) { return e.Reshape(ctx, x, shape) },
		func(ctx context.Context, g, x, _ t64) (t64, error) { return e.Reshape(ctx, g, x.Shape()) })
}

func newHadamardNode(e engineT) *opNode {
	return unary("HadamardTransform",
		func(ctx context.Context, x t64) (t64, error) { return e.HadamardTransform(ctx, x) },
		// The normalized Walsh-Hadamard matrix is symmetric, so dX = H^T g = H g.
		func(ctx context.Context, g, _, _ t64) (t64, error) { return e.HadamardTransform(ctx, g) })
}

// --- softmax and reductions ---------------------------------------------------

func newSoftmaxNode(e engineT, axis int) *opNode {
	return unary("Softmax",
		func(ctx context.Context, x t64) (t64, error) { return e.Softmax(ctx, x, axis) },
		func(ctx context.Context, g, _, y t64) (t64, error) {
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

func newReduceSumNode(e engineT, axis int) *opNode {
	return unary("ReduceSum",
		func(ctx context.Context, x t64) (t64, error) { return e.ReduceSum(ctx, x, axis, true) },
		func(ctx context.Context, g, x, _ t64) (t64, error) {
			return e.Repeat(ctx, g, axis, x.Shape()[axis])
		})
}

func newReduceMeanNode(e engineT, axis int) *opNode {
	return unary("ReduceMean",
		func(ctx context.Context, x t64) (t64, error) { return e.ReduceMean(ctx, x, axis, true) },
		func(ctx context.Context, g, x, _ t64) (t64, error) {
			n := x.Shape()[axis]
			r, err := e.Repeat(ctx, g, axis, n)
			if err != nil {
				return nil, err
			}
			return e.DivScalar(ctx, r, float64(n))
		})
}

// newReduceMaxNode reduces along axis 1 of a 2D input with keepDims=true.
// The gradient flows only to the (unique) argmax per row; the OpInfo domain
// relies on continuous sampling to avoid ties (the non-differentiable point),
// as PyTorch's OpInfo does for max-like ops.
func newReduceMaxNode(e engineT) *opNode {
	return unary("ReduceMax",
		func(ctx context.Context, x t64) (t64, error) { return e.ReduceMax(ctx, x, 1, true) },
		func(_ context.Context, g, x, _ t64) (t64, error) {
			shape := x.Shape()
			if len(shape) != 2 {
				return nil, errors.New("ReduceMax wrapper: input must be 2D")
			}
			rows, cols := shape[0], shape[1]
			xd := x.Data()
			dx := make([]float64, rows*cols)
			for i := 0; i < rows; i++ {
				arg := 0
				for j := 1; j < cols; j++ {
					if xd[i*cols+j] > xd[i*cols+arg] {
						arg = j
					}
				}
				dx[i*cols+arg] = g.Data()[i]
			}
			return tensor.New[float64]([]int{rows, cols}, dx)
		})
}

// --- layernorm-like (with trainable parameters) -------------------------------

// layerNormNode normalizes the last axis of a 2D input and applies a
// trainable elementwise affine (gamma, beta), mirroring the production
// LayerNorm whose cached-statistics GPU bug motivated this harness. It caches
// xhat and the inverse stddev in Forward and consumes them in Backward.
type layerNormNode struct {
	engine compute.Engine[float64]
	gamma  *graph.Parameter[float64]
	beta   *graph.Parameter[float64]
	eps    float64

	xhat t64
	inv  t64
}

func newTensor64(shape []int, data []float64) (t64, error) {
	return tensor.New[float64](shape, data)
}

func newLayerNormNode(e engineT, dim int) (*layerNormNode, error) {
	gammaData := make([]float64, dim)
	betaData := make([]float64, dim)
	for i := 0; i < dim; i++ {
		// Deterministic, non-uniform initial values so parameter gradients
		// are structurally informative.
		gammaData[i] = 0.8 + 0.1*float64(i)
		betaData[i] = -0.2 + 0.15*float64(i)
	}
	gv, err := newTensor64([]int{1, dim}, gammaData)
	if err != nil {
		return nil, err
	}
	bv, err := newTensor64([]int{1, dim}, betaData)
	if err != nil {
		return nil, err
	}
	gamma, err := graph.NewParameter[float64]("gamma", gv, newTensor64)
	if err != nil {
		return nil, err
	}
	beta, err := graph.NewParameter[float64]("beta", bv, newTensor64)
	if err != nil {
		return nil, err
	}
	return &layerNormNode{engine: e, gamma: gamma, beta: beta, eps: 1e-5}, nil
}

func (n *layerNormNode) OpType() string { return "LayerNorm" }
func (n *layerNormNode) Attributes() map[string]interface{} {
	return map[string]interface{}{"epsilon": n.eps}
}
func (n *layerNormNode) Parameters() []*graph.Parameter[float64] {
	return []*graph.Parameter[float64]{n.gamma, n.beta}
}

func (n *layerNormNode) OutputShape() []int {
	if n.xhat == nil {
		return nil
	}
	return n.xhat.Shape()
}

func (n *layerNormNode) Forward(ctx context.Context, inputs ...t64) (t64, error) {
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
	veps, err := e.AddScalar(ctx, variance, n.eps)
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
	scaled, err := e.Mul(ctx, xhat, n.gamma.Value)
	if err != nil {
		return nil, err
	}
	return e.Add(ctx, scaled, n.beta.Value)
}

func (n *layerNormNode) Backward(ctx context.Context, _ types.BackwardMode, g t64, _ ...t64) ([]t64, error) {
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
	return []t64{dx}, nil
}

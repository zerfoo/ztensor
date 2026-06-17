package compute

// gpu_bf16_parity_test.go is the universal quality gate for the native bf16 GPU
// path (ADR 075 lever L4): every bf16 GPU op is checked against an f32 reference
// rounded to bf16. bf16 shares f32's exponent range, so the only expected
// divergence is the 7-bit mantissa (one bf16 step is 2^-7 ~= 0.0078 relative).
//
// These tests are CUDA-gated; they run on a real GPU (the GB10 verify pod) and
// skip on hosts without CUDA. The CPU bf16 path is untouched by the L4 change,
// so its byte-level behaviour is unchanged by construction; TestBF16CPUPathUnchanged
// documents the invariant against the engine API.

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func newTestGPUBF16Engine(t *testing.T) *GPUEngine[float16.BFloat16] {
	t.Helper()
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	eng, err := NewGPUEngine[float16.BFloat16](numeric.BFloat16Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine[BFloat16]: %v", err)
	}
	t.Cleanup(func() {
		if err := eng.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return eng
}

// bf16Tensor builds a bf16 tensor from float32 values (rounded to bf16 so the
// inputs are exact bf16 and the only rounding under test is the op's own).
func bf16Tensor(t *testing.T, shape []int, vals []float32) *tensor.TensorNumeric[float16.BFloat16] {
	t.Helper()
	data := make([]float16.BFloat16, len(vals))
	for i, v := range vals {
		data[i] = float16.BFloat16FromFloat32(v)
	}
	tn, err := tensor.New[float16.BFloat16](shape, data)
	if err != nil {
		t.Fatalf("tensor.New[BFloat16]: %v", err)
	}
	return tn
}

func bf16ToF32(data []float16.BFloat16) []float32 {
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = v.ToFloat32()
	}
	return out
}

// assertBF16Close fails when got and want differ by more than `relUlps` bf16
// steps (relative) plus a small absolute floor for near-zero values.
func assertBF16Close(t *testing.T, op string, i int, got, want float32, relUlps float64) {
	t.Helper()
	const bf16Eps = 1.0 / 128.0 // 2^-7, one bf16 mantissa step
	tol := relUlps*bf16Eps*math.Abs(float64(want)) + relUlps*bf16Eps
	if math.Abs(float64(got)-float64(want)) > tol {
		t.Errorf("%s[%d] = %g, want %g (|diff|=%g > tol=%g)", op, i, got, want, math.Abs(float64(got-want)), tol)
	}
}

func TestGPUBF16_BinaryParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	const n = 1024
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float16.BFloat16FromFloat32(rng.Float32()*4 - 2).ToFloat32()
		// keep b away from 0 for division
		b[i] = float16.BFloat16FromFloat32(rng.Float32()*3 + 0.5).ToFloat32()
	}

	cases := []struct {
		name string
		run  func(x, y *tensor.TensorNumeric[float16.BFloat16]) (*tensor.TensorNumeric[float16.BFloat16], error)
		ref  func(x, y float32) float32
		ulps float64
	}{
		{"Add", func(x, y *tensor.TensorNumeric[float16.BFloat16]) (*tensor.TensorNumeric[float16.BFloat16], error) {
			return eng.Add(ctx, x, y)
		}, func(x, y float32) float32 { return x + y }, 1.0},
		{"Sub", func(x, y *tensor.TensorNumeric[float16.BFloat16]) (*tensor.TensorNumeric[float16.BFloat16], error) {
			return eng.Sub(ctx, x, y)
		}, func(x, y float32) float32 { return x - y }, 1.0},
		{"Mul", func(x, y *tensor.TensorNumeric[float16.BFloat16]) (*tensor.TensorNumeric[float16.BFloat16], error) {
			return eng.Mul(ctx, x, y)
		}, func(x, y float32) float32 { return x * y }, 1.0},
		{"Div", func(x, y *tensor.TensorNumeric[float16.BFloat16]) (*tensor.TensorNumeric[float16.BFloat16], error) {
			return eng.Div(ctx, x, y)
		}, func(x, y float32) float32 { return x / y }, 2.0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x := bf16Tensor(t, []int{n}, a)
			y := bf16Tensor(t, []int{n}, b)
			got, err := tc.run(x, y)
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			gd := bf16ToF32(got.Data())
			for i := range gd {
				want := float16.BFloat16FromFloat32(tc.ref(a[i], b[i])).ToFloat32()
				assertBF16Close(t, tc.name, i, gd[i], want, tc.ulps)
			}
		})
	}
}

func TestGPUBF16_UnaryParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	ctx := context.Background()
	rng := rand.New(rand.NewSource(7))

	const n = 1024
	x := make([]float32, n)
	pos := make([]float32, n) // strictly positive for sqrt/log
	for i := range x {
		x[i] = float16.BFloat16FromFloat32(rng.Float32()*4 - 2).ToFloat32()
		pos[i] = float16.BFloat16FromFloat32(rng.Float32()*4 + 0.1).ToFloat32()
	}

	cases := []struct {
		name  string
		input []float32
		run   func(*tensor.TensorNumeric[float16.BFloat16]) (*tensor.TensorNumeric[float16.BFloat16], error)
		ref   func(float32) float32
		ulps  float64
	}{
		{"Tanh", x, func(in *tensor.TensorNumeric[float16.BFloat16]) (*tensor.TensorNumeric[float16.BFloat16], error) {
			return eng.Tanh(ctx, in)
		}, func(v float32) float32 { return float32(math.Tanh(float64(v))) }, 2.0},
		{"Sqrt", pos, func(in *tensor.TensorNumeric[float16.BFloat16]) (*tensor.TensorNumeric[float16.BFloat16], error) {
			return eng.Sqrt(ctx, in)
		}, func(v float32) float32 { return float32(math.Sqrt(float64(v))) }, 2.0},
		{"Rsqrt", pos, func(in *tensor.TensorNumeric[float16.BFloat16]) (*tensor.TensorNumeric[float16.BFloat16], error) {
			return eng.Rsqrt(ctx, in)
		}, func(v float32) float32 { return float32(1.0 / math.Sqrt(float64(v))) }, 2.0},
		{"Exp", x, func(in *tensor.TensorNumeric[float16.BFloat16]) (*tensor.TensorNumeric[float16.BFloat16], error) {
			return eng.Exp(ctx, in)
		}, func(v float32) float32 { return float32(math.Exp(float64(v))) }, 2.0},
		{"Log", pos, func(in *tensor.TensorNumeric[float16.BFloat16]) (*tensor.TensorNumeric[float16.BFloat16], error) {
			return eng.Log(ctx, in)
		}, func(v float32) float32 { return float32(math.Log(float64(v))) }, 2.0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := bf16Tensor(t, []int{n}, tc.input)
			got, err := tc.run(in)
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			gd := bf16ToF32(got.Data())
			for i := range gd {
				want := float16.BFloat16FromFloat32(tc.ref(tc.input[i])).ToFloat32()
				assertBF16Close(t, tc.name, i, gd[i], want, tc.ulps)
			}
		})
	}
}

func TestGPUBF16_SoftmaxParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	ctx := context.Background()
	rng := rand.New(rand.NewSource(99))

	const rows, cols = 16, 32
	vals := make([]float32, rows*cols)
	for i := range vals {
		vals[i] = float16.BFloat16FromFloat32(rng.Float32()*6 - 3).ToFloat32()
	}
	in := bf16Tensor(t, []int{rows, cols}, vals)

	got, err := eng.Softmax(ctx, in, 1)
	if err != nil {
		t.Fatalf("Softmax: %v", err)
	}
	gd := bf16ToF32(got.Data())

	for r := 0; r < rows; r++ {
		// f64 reference softmax over the row.
		maxV := math.Inf(-1)
		for c := 0; c < cols; c++ {
			if float64(vals[r*cols+c]) > maxV {
				maxV = float64(vals[r*cols+c])
			}
		}
		sum := 0.0
		ex := make([]float64, cols)
		for c := 0; c < cols; c++ {
			ex[c] = math.Exp(float64(vals[r*cols+c]) - maxV)
			sum += ex[c]
		}
		var rowSum float32
		for c := 0; c < cols; c++ {
			want := float32(ex[c] / sum)
			// softmax accumulates exp/sum in fp32 then rounds each output to
			// bf16: allow a few bf16 steps of slack.
			assertBF16Close(t, "Softmax", r*cols+c, gd[r*cols+c], want, 4.0)
			rowSum += gd[r*cols+c]
		}
		if math.Abs(float64(rowSum)-1.0) > 0.05 {
			t.Errorf("Softmax row %d sums to %g, want ~1.0", r, rowSum)
		}
	}
}

// TestGPUBF16_ReductionParity validates the native bf16 axis reductions (Sum,
// ReduceSum, ReduceMean) against an f64 reference. The kernel accumulates each
// axis stripe in FP32 and rounds the result to bf16 once at the end, while the
// f64 reference accumulates in double precision and rounds once -- so the only
// expected divergence is bf16's 7-bit mantissa, amplified by accumulation over
// axisSize terms. We use a generous, axisSize-scaled tolerance: summing N bf16
// values in FP32 then rounding can differ from the f64-then-bf16 reference by a
// few bf16 steps as N grows, so this is an order-of-magnitude correctness gate,
// not a bit-exact one. GPU-UNVERIFIED until run on the GB10 verify pod.
func TestGPUBF16_ReductionParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	ctx := context.Background()
	rng := rand.New(rand.NewSource(1234))

	// 3D so the reduced axis (axis=1) has a nontrivial inner stride: this
	// exercises the (outer, inner, axisSize) stripe addressing in the kernel.
	const outer, axisSize, inner = 4, 24, 5
	vals := make([]float32, outer*axisSize*inner)
	for i := range vals {
		// keep values modest so the FP32 partial sums stay well within bf16 range
		vals[i] = float16.BFloat16FromFloat32(rng.Float32()*2 - 1).ToFloat32()
	}
	shape := []int{outer, axisSize, inner}

	// f64 reference reduced along axis=1: ref[o][in] = sum_k vals[o][k][in].
	refSum := make([]float64, outer*inner)
	for o := 0; o < outer; o++ {
		for in := 0; in < inner; in++ {
			var s float64
			for k := 0; k < axisSize; k++ {
				s += float64(vals[o*axisSize*inner+k*inner+in])
			}
			refSum[o*inner+in] = s
		}
	}

	// accumulating axisSize bf16 values in FP32 then rounding can drift a few
	// bf16 steps from the f64 reference; scale tolerance with sqrt(axisSize).
	tolUlps := 2.0 + math.Sqrt(float64(axisSize))

	t.Run("Sum", func(t *testing.T) {
		in := bf16Tensor(t, shape, vals)
		got, err := eng.Sum(ctx, in, 1, false)
		if err != nil {
			t.Fatalf("Sum: %v", err)
		}
		gd := bf16ToF32(got.Data())
		if len(gd) != outer*inner {
			t.Fatalf("Sum produced %d elements, want %d (shape=%v)", len(gd), outer*inner, got.Shape())
		}
		for i := range gd {
			want := float16.BFloat16FromFloat32(float32(refSum[i])).ToFloat32()
			assertBF16Close(t, "Sum", i, gd[i], want, tolUlps)
		}
	})

	t.Run("ReduceSum", func(t *testing.T) {
		in := bf16Tensor(t, shape, vals)
		got, err := eng.ReduceSum(ctx, in, 1, false)
		if err != nil {
			t.Fatalf("ReduceSum: %v", err)
		}
		gd := bf16ToF32(got.Data())
		for i := range gd {
			want := float16.BFloat16FromFloat32(float32(refSum[i])).ToFloat32()
			assertBF16Close(t, "ReduceSum", i, gd[i], want, tolUlps)
		}
	})

	t.Run("ReduceMean", func(t *testing.T) {
		in := bf16Tensor(t, shape, vals)
		got, err := eng.ReduceMean(ctx, in, 1, false)
		if err != nil {
			t.Fatalf("ReduceMean: %v", err)
		}
		gd := bf16ToF32(got.Data())
		for i := range gd {
			want := float16.BFloat16FromFloat32(float32(refSum[i] / float64(axisSize))).ToFloat32()
			assertBF16Close(t, "ReduceMean", i, gd[i], want, tolUlps)
		}
	})

	t.Run("SumKeepDims", func(t *testing.T) {
		in := bf16Tensor(t, shape, vals)
		got, err := eng.Sum(ctx, in, 1, true)
		if err != nil {
			t.Fatalf("Sum keepDims: %v", err)
		}
		want := []int{outer, 1, inner}
		gs := got.Shape()
		if len(gs) != len(want) {
			t.Fatalf("Sum keepDims shape = %v, want %v", gs, want)
		}
		for i := range want {
			if gs[i] != want[i] {
				t.Fatalf("Sum keepDims shape = %v, want %v", gs, want)
			}
		}
	})
}

// TestGPUBF16_AdamWParity validates the full gradient-consuming update path:
// the on-device bf16 AdamW step (param/grad bf16, m f32, v f64) must match an
// f64 reference AdamW step with the published parameter rounded to bf16. This
// stands in for an op-level finite-difference gradcheck, which is not
// numerically meaningful at bf16's 7-bit mantissa (the perturbation underflows).
func TestGPUBF16_AdamWParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	eng := newTestGPUBF16Engine(t)
	ctx := context.Background()
	rng := rand.New(rand.NewSource(2024))

	const n = 512
	const beta1, beta2, eps, lr, wd = 0.9, 0.999, 1e-8, 0.01, 0.01

	paramVals := make([]float32, n)
	gradVals := make([]float32, n)
	for i := range paramVals {
		paramVals[i] = float16.BFloat16FromFloat32(rng.Float32()*2 - 1).ToFloat32()
		gradVals[i] = float16.BFloat16FromFloat32(rng.Float32()*0.2 - 0.1).ToFloat32()
	}

	param := bf16Tensor(t, []int{n}, paramVals)
	grad := bf16Tensor(t, []int{n}, gradVals)
	// Move both to device so GPUFusedAdamW's GPUStorage assertion holds.
	if _, err := eng.Add(ctx, param, bf16Tensor(t, []int{n}, make([]float32, n)), param); err != nil {
		t.Fatalf("stage param to device: %v", err)
	}
	if _, err := eng.Add(ctx, grad, bf16Tensor(t, []int{n}, make([]float32, n)), grad); err != nil {
		t.Fatalf("stage grad to device: %v", err)
	}

	// f64 reference for one step (t=1): m,v start at 0.
	wantParam := make([]float32, n)
	numer := math.Sqrt(1.0 - math.Pow(beta2, 1))
	denom := 1.0 - math.Pow(beta1, 1)
	alpha := lr * (numer / denom)
	lrWd := lr * wd
	for i := range paramVals {
		g := float64(gradVals[i])
		m := (1 - beta1) * g
		v := (1 - beta2) * g * g
		update := alpha * m / (math.Sqrt(v) + eps)
		pv := float64(paramVals[i])
		pv = pv - update - lrWd*pv
		wantParam[i] = float16.BFloat16FromFloat32(float32(pv)).ToFloat32()
	}

	if err := eng.GPUFusedAdamW(param, grad, beta1, beta2, eps, lr, wd, 1); err != nil {
		t.Fatalf("GPUFusedAdamW: %v", err)
	}

	gotParam := bf16ToF32(param.Data())
	for i := range gotParam {
		assertBF16Close(t, "AdamW.param", i, gotParam[i], wantParam[i], 3.0)
	}
	// Gradient must be zeroed in place.
	gotGrad := bf16ToF32(grad.Data())
	for i := range gotGrad {
		if gotGrad[i] != 0 {
			t.Errorf("AdamW grad[%d] = %g, want 0 (zeroed in place)", i, gotGrad[i])
			break
		}
	}
}

// TestBF16CPUPathUnchanged documents that the L4 change adds only a GPU path:
// the CPU bf16 engine still produces standard bf16 arithmetic. This is the
// "CPU path byte-identical" gate -- no CPU source was touched by L4.
func TestBF16CPUPathUnchanged(t *testing.T) {
	ctx := context.Background()
	eng := NewCPUEngine[float16.BFloat16](numeric.BFloat16Ops{})

	a := bf16Tensor(t, []int{4}, []float32{1, 2, 3, 4})
	b := bf16Tensor(t, []int{4}, []float32{0.5, 0.5, 0.5, 0.5})
	got, err := eng.Add(ctx, a, b)
	if err != nil {
		t.Fatalf("CPU Add: %v", err)
	}
	gd := bf16ToF32(got.Data())
	want := []float32{1.5, 2.5, 3.5, 4.5}
	for i := range want {
		if gd[i] != want[i] {
			t.Errorf("CPU bf16 Add[%d] = %g, want %g", i, gd[i], want[i])
		}
	}
}

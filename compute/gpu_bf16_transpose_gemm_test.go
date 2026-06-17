package compute

// Parity tests for the native bf16 transpose-variant GEMMs added for the bf16
// GPU backward pass (ADR 075 lever L4):
//
//	MatMulTransposeB: C = A * B^T  (dX = dY * W),  A=[m,k], B=[n,k]
//	MatMulTransposeA: C = A^T * B  (dW = X^T * dY), A=[k,m], B=[k,n]
//
// Before this change the bf16 engine routed both through GPUEngine.Transpose,
// which falls back to the CPU engine for all non-float32 types -- and the
// transpose-B fallback handed rank-3 axes ([0,2,1]) to a 2D weight, raising
// "number of axes 3 must match tensor dimensions 2". These tests exercise the
// 2D (weight-gradient) shapes that triggered that crash, plus a batched case,
// against an f32 reference rounded to bf16. CUDA-gated.

import (
	"context"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/tensor"
)

func TestGPUBF16_MatMulTransposeBParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	ctx := context.Background()

	// C[m,n] = A[m,k] * B[n,k]^T. Exercises the 2D weight shape that crashed.
	cases := []struct{ m, k, n int }{
		{2, 3, 4},
		{4, 4, 4},
		{8, 16, 8},
		{1, 12, 32}, // single-row grad x 2D weight, the CrossAsset Linear dX shape
	}
	for _, c := range cases {
		aVals := ramp(c.m * c.k)
		bVals := ramp(c.n * c.k)
		a := bf16Tensor(t, []int{c.m, c.k}, aVals)
		b := bf16Tensor(t, []int{c.n, c.k}, bVals)

		got, err := eng.MatMulTransposeB(ctx, a, b)
		if err != nil {
			t.Fatalf("MatMulTransposeB (%dx%dx%d): %v", c.m, c.k, c.n, err)
		}
		gd := bf16ToF32(got.Data())
		if want := []int{c.m, c.n}; !shapeEq(got.Shape(), want) {
			t.Fatalf("MatMulTransposeB shape = %v, want %v", got.Shape(), want)
		}
		// Reference: C[i,j] = sum_l A[i,l] * B[j,l].
		for i := 0; i < c.m; i++ {
			for j := 0; j < c.n; j++ {
				var sum float32
				for l := 0; l < c.k; l++ {
					sum += aVals[i*c.k+l] * bVals[j*c.k+l]
				}
				want := float16.BFloat16FromFloat32(sum).ToFloat32()
				assertBF16Close(t, "MatMulTransposeB", i*c.n+j, gd[i*c.n+j], want, 4.0)
			}
		}
	}
}

func TestGPUBF16_MatMulTransposeAParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	ctx := context.Background()

	// C[m,n] = A[k,m]^T * B[k,n]. The dW gradient shape (X^T * dY).
	cases := []struct{ k, m, n int }{
		{3, 2, 4},
		{4, 4, 4},
		{16, 8, 8},
		{12, 32, 1}, // batch=12 samples, in=32, out=1 -- CrossAsset Linear dW shape
	}
	for _, c := range cases {
		aVals := ramp(c.k * c.m)
		bVals := ramp(c.k * c.n)
		a := bf16Tensor(t, []int{c.k, c.m}, aVals)
		b := bf16Tensor(t, []int{c.k, c.n}, bVals)

		got, err := eng.MatMulTransposeA(ctx, a, b)
		if err != nil {
			t.Fatalf("MatMulTransposeA (%dx%dx%d): %v", c.k, c.m, c.n, err)
		}
		gd := bf16ToF32(got.Data())
		if want := []int{c.m, c.n}; !shapeEq(got.Shape(), want) {
			t.Fatalf("MatMulTransposeA shape = %v, want %v", got.Shape(), want)
		}
		// Reference: C[i,j] = sum_l A[l,i] * B[l,j].
		for i := 0; i < c.m; i++ {
			for j := 0; j < c.n; j++ {
				var sum float32
				for l := 0; l < c.k; l++ {
					sum += aVals[l*c.m+i] * bVals[l*c.n+j]
				}
				want := float16.BFloat16FromFloat32(sum).ToFloat32()
				assertBF16Close(t, "MatMulTransposeA", i*c.n+j, gd[i*c.n+j], want, 4.0)
			}
		}
	}
}

func TestGPUBF16_MatMulTransposeBBatchedParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	ctx := context.Background()

	// Batched C[b,m,n] = A[b,m,k] * B[b,n,k]^T.
	const batch, m, k, n = 3, 2, 4, 3
	aVals := ramp(batch * m * k)
	bVals := ramp(batch * n * k)
	a := bf16Tensor(t, []int{batch, m, k}, aVals)
	b := bf16Tensor(t, []int{batch, n, k}, bVals)

	got, err := eng.MatMulTransposeB(ctx, a, b)
	if err != nil {
		t.Fatalf("batched MatMulTransposeB: %v", err)
	}
	gd := bf16ToF32(got.Data())
	if want := []int{batch, m, n}; !shapeEq(got.Shape(), want) {
		t.Fatalf("batched MatMulTransposeB shape = %v, want %v", got.Shape(), want)
	}
	for bi := 0; bi < batch; bi++ {
		aB := aVals[bi*m*k : (bi+1)*m*k]
		bB := bVals[bi*n*k : (bi+1)*n*k]
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var sum float32
				for l := 0; l < k; l++ {
					sum += aB[i*k+l] * bB[j*k+l]
				}
				want := float16.BFloat16FromFloat32(sum).ToFloat32()
				idx := bi*m*n + i*n + j
				assertBF16Close(t, "MatMulTransposeB(batched)", idx, gd[idx], want, 4.0)
			}
		}
	}
}

func TestGPUBF16_TransposeParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	ctx := context.Background()

	// bf16 transpose is a pure bitwise element move (no arithmetic), so the
	// output values must match the reference EXACTLY. This exercises the native
	// bf16 GPU transpose kernels (Transpose2DBF16/NDBF16) that keep bf16
	// transposes on-device (CUDA-graph-capturable) instead of the CPU fallback.

	// 2D: [rows, cols] -> [cols, rows] via axes [1,0].
	t.Run("2D", func(t *testing.T) {
		const rows, cols = 5, 7
		vals := ramp(rows * cols)
		x := bf16Tensor(t, []int{rows, cols}, vals)
		got, err := eng.Transpose(ctx, x, []int{1, 0})
		if err != nil {
			t.Fatalf("Transpose 2D: %v", err)
		}
		if want := []int{cols, rows}; !shapeEq(got.Shape(), want) {
			t.Fatalf("2D transpose shape = %v, want %v", got.Shape(), want)
		}
		gd := bf16ToF32(got.Data())
		for i := 0; i < cols; i++ {
			for j := 0; j < rows; j++ {
				want := vals[j*cols+i] // input[j,i] -> output[i,j]
				if gd[i*rows+j] != want {
					t.Fatalf("2D[%d,%d] = %g, want %g", i, j, gd[i*rows+j], want)
				}
			}
		}
	})

	// 3D: [d0,d1,d2] -> [d0,d2,d1] via axes [0,2,1] (the QKL2Norm-style case).
	t.Run("3D_021", func(t *testing.T) {
		const d0, d1, d2 = 4, 12, 64
		vals := ramp(d0 * d1 * d2)
		x := bf16Tensor(t, []int{d0, d1, d2}, vals)
		got, err := eng.Transpose(ctx, x, []int{0, 2, 1})
		if err != nil {
			t.Fatalf("Transpose 3D: %v", err)
		}
		if want := []int{d0, d2, d1}; !shapeEq(got.Shape(), want) {
			t.Fatalf("3D transpose shape = %v, want %v", got.Shape(), want)
		}
		gd := bf16ToF32(got.Data())
		for a := 0; a < d0; a++ {
			for c := 0; c < d2; c++ {
				for b := 0; b < d1; b++ {
					want := vals[a*d1*d2+b*d2+c] // input[a,b,c]
					gotV := gd[a*d2*d1+c*d1+b]   // output[a,c,b]
					if gotV != want {
						t.Fatalf("3D[%d,%d,%d] = %g, want %g", a, c, b, gotV, want)
					}
				}
			}
		}
	})
}

func TestGPUBF16_ReshapeStaysOnDevice(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	ctx := context.Background()

	// A GPU-resident bf16 tensor (output of a GPU op) reshaped must stay a
	// GPUStorage view, not bounce to the CPU engine -- otherwise the next op
	// (e.g. Transpose feeding QKL2Norm) is forced onto the CPU and breaks
	// CUDA-graph capture.
	vals := ramp(2 * 3 * 4)
	x := bf16Tensor(t, []int{2, 3, 4}, vals)
	// Mul produces a device-resident GPUStorage[bf16] result.
	gpuRes, err := eng.Mul(ctx, x, x)
	if err != nil {
		t.Fatalf("Mul: %v", err)
	}
	if _, ok := gpuRes.GetStorage().(*tensor.GPUStorage[float16.BFloat16]); !ok {
		t.Fatalf("precondition: Mul result storage = %T, want *GPUStorage[bf16]", gpuRes.GetStorage())
	}

	r, err := eng.Reshape(ctx, gpuRes, []int{6, 4}, nil)
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	if want := []int{6, 4}; !shapeEq(r.Shape(), want) {
		t.Fatalf("Reshape shape = %v, want %v", r.Shape(), want)
	}
	if _, ok := r.GetStorage().(*tensor.GPUStorage[float16.BFloat16]); !ok {
		t.Fatalf("Reshape result storage = %T, want *GPUStorage[bf16] (must stay on device)", r.GetStorage())
	}
	// Data preserved (x*x).
	rd := bf16ToF32(r.Data())
	for i := range vals {
		want := float16.BFloat16FromFloat32(vals[i] * vals[i]).ToFloat32()
		if rd[i] != want {
			t.Fatalf("Reshape[%d] = %g, want %g", i, rd[i], want)
		}
	}
}

func TestGPUBF16_BroadcastAndScalarParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	ctx := context.Background()

	// These are the QKL2Norm ops that broke CUDA-graph capture by falling to the
	// CPU engine: a column-broadcast Mul (x[M,D] * inv[M,1]) and AddScalar.
	const M, D = 6, 4
	xv := ramp(M * D)
	invv := make([]float32, M) // [M,1] column vector
	for i := range invv {
		invv[i] = float32((i%5)+1) * 0.25 // bf16-exact
	}
	x := bf16Tensor(t, []int{M, D}, xv)
	inv := bf16Tensor(t, []int{M, 1}, invv)

	// Broadcast Mul: out[i,j] = x[i,j] * inv[i].
	got, err := eng.Mul(ctx, x, inv)
	if err != nil {
		t.Fatalf("broadcast Mul: %v", err)
	}
	if want := []int{M, D}; !shapeEq(got.Shape(), want) {
		t.Fatalf("broadcast Mul shape = %v, want %v", got.Shape(), want)
	}
	gd := bf16ToF32(got.Data())
	for i := 0; i < M; i++ {
		for j := 0; j < D; j++ {
			want := float16.BFloat16FromFloat32(xv[i*D+j] * invv[i]).ToFloat32()
			assertBF16Close(t, "broadcastMul", i*D+j, gd[i*D+j], want, 2.0)
		}
	}

	// AddScalar: out[i] = x[i] + eps.
	eps := float16.BFloat16FromFloat32(0.125)
	gotS, err := eng.AddScalar(ctx, x, eps)
	if err != nil {
		t.Fatalf("AddScalar: %v", err)
	}
	sd := bf16ToF32(gotS.Data())
	for i := range xv {
		want := float16.BFloat16FromFloat32(xv[i] + 0.125).ToFloat32()
		assertBF16Close(t, "addScalar", i, sd[i], want, 2.0)
	}
}

// ramp returns n small, bf16-exact values centered near zero so GEMM sums stay
// well-conditioned (no catastrophic cancellation).
func ramp(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		v := float32((i%9)-4) * 0.125 // multiples of 1/8 are exact in bf16
		out[i] = v
	}
	return out
}

func shapeEq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

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

package compute

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestGPUEngine_TinyBatchedGemm_AttentionShapes verifies the custom tiny-matrix
// batched-GEMM kernel (ADR 075 L3), which the GPU MatMul dispatches to for small
// m,n,k, matches the CPU reference GEMM within f32 tolerance for the exact
// CrossAsset attention shapes: Q@K^T [12,64]@[64,12]->[12,12] and
// weights@V [12,12]@[12,64]->[12,64], batched over B*heads = 1024.
func TestGPUEngine_TinyBatchedGemm_AttentionShapes(t *testing.T) {
	gpuEng := newTestGPUEngine(t)
	cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	cases := []struct {
		name       string
		batch      int
		m, k, n    int
		bBroadcast bool // B has batch dim 1 (strideB=0 broadcast)
	}{
		{"QKt_12x64x12_b1024", 1024, 12, 64, 12, false},
		{"weightsV_12x12x64_b1024", 1024, 12, 12, 64, false},
		{"QKt_broadcastB", 256, 12, 64, 12, true},
		{"tile_boundary_64x64x64", 64, 64, 64, 64, false},
		{"asym_3x7x5", 100, 3, 7, 5, false},
	}

	rng := rand.New(rand.NewSource(42))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			aData := make([]float32, tc.batch*tc.m*tc.k)
			for i := range aData {
				aData[i] = float32(rng.NormFloat64())
			}
			bBatch := tc.batch
			if tc.bBroadcast {
				bBatch = 1
			}
			bData := make([]float32, bBatch*tc.k*tc.n)
			for i := range bData {
				bData[i] = float32(rng.NormFloat64())
			}

			a, _ := tensor.New[float32]([]int{tc.batch, tc.m, tc.k}, aData)
			b, _ := tensor.New[float32]([]int{bBatch, tc.k, tc.n}, bData)

			gpuRes, err := gpuEng.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("GPU MatMul: %v", err)
			}
			// CPU reference uses the same broadcasting semantics.
			aCPU, _ := tensor.New[float32]([]int{tc.batch, tc.m, tc.k}, aData)
			bCPU, _ := tensor.New[float32]([]int{bBatch, tc.k, tc.n}, bData)
			cpuRes, err := cpuEng.MatMul(ctx, aCPU, bCPU)
			if err != nil {
				t.Fatalf("CPU MatMul: %v", err)
			}

			gd := gpuRes.Data()
			cd := cpuRes.Data()
			if len(gd) != len(cd) {
				t.Fatalf("length mismatch: GPU=%d CPU=%d", len(gd), len(cd))
			}
			var maxAbs, maxRel float64
			for i := range gd {
				if math.IsNaN(float64(gd[i])) {
					t.Fatalf("NaN in GPU result at [%d]", i)
				}
				diff := math.Abs(float64(gd[i] - cd[i]))
				if diff > maxAbs {
					maxAbs = diff
				}
				denom := math.Abs(float64(cd[i]))
				if denom > 1e-6 {
					if rel := diff / denom; rel > maxRel {
						maxRel = rel
					}
				}
			}
			// f32 GEMM with k up to 64: accumulation error ~ k*eps. Tolerate a
			// small absolute and relative bound (the kernel and CPU sum in the
			// same f32 precision; differences come only from summation order).
			if maxAbs > 1e-3 || maxRel > 1e-4 {
				t.Errorf("%s: tiny-GEMM vs CPU mismatch maxAbs=%e maxRel=%e", tc.name, maxAbs, maxRel)
			}
		})
	}
}

// TestGPUEngine_TinyBatchedGemm_MatchesCublas verifies the tiny kernel result
// equals the cuBLAS SgemmStridedBatched result for the same inputs (the path
// it replaces), by toggling ZERFOO_DISABLE_TINY_GEMM. This guards against any
// silent divergence between the custom path and the framework's prior behavior.
func TestGPUEngine_TinyBatchedGemm_MatchesCublas(t *testing.T) {
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	batch, m, k, n := 1024, 12, 64, 12
	rng := rand.New(rand.NewSource(7))
	aData := make([]float32, batch*m*k)
	for i := range aData {
		aData[i] = float32(rng.NormFloat64())
	}
	bData := make([]float32, batch*k*n)
	for i := range bData {
		bData[i] = float32(rng.NormFloat64())
	}

	run := func(disable bool) []float32 {
		t.Helper()
		old := disableTinyGemm
		disableTinyGemm = disable
		defer func() { disableTinyGemm = old }()
		a, _ := tensor.New[float32]([]int{batch, m, k}, aData)
		b, _ := tensor.New[float32]([]int{batch, k, n}, bData)
		res, err := eng.MatMul(ctx, a, b)
		if err != nil {
			t.Fatalf("MatMul (disable=%v): %v", disable, err)
		}
		out := make([]float32, len(res.Data()))
		copy(out, res.Data())
		return out
	}

	tiny := run(false)
	cublas := run(true)
	if len(tiny) != len(cublas) {
		t.Fatalf("length mismatch tiny=%d cublas=%d", len(tiny), len(cublas))
	}
	var maxAbs float64
	for i := range tiny {
		d := math.Abs(float64(tiny[i] - cublas[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	// Both accumulate in f32; only summation order differs, so the gap is tiny.
	if maxAbs > 1e-3 {
		t.Errorf("tiny-GEMM vs cuBLAS divergence: maxAbs=%e", maxAbs)
	}
}

// TestGPUEngine_TinyBatchedGemm_Gradcheck does a finite-difference gradient
// check through the GPU batched MatMul (which uses the tiny kernel) to confirm
// the forward result is the true matmul (so any autograd built on MatMul has
// correct gradients). dC/dA = upstream @ B^T, dC/dB = A^T @ upstream; we verify
// the forward against a numerical perturbation of a few entries.
func TestGPUEngine_TinyBatchedGemm_Gradcheck(t *testing.T) {
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	batch, m, k, n := 8, 12, 64, 12
	rng := rand.New(rand.NewSource(99))
	aData := make([]float32, batch*m*k)
	for i := range aData {
		aData[i] = float32(rng.NormFloat64()) * 0.1
	}
	bData := make([]float32, batch*k*n)
	for i := range bData {
		bData[i] = float32(rng.NormFloat64()) * 0.1
	}

	matmul := func(ad, bd []float32) []float32 {
		a, _ := tensor.New[float32]([]int{batch, m, k}, ad)
		b, _ := tensor.New[float32]([]int{batch, k, n}, bd)
		res, err := eng.MatMul(ctx, a, b)
		if err != nil {
			t.Fatalf("MatMul: %v", err)
		}
		out := make([]float32, len(res.Data()))
		copy(out, res.Data())
		return out
	}

	base := matmul(aData, bData)

	// Analytic dC[outIdx]/dA[aIdx]: pick batch 0, perturb A[0,i0,l0] and confirm
	// the change in C[0,i0,j] equals B[0,l0,j] * eps for all j (linear in A).
	const eps = float32(1e-2)
	i0, l0 := 2, 5
	aIdx := (0*m+i0)*k + l0
	perturbed := make([]float32, len(aData))
	copy(perturbed, aData)
	perturbed[aIdx] += eps
	after := matmul(perturbed, bData)

	for j := 0; j < n; j++ {
		outIdx := (0*m+i0)*n + j
		got := (after[outIdx] - base[outIdx]) / eps
		want := bData[(0*k+l0)*n+j] // B[0, l0, j]
		if math.Abs(float64(got-want)) > 1e-2 {
			t.Errorf("grad dC[0,%d,%d]/dA[0,%d,%d]: got=%f want=%f", i0, j, i0, l0, got, want)
		}
	}
}

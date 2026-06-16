package compute

// gpu_bf16_fused_norm_parity_test.go is the universal quality gate for the bf16
// variants of the three forward-only fused normalization kernels (ADR 075 lever
// L4 follow-on): GPUFusedAddRMSNorm, GPUFusedNormAdd, and GPUFusedQKNormRoPE on
// T = float16.BFloat16. Each bf16 GPU result is checked against an f64 reference
// rounded to bf16. The kernels accumulate reductions and the normalization in
// FP32, so the only expected divergence is bf16's 7-bit mantissa (a few bf16
// steps of slack).
//
// These tests are CUDA-gated; they run on a real GPU (the GB10 verify pod) and
// skip on hosts without CUDA. GPU verification is PENDING -- the bf16 fused norm
// kernels have not yet been executed on a GB10; these tests are the gate that
// will confirm parity when they next run on-device.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/internal/cuda"
)

// rmsNormF64 computes the f64 reference RMSNorm of one row:
// out[d] = x[d] / sqrt(mean(x^2) + eps) * w[d].
func rmsNormF64(x, w []float64, eps float64) []float64 {
	var sq float64
	for _, v := range x {
		sq += v * v
	}
	scale := 1.0 / math.Sqrt(sq/float64(len(x))+eps)
	out := make([]float64, len(x))
	for d := range x {
		out[d] = x[d] * scale * w[d]
	}
	return out
}

func f32SliceToF64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func TestGPUBF16_FusedAddRMSNormParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	rng := rand.New(rand.NewSource(11))

	const rows, D = 8, 64
	const eps = float32(1e-6)

	inV := make([]float32, rows*D)
	resV := make([]float32, rows*D)
	wV := make([]float32, D)
	for i := range inV {
		inV[i] = float16.BFloat16FromFloat32(rng.Float32()*2 - 1).ToFloat32()
		resV[i] = float16.BFloat16FromFloat32(rng.Float32()*2 - 1).ToFloat32()
	}
	for d := range wV {
		wV[d] = float16.BFloat16FromFloat32(rng.Float32()*0.5 + 0.75).ToFloat32()
	}

	input := bf16Tensor(t, []int{rows, D}, inV)
	residual := bf16Tensor(t, []int{rows, D}, resV)
	weight := bf16Tensor(t, []int{D}, wV)

	normed, sumOut, _, err := eng.GPUFusedAddRMSNorm(input, residual, weight, eps)
	if err != nil {
		t.Fatalf("GPUFusedAddRMSNorm(bf16): %v", err)
	}
	gotNormed := bf16ToF32(normed.Data())
	gotSum := bf16ToF32(sumOut.Data())

	wF64 := f32SliceToF64(wV)
	for r := 0; r < rows; r++ {
		// sum = input + residual (each operand is already exact bf16; the sum
		// rounds to bf16 once).
		sumRow := make([]float64, D)
		for d := 0; d < D; d++ {
			idx := r*D + d
			wantSum := float16.BFloat16FromFloat32(inV[idx] + resV[idx]).ToFloat32()
			assertBF16Close(t, "AddRMSNorm.sum", idx, gotSum[idx], wantSum, 1.0)
			sumRow[d] = float64(wantSum)
		}
		ref := rmsNormF64(sumRow, wF64, float64(eps))
		for d := 0; d < D; d++ {
			idx := r*D + d
			want := float16.BFloat16FromFloat32(float32(ref[d])).ToFloat32()
			assertBF16Close(t, "AddRMSNorm.normed", idx, gotNormed[idx], want, 4.0)
		}
	}
}

func TestGPUBF16_FusedNormAddParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	rng := rand.New(rand.NewSource(13))

	const rows, D = 8, 64
	const eps = float32(1e-6)

	inV := make([]float32, rows*D)
	resV := make([]float32, rows*D)
	wV := make([]float32, D)
	for i := range inV {
		inV[i] = float16.BFloat16FromFloat32(rng.Float32()*2 - 1).ToFloat32()
		resV[i] = float16.BFloat16FromFloat32(rng.Float32()*2 - 1).ToFloat32()
	}
	for d := range wV {
		wV[d] = float16.BFloat16FromFloat32(rng.Float32()*0.5 + 0.75).ToFloat32()
	}

	input := bf16Tensor(t, []int{rows, D}, inV)
	weight := bf16Tensor(t, []int{D}, wV)
	residual := bf16Tensor(t, []int{rows, D}, resV)

	got, err := eng.GPUFusedNormAdd(input, weight, residual, eps)
	if err != nil {
		t.Fatalf("GPUFusedNormAdd(bf16): %v", err)
	}
	gotData := bf16ToF32(got.Data())

	wF64 := f32SliceToF64(wV)
	for r := 0; r < rows; r++ {
		xRow := f32SliceToF64(inV[r*D : (r+1)*D])
		ref := rmsNormF64(xRow, wF64, float64(eps))
		for d := 0; d < D; d++ {
			idx := r*D + d
			want := float16.BFloat16FromFloat32(float32(ref[d]) + resV[idx]).ToFloat32()
			assertBF16Close(t, "NormAdd", idx, gotData[idx], want, 4.0)
		}
	}
}

func TestGPUBF16_FusedQKNormRoPEParity(t *testing.T) {
	eng := newTestGPUBF16Engine(t)
	rng := rand.New(rand.NewSource(17))

	const numQHeads, numKVHeads, headDim = 2, 2, 16
	const totalHeads = numQHeads + numKVHeads
	const halfRotary = headDim / 2
	const eps = float32(1e-6)

	inV := make([]float32, totalHeads*headDim)
	for i := range inV {
		inV[i] = float16.BFloat16FromFloat32(rng.Float32()*2 - 1).ToFloat32()
	}
	wqV := make([]float32, headDim)
	wkV := make([]float32, headDim)
	for d := range wqV {
		wqV[d] = float16.BFloat16FromFloat32(rng.Float32()*0.5 + 0.75).ToFloat32()
		wkV[d] = float16.BFloat16FromFloat32(rng.Float32()*0.5 + 0.75).ToFloat32()
	}
	// RoPE angle tables (bf16, matching the engine's generic tensor type).
	cosV := make([]float32, halfRotary)
	sinV := make([]float32, halfRotary)
	for d := 0; d < halfRotary; d++ {
		theta := math.Pow(10000, -2*float64(d)/float64(headDim))
		cosV[d] = float16.BFloat16FromFloat32(float32(math.Cos(theta))).ToFloat32()
		sinV[d] = float16.BFloat16FromFloat32(float32(math.Sin(theta))).ToFloat32()
	}

	input := bf16Tensor(t, []int{totalHeads, headDim}, inV)
	weightQ := bf16Tensor(t, []int{headDim}, wqV)
	weightK := bf16Tensor(t, []int{headDim}, wkV)
	cosAngles := bf16Tensor(t, []int{halfRotary}, cosV)
	sinAngles := bf16Tensor(t, []int{halfRotary}, sinV)

	got, err := eng.GPUFusedQKNormRoPE(input, weightQ, weightK, cosAngles, sinAngles, eps, totalHeads, headDim, numQHeads, halfRotary)
	if err != nil {
		t.Fatalf("GPUFusedQKNormRoPE(bf16): %v", err)
	}
	gotData := bf16ToF32(got.Data())

	wqF64 := f32SliceToF64(wqV)
	wkF64 := f32SliceToF64(wkV)
	for h := 0; h < totalHeads; h++ {
		w := wqF64
		if h >= numQHeads {
			w = wkF64
		}
		xRow := f32SliceToF64(inV[h*headDim : (h+1)*headDim])
		normed := rmsNormF64(xRow, w, float64(eps))
		// Apply RoPE in f64 against the same bf16 angle tables.
		ref := make([]float64, headDim)
		for d := 0; d < headDim; d++ {
			if d < halfRotary {
				c := float64(cosV[d])
				s := float64(sinV[d])
				x0 := normed[d]
				x1 := normed[d+halfRotary]
				ref[d] = x0*c - x1*s
				ref[d+halfRotary] = x0*s + x1*c
			} else if d >= 2*halfRotary {
				ref[d] = normed[d]
			}
		}
		for d := 0; d < headDim; d++ {
			idx := h*headDim + d
			want := float16.BFloat16FromFloat32(float32(ref[d])).ToFloat32()
			assertBF16Close(t, "QKNormRoPE", idx, gotData[idx], want, 4.0)
		}
	}
}

// ensure cuda import is used even if all tests skip (mirrors the sibling file).
var _ = cuda.Available

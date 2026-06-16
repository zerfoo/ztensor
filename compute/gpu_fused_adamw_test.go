package compute

import (
	"math"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestGPUFusedAdamW_RejectsCPUBacked asserts that GPUFusedAdamW refuses
// CPU-backed (non-GPUStorage) param/grad tensors so the optimizer cleanly
// falls back to the host stepMixedV path instead of silently no-op'ing.
//
// Constructing a real GPUEngine requires a device, so this validates the guard
// on a zero-value engine: a CPU-backed param trips the storage check before any
// device call. The numerical equivalence of the kernel to the host f64 update
// is covered by zerfoo's adamw_mixedv_equivalence_test, which drives the actual
// on-device path on GB10.
func TestGPUFusedAdamW_RejectsCPUBacked(t *testing.T) {
	var e GPUEngine[float32]

	paramT, err := tensor.New[float32]([]int{4}, []float32{0.5, -0.25, 1.0, -1.0})
	if err != nil {
		t.Fatalf("new param: %v", err)
	}
	gradT, err := tensor.New[float32]([]int{4}, []float32{0.1, 0.2, 0.3, 0.4})
	if err != nil {
		t.Fatalf("new grad: %v", err)
	}

	err = e.GPUFusedAdamW(paramT, gradT, 0.9, 0.999, 1e-8, 1e-3, 0.0, 1)
	if err == nil {
		t.Fatalf("expected error for CPU-backed param, got nil")
	}
	if !strings.Contains(err.Error(), "not GPU-resident") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// adamwHostStep reproduces the fused_adamw.cu arithmetic in pure Go (the exact
// f64 math the kernel performs, with the same f32 rounding points) so the
// kernel's documented algorithm is checked against the AdamW reference here in
// ztensor, independent of zerfoo. It mirrors training/optimizer stepMixedV.
func adamwHostStep(param, m []float32, v64 []float64, grad []float32,
	beta1, beta2, eps, lr, wd float64, t int) {
	numer := math.Sqrt(1.0 - math.Pow(beta2, float64(t)))
	denom := 1.0 - math.Pow(beta1, float64(t))
	alpha := lr * (numer / denom)
	lrWd := lr * wd
	for i := range param {
		g := float64(grad[i])
		mOld := float64(m[i])
		mNew := beta1*mOld + (1.0-beta1)*g
		m[i] = float32(mNew)
		v64[i] = beta2*v64[i] + (1.0-beta2)*g*g
		denomI := math.Sqrt(v64[i]) + eps
		update := alpha * mNew / denomI
		pv := float64(param[i])
		pv = pv - update - lrWd*pv
		param[i] = float32(pv)
	}
}

// TestAdamWKernelArithmetic_MatchesReference asserts the kernel's documented
// f64-second-moment arithmetic tracks a straightforward all-f64 AdamW within a
// tight tolerance over many steps, including the near-zero gradient regime that
// motivates the f64 v. This is the ztensor-side numerics gate for the kernel
// algorithm; the on-device kernel itself is exercised on GB10.
func TestAdamWKernelArithmetic_MatchesReference(t *testing.T) {
	_ = numeric.Float32Ops{} // keep numeric import in case helpers move here.

	n := 6
	param := []float32{0.5, -0.25, 1.0, -1.0, 0.5, 0.5}
	m := make([]float32, n)
	v64 := make([]float64, n)

	// Full-f64 reference.
	pf := make([]float64, n)
	mf := make([]float64, n)
	vf := make([]float64, n)
	for i := range param {
		pf[i] = float64(param[i])
	}

	beta1, beta2, eps, lr, wd := 0.9, 0.999, 1e-8, 1e-3, 0.0
	for step := 1; step <= 60; step++ {
		grad := make([]float32, n)
		for i := range grad {
			if i%2 == 0 {
				grad[i] = float32(0.1 * math.Cos(float64(step)))
			} else {
				grad[i] = 1e-10 // near-zero: f32 v would underflow.
			}
		}
		adamwHostStep(param, m, v64, grad, beta1, beta2, eps, lr, wd, step)

		// Full-f64 reference update.
		numer := math.Sqrt(1.0 - math.Pow(beta2, float64(step)))
		denom := 1.0 - math.Pow(beta1, float64(step))
		alpha := lr * (numer / denom)
		for i := range pf {
			g := float64(grad[i])
			mf[i] = beta1*mf[i] + (1.0-beta1)*g
			vf[i] = beta2*vf[i] + (1.0-beta2)*g*g
			update := alpha * mf[i] / (math.Sqrt(vf[i]) + eps)
			pf[i] = pf[i] - update
		}
	}

	for i := range param {
		if diff := math.Abs(float64(param[i]) - pf[i]); diff > 1e-5 {
			t.Fatalf("param[%d]: kernel-arith=%v fullF64=%v diff=%g", i, param[i], pf[i], diff)
		}
	}
}

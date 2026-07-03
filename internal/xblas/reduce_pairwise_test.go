package xblas

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
)

// TestPairwiseSumF32_TightensAndStable verifies the fixed-order pairwise float32
// reduction is bitwise-stable and no worse than a naive fold against a float64
// ground truth for ill-conditioned inputs.
func TestPairwiseSumF32_TightensAndStable(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 42))
	for _, n := range []int{2048, 8192, 32768} {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(0.1 + 0.001*rng.NormFloat64())
		}
		get := func(i int) float32 { return s[i] }
		first := pairwiseSumF32(n, get)
		for r := 0; r < 4; r++ {
			if pairwiseSumF32(n, get) != first {
				t.Fatalf("n=%d: not bitwise-stable", n)
			}
		}

		var truth, c float64
		for _, v := range s {
			y := float64(v) - c
			tt := truth + y
			c = (tt - truth) - y
			truth = tt
		}
		var naive float32
		for _, v := range s {
			naive += v
		}
		en := math.Abs((float64(naive) - truth) / truth)
		ep := math.Abs((float64(first) - truth) / truth)
		if ep > en {
			t.Fatalf("n=%d: pairwise relerr %.3e worse than naive %.3e", n, ep, en)
		}
	}
}

// TestRMSNormF32_SumOfSquaresAccurate checks the RMSNorm scale factor agrees
// with a float64 reference within a tight tolerance. On arm64 this exercises the
// NEON asm (already fixed-order fp32); on other arches it exercises the pairwise
// generic fallback. Either way the result must be order-stable and accurate.
func TestRMSNormF32_SumOfSquaresAccurate(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 3))
	const eps = float32(1e-6)
	for _, d := range []int{2048, 4096} {
		x := make([]float32, d)
		w := make([]float32, d)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
			w[i] = 1
		}
		out := make([]float32, d)
		var scale float32
		RMSNormF32(&out[0], &x[0], &w[0], d, eps, &scale)

		var sumSq float64
		for _, v := range x {
			sumSq += float64(v) * float64(v)
		}
		wantScale := 1.0 / math.Sqrt(sumSq/float64(d)+float64(eps))
		if rel := math.Abs((float64(scale) - wantScale) / wantScale); rel > 1e-4 {
			t.Fatalf("d=%d arch=%s: scale relerr %.3e exceeds 1e-4", d, runtime.GOARCH, rel)
		}
	}
}

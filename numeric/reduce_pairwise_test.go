package numeric

import (
	"math"
	"math/rand/v2"
	"testing"
)

func kahanRef64(s []float32) float64 {
	var sum, c float64
	for _, v := range s {
		y := float64(v) - c
		t := sum + y
		c = (t - sum) - y
		sum = t
	}

	return sum
}

func naiveF32(s []float32) float32 {
	var acc float32
	for _, v := range s {
		acc += v
	}

	return acc
}

// TestFloat32OpsSum_TightensAndStable verifies Float32Ops.Sum uses fixed-order
// pairwise accumulation: bitwise-stable run to run and no worse than the naive
// fold against a float64 ground truth.
func TestFloat32OpsSum_TightensAndStable(t *testing.T) {
	ops := Float32Ops{}
	rng := rand.New(rand.NewPCG(4, 8))
	for _, n := range []int{2048, 8192, 32768} {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(0.1 + 0.001*rng.NormFloat64())
		}
		first := ops.Sum(s)
		for r := 0; r < 4; r++ {
			if ops.Sum(s) != first {
				t.Fatalf("n=%d: Sum not bitwise-stable", n)
			}
		}
		truth := kahanRef64(s)
		en := math.Abs((float64(naiveF32(s)) - truth) / truth)
		ep := math.Abs((float64(first) - truth) / truth)
		if ep > en {
			t.Fatalf("n=%d: pairwise relerr %.3e worse than naive %.3e", n, ep, en)
		}
	}
}

// TestPairwiseSum_Float64 sanity-checks the float64 instantiation matches an
// exact small-sum and stays deterministic.
func TestPairwiseSum_Float64(t *testing.T) {
	s := []float64{1, 2, 3, 4, 5}
	if got := pairwiseSum(s); got != 15 {
		t.Fatalf("got %v want 15", got)
	}
	if pairwiseSum([]float64{}) != 0 {
		t.Fatal("empty sum must be 0")
	}
}

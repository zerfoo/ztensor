package compute

import (
	"context"
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// kahanRef64 is a compensated float64 ground-truth sum for accuracy comparisons.
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

func naiveSumF32(s []float32) float32 {
	var acc float32
	for _, v := range s {
		acc += v
	}

	return acc
}

func relErr(got float32, truth float64) float64 {
	if truth == 0 {
		return math.Abs(float64(got))
	}

	return math.Abs((float64(got) - truth) / truth)
}

// TestPairwiseReduce_EqualsNaiveForSmallN verifies the base case is an ordinary
// fold, so behavior is unchanged for short reductions.
func TestPairwiseReduce_EqualsNaiveForSmallN(t *testing.T) {
	for _, n := range []int{1, 2, 7, 63, pairwiseBlock} {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(i) * 0.5
		}
		got := pairwiseReduce(len(s), float32(0), func(i int) float32 { return s[i] }, func(a, b float32) float32 { return a + b })
		if want := naiveSumF32(s); got != want {
			t.Fatalf("n=%d: pairwise=%v naive=%v (must match for n<=block)", n, got, want)
		}
	}
}

// TestPairwiseReduce_Deterministic verifies the accumulation order is a pure
// function of length: repeated reductions are bitwise-identical.
func TestPairwiseReduce_Deterministic(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	s := make([]float32, 40000)
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	add := func(a, b float32) float32 { return a + b }
	first := pairwiseReduce(len(s), float32(0), func(i int) float32 { return s[i] }, add)
	for r := 0; r < 8; r++ {
		if got := pairwiseReduce(len(s), float32(0), func(i int) float32 { return s[i] }, add); got != first {
			t.Fatalf("run %d not bitwise-stable: %v != %v", r, got, first)
		}
	}
}

// TestPairwiseReduce_TightensVsNaive is the core numeric-correctness guard: the
// fixed-order pairwise sum must be at least as close to the float64 ground truth
// as the naive left-to-right fold for ill-conditioned float32 inputs. A change
// that loosened any reduction would regress here.
func TestPairwiseReduce_TightensVsNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for _, n := range []int{1024, 4096, 16384, 65536} {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(0.1 + 0.001*rng.NormFloat64())
		}
		truth := kahanRef64(s)
		add := func(a, b float32) float32 { return a + b }
		pw := pairwiseReduce(len(s), float32(0), func(i int) float32 { return s[i] }, add)
		en, ep := relErr(naiveSumF32(s), truth), relErr(pw, truth)
		if ep > en {
			t.Fatalf("n=%d: pairwise relerr %.3e worse than naive %.3e", n, ep, en)
		}
	}
}

// TestCPUEngineSum_WorkerInvariant verifies ReduceSum is bitwise-identical
// regardless of GOMAXPROCS: the per-stripe accumulation order must not depend on
// runtime scheduling. This is the determinism guarantee downstream modes rely on.
func TestCPUEngineSum_WorkerInvariant(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	rng := rand.New(rand.NewPCG(3, 5))
	rows, cols := 17, 4096
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}
	a, err := tensor.New[float32]([]int{rows, cols}, data)
	if err != nil {
		t.Fatal(err)
	}

	sumWith := func(procs int) []float32 {
		old := runtime.GOMAXPROCS(procs)
		defer runtime.GOMAXPROCS(old)
		out, err := engine.Sum(context.Background(), a, 1, false, nil)
		if err != nil {
			t.Fatal(err)
		}
		cp := make([]float32, len(out.Data()))
		copy(cp, out.Data())

		return cp
	}

	ref := sumWith(1)
	for _, p := range []int{2, 4, 8} {
		got := sumWith(p)
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("GOMAXPROCS=%d row %d: %v != %v (worker-dependent order)", p, i, got[i], ref[i])
			}
		}
	}
}

// TestCPUEngineSoftmax_Deterministic verifies the softmax denominator reduction
// is run-to-run bitwise-stable.
func TestCPUEngineSoftmax_Deterministic(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	rng := rand.New(rand.NewPCG(9, 13))
	// inner>1 shape forces the generic (non-xblas) softmax path under test.
	data := make([]float32, 3*2048*2)
	for i := range data {
		data[i] = float32(10 * rng.NormFloat64())
	}
	a, err := tensor.New[float32]([]int{3, 2048, 2}, data)
	if err != nil {
		t.Fatal(err)
	}
	first, err := engine.Softmax(context.Background(), a, 1)
	if err != nil {
		t.Fatal(err)
	}
	ref := make([]float32, len(first.Data()))
	copy(ref, first.Data())
	for r := 0; r < 4; r++ {
		out, err := engine.Softmax(context.Background(), a, 1)
		if err != nil {
			t.Fatal(err)
		}
		for i := range ref {
			if out.Data()[i] != ref[i] {
				t.Fatalf("softmax not deterministic at %d on run %d", i, r)
			}
		}
	}
}

package parity

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestTrainingLoop_WolfPattern_GPU is the GB10 training-loop proof for
// poison-mode hardening (zerfoo plan S2.3.1): a small two-layer training
// loop running the EXACT hazard schedule that bit Wolf (gr-12 /
// zerfoo#850): per-sample forward+backward with the gradients accumulated
// into PERSISTENT (non-arena) device buffers, engine ResetPool after every
// sample, and an in-place optimizer step once per batch -- all on the real
// CUDA arena with ZTENSOR_ARENA_POISON semantics enabled and a deliberately
// small arena. Any read of recycled arena memory (a stale activation, a
// gradient left arena-backed, an accumulator silently re-homed into the
// pool) surfaces as a deterministic NaN and fails the run.
//
// PASS requires (1) every parameter finite after training and (2) the GPU
// result matching a plain CPU-engine run of the byte-identical loop within
// f32 reduction tolerance -- catching silent corruption, not just NaN.
//
// Runs on the DGX via the Spark pod in scripts/parity/; skips without CUDA.
func TestTrainingLoop_WolfPattern_GPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	restorePoison := cuda.SetArenaPoisonEnabledForTesting(true)
	defer restorePoison()
	restoreArena := compute.SetArenaBytesForTesting(gpuParityArenaBytes)
	defer restoreArena()

	gpuEng, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	gpuW1, gpuW2 := runWolfTrainingLoop(t, trainSide{
		eng:   gpuEng,
		reset: gpuEng.ResetPool,
		persistent: func(shape []int, data []float32) *t32 {
			st, err := tensor.NewGPUStorageFromSlice(append([]float32(nil), data...))
			if err != nil {
				t.Fatalf("NewGPUStorageFromSlice: %v", err)
			}
			tt, err := tensor.NewWithStorage[float32](shape, st)
			if err != nil {
				t.Fatalf("NewWithStorage: %v", err)
			}
			return tt
		},
	})

	cpuW1, cpuW2 := runWolfTrainingLoop(t, cpuTrainSide(t))

	assertFiniteAndClose(t, "W1", gpuW1, cpuW1)
	assertFiniteAndClose(t, "W2", gpuW2, cpuW2)
}

// TestTrainingLoop_WolfPattern_StressCI runs the identical loop in ordinary
// CI with the host-backed-arena StressEngine as the candidate (GPU lifetime
// semantics without a GPU) under poison, against the plain CPU reference --
// catching lifetime regressions in the loop pattern before a GB10 run.
func TestTrainingLoop_WolfPattern_StressCI(t *testing.T) {
	enablePoison(t)
	stress := NewStressEngine(compute.NewCPUEngine[float32](numeric.Float32Ops{}), 1<<20)
	sW1, sW2 := runWolfTrainingLoop(t, trainSide{
		eng:        stress,
		reset:      stress.ResetArena,
		persistent: hostPersistent(t),
	})

	cpuW1, cpuW2 := runWolfTrainingLoop(t, cpuTrainSide(t))

	assertFiniteAndClose(t, "W1", sW1, cpuW1)
	assertFiniteAndClose(t, "W2", sW2, cpuW2)
}

// cpuTrainSide is the plain CPU reference side: no arena, no reset.
func cpuTrainSide(t *testing.T) trainSide {
	t.Helper()
	return trainSide{
		eng:        compute.NewCPUEngine[float32](numeric.Float32Ops{}),
		persistent: hostPersistent(t),
	}
}

// hostPersistent allocates GC-owned host tensors, which can never be
// recycled behind a live reference.
func hostPersistent(t *testing.T) func(shape []int, data []float32) *t32 {
	t.Helper()
	return func(shape []int, data []float32) *t32 {
		tt, err := tensor.New[float32](shape, append([]float32(nil), data...))
		if err != nil {
			t.Fatalf("tensor.New: %v", err)
		}
		return tt
	}
}

// trainSide abstracts the engine-specific pieces of the loop: how persistent
// (never-recycled) parameter/accumulator tensors are allocated and how the
// per-sample arena reset is performed (nil for plain CPU).
type trainSide struct {
	eng        compute.Engine[float32]
	reset      func()
	persistent func(shape []int, data []float32) *t32
}

// runWolfTrainingLoop trains a tiny x->MatMul->Tanh->MatMul->Softmax net for
// several batches with per-sample ResetPool and per-batch SGD, returning the
// final parameter values (host copies).
func runWolfTrainingLoop(t *testing.T, side trainSide) ([]float32, []float32) {
	t.Helper()
	ctx := context.Background()
	eng := side.eng

	const (
		inDim   = 8
		hidDim  = 8
		outDim  = 4
		batches = 3
		samples = 4
		lr      = float32(0.05)
	)

	// Deterministic, engine-independent initialization and data.
	w1Init := rampData(inDim*hidDim, 0.31)
	w2Init := rampData(hidDim*outDim, 0.17)

	// Persistent parameters and gradient accumulators: non-arena storage
	// (raw GPUStorage on the GPU side), so ResetPool never recycles them.
	w1 := side.persistent([]int{inDim, hidDim}, w1Init)
	w2 := side.persistent([]int{hidDim, outDim}, w2Init)
	g1 := side.persistent([]int{inDim, hidDim}, make([]float32, inDim*hidDim))
	g2 := side.persistent([]int{hidDim, outDim}, make([]float32, hidDim*outDim))

	for b := 0; b < batches; b++ {
		for s := 0; s < samples; s++ {
			x := mustNew(t, []int{1, inDim}, rampData(inDim, 0.41+float32(b*samples+s)*0.07))
			target := oneHot(t, outDim, (b+s)%outDim)

			// Forward (every intermediate is arena-backed on the GPU side).
			h := mustOp(t, "MatMul(x,W1)")(eng.MatMul(ctx, x, w1))
			a := mustOp(t, "Tanh")(eng.Tanh(ctx, h))
			y := mustOp(t, "MatMul(a,W2)")(eng.MatMul(ctx, a, w2))
			p := mustOp(t, "Softmax")(eng.Softmax(ctx, y, 1))

			// Backward (softmax+cross-entropy: dy = p - target).
			dy := mustOp(t, "Sub(p,target)")(eng.Sub(ctx, p, target))
			aT := mustOp(t, "Transpose(a)")(eng.Transpose(ctx, a, []int{1, 0}))
			dW2 := mustOp(t, "MatMul(aT,dy)")(eng.MatMul(ctx, aT, dy))
			w2T := mustOp(t, "Transpose(W2)")(eng.Transpose(ctx, w2, []int{1, 0}))
			da := mustOp(t, "MatMul(dy,W2T)")(eng.MatMul(ctx, dy, w2T))
			dh := mustOp(t, "TanhPrime")(eng.TanhPrime(ctx, h, da))
			xT := mustOp(t, "Transpose(x)")(eng.Transpose(ctx, x, []int{1, 0}))
			dW1 := mustOp(t, "MatMul(xT,dh)")(eng.MatMul(ctx, xT, dh))

			// Accumulate into the persistent buffers IN PLACE (the
			// zerfoo#855 contract: dst's storage must be preserved).
			accumulateInPlace(t, ctx, eng, g1, dW1, "G1")
			accumulateInPlace(t, ctx, eng, g2, dW2, "G2")

			// Wolf's per-sample ResetPool: recycle (and poison) every
			// arena intermediate of this sample.
			if side.reset != nil {
				side.reset()
			}
		}

		// Optimizer step, once per batch: W -= lr * G, in place.
		sgdStepInPlace(t, ctx, eng, w1, g1, lr, "W1")
		sgdStepInPlace(t, ctx, eng, w2, g2, lr, "W2")
		if err := eng.Zero(ctx, g1); err != nil {
			t.Fatalf("Zero(G1): %v", err)
		}
		if err := eng.Zero(ctx, g2); err != nil {
			t.Fatalf("Zero(G2): %v", err)
		}
		// Recycle the step's arena temporaries before the next batch.
		if side.reset != nil {
			side.reset()
		}
	}

	return append([]float32(nil), w1.Data()...), append([]float32(nil), w2.Data()...)
}

// accumulateInPlace performs accum += grad via the engine and asserts the
// engine honored the dst contract (no re-homing of the persistent buffer
// into the arena -- the exact failure mode of zerfoo#850/#855).
func accumulateInPlace(t *testing.T, ctx context.Context, eng compute.Engine[float32], accum, grad *t32, name string) {
	t.Helper()
	before := accum.GetStorage()
	res, err := eng.Add(ctx, accum, grad, accum)
	if err != nil {
		t.Fatalf("Add into %s: %v", name, err)
	}
	if res != accum || res.GetStorage() != before {
		t.Fatalf("engine.Add relocated persistent accumulator %s (storage %p -> %p); dst must be written in place", name, before, res.GetStorage())
	}
}

// sgdStepInPlace performs w -= lr*g with the scaled gradient as an arena
// temporary and the subtraction writing in place into the persistent w.
func sgdStepInPlace(t *testing.T, ctx context.Context, eng compute.Engine[float32], w, g *t32, lr float32, name string) {
	t.Helper()
	step, err := eng.MulScalar(ctx, g, lr)
	if err != nil {
		t.Fatalf("MulScalar(%s): %v", name, err)
	}
	before := w.GetStorage()
	res, err := eng.Sub(ctx, w, step, w)
	if err != nil {
		t.Fatalf("Sub into %s: %v", name, err)
	}
	if res != w || res.GetStorage() != before {
		t.Fatalf("engine.Sub relocated persistent parameter %s; dst must be written in place", name)
	}
}

// assertFiniteAndClose checks every GPU value is finite (poison NaN would
// land here) and within f32 reduction tolerance of the CPU reference.
func assertFiniteAndClose(t *testing.T, name string, gpu, cpu []float32) {
	t.Helper()
	if len(gpu) != len(cpu) {
		t.Fatalf("%s: length mismatch gpu=%d cpu=%d", name, len(gpu), len(cpu))
	}
	const (
		atol = 1e-5
		rtol = 1e-3
	)
	for i := range gpu {
		gv, cv := float64(gpu[i]), float64(cpu[i])
		if math.IsNaN(gv) || math.IsInf(gv, 0) {
			t.Fatalf("%s[%d] = %v: non-finite after training under poison (stale arena read)", name, i, gv)
		}
		if diff := math.Abs(gv - cv); diff > atol+rtol*math.Abs(cv) {
			t.Errorf("%s[%d]: gpu=%v cpu=%v |diff|=%v exceeds atol %v + rtol %v", name, i, gv, cv, diff, atol, rtol)
		}
	}
	t.Logf("%s: %d values finite, GPU within atol %v / rtol %v of CPU", name, len(gpu), atol, rtol)
}

// rampData generates deterministic, well-conditioned values in (-1, 1).
func rampData(n int, seed float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(seed) + 0.7*float64(i)))
	}
	return out
}

func oneHot(t *testing.T, n, hot int) *t32 {
	t.Helper()
	data := make([]float32, n)
	data[hot] = 1
	return mustNew(t, []int{1, n}, data)
}

func mustNew(t *testing.T, shape []int, data []float32) *t32 {
	t.Helper()
	tt, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	return tt
}

// mustOp adapts the (tensor, error) engine-op return for inline use.
func mustOp(t *testing.T, what string) func(*t32, error) *t32 {
	t.Helper()
	return func(res *t32, err error) *t32 {
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		return res
	}
}

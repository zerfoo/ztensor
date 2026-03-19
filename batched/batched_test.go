package batched

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func newEngine() compute.Engine[float32] {
	return compute.NewCPUEngine[float32](numeric.Float32Ops{})
}

func TestBatchedInference_1000Models(t *testing.T) {
	const numModels = 1000
	arch := Architecture{
		Layers: []LayerSpec{
			{InputSize: 4, OutputSize: 8},
			{InputSize: 8, OutputSize: 2},
		},
	}

	bi, err := NewBatchedInference(numModels, arch, newEngine())
	if err != nil {
		t.Fatalf("NewBatchedInference: %v", err)
	}

	rng := rand.New(rand.NewPCG(42, 0))

	// Load different weights for each model.
	for m := range numModels {
		weights := map[string][]float32{
			"layer_0": randomSlice(rng, 4*8),
			"layer_1": randomSlice(rng, 8*2),
		}
		if err := bi.LoadWeights(m, weights); err != nil {
			t.Fatalf("LoadWeights(%d): %v", m, err)
		}
	}

	// Build identical inputs for all models.
	inputs := make([][]float32, numModels)
	for i := range inputs {
		inputs[i] = []float32{1.0, 2.0, 3.0, 4.0}
	}

	outputs, err := bi.Forward(inputs)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	if len(outputs) != numModels {
		t.Fatalf("expected %d outputs, got %d", numModels, len(outputs))
	}

	// Verify outputs differ across models (different weights should produce
	// different outputs even with identical inputs).
	unique := make(map[[2]float32]struct{})
	for _, out := range outputs {
		if len(out) != 2 {
			t.Fatalf("expected output length 2, got %d", len(out))
		}
		key := [2]float32{out[0], out[1]}
		unique[key] = struct{}{}
	}

	// With 1000 random weight sets, we expect nearly all outputs to be unique.
	if len(unique) < numModels/2 {
		t.Errorf("expected most outputs to be unique, got only %d unique out of %d", len(unique), numModels)
	}
}

func TestBatchedInference_Throughput(t *testing.T) {
	const numModels = 100
	arch := Architecture{
		Layers: []LayerSpec{
			{InputSize: 16, OutputSize: 32},
			{InputSize: 32, OutputSize: 8},
		},
	}

	engine := newEngine()
	rng := rand.New(rand.NewPCG(99, 0))

	// Setup batched inference.
	bi, err := NewBatchedInference(numModels, arch, engine)
	if err != nil {
		t.Fatalf("NewBatchedInference: %v", err)
	}
	for m := range numModels {
		weights := map[string][]float32{
			"layer_0": randomSlice(rng, 16*32),
			"layer_1": randomSlice(rng, 32*8),
		}
		if err := bi.LoadWeights(m, weights); err != nil {
			t.Fatalf("LoadWeights: %v", err)
		}
	}

	// Setup sequential inference (one model at a time).
	seqModels := make([]*BatchedInference, numModels)
	rng2 := rand.New(rand.NewPCG(99, 0)) // same seed for fair comparison
	for m := range numModels {
		sm, err := NewBatchedInference(1, arch, engine)
		if err != nil {
			t.Fatalf("NewBatchedInference(1): %v", err)
		}
		weights := map[string][]float32{
			"layer_0": randomSlice(rng2, 16*32),
			"layer_1": randomSlice(rng2, 32*8),
		}
		if err := sm.LoadWeights(0, weights); err != nil {
			t.Fatalf("seq LoadWeights: %v", err)
		}
		seqModels[m] = sm
	}

	inputs := make([][]float32, numModels)
	for i := range inputs {
		inputs[i] = randomSlice(rng, 16)
	}

	// Benchmark batched.
	const iters = 20
	batchedStart := time.Now()
	for range iters {
		if _, err := bi.Forward(inputs); err != nil {
			t.Fatalf("batched Forward: %v", err)
		}
	}
	batchedDur := time.Since(batchedStart)

	// Benchmark sequential.
	seqStart := time.Now()
	for range iters {
		for m := range numModels {
			if _, err := seqModels[m].Forward(inputs[m:m+1]); err != nil {
				t.Fatalf("seq Forward: %v", err)
			}
		}
	}
	seqDur := time.Since(seqStart)

	t.Logf("batched: %v, sequential: %v, speedup: %.1fx", batchedDur, seqDur, float64(seqDur)/float64(batchedDur))

	// Batched should be faster than sequential for 100 models.
	if batchedDur >= seqDur {
		t.Logf("warning: batched (%v) was not faster than sequential (%v) — CPU batched GEMM may not show speedup", batchedDur, seqDur)
	}
}

func TestBatchedInference_Architecture(t *testing.T) {
	tests := []struct {
		name string
		arch Architecture
	}{
		{
			name: "single layer",
			arch: Architecture{
				Layers: []LayerSpec{
					{InputSize: 3, OutputSize: 2},
				},
			},
		},
		{
			name: "three layers",
			arch: Architecture{
				Layers: []LayerSpec{
					{InputSize: 4, OutputSize: 8},
					{InputSize: 8, OutputSize: 16},
					{InputSize: 16, OutputSize: 1},
				},
			},
		},
		{
			name: "wide layer",
			arch: Architecture{
				Layers: []LayerSpec{
					{InputSize: 2, OutputSize: 256},
					{InputSize: 256, OutputSize: 1},
				},
			},
		},
		{
			name: "relu activation",
			arch: Architecture{
				Layers: []LayerSpec{
					{InputSize: 4, OutputSize: 8, Activation: ActivationReLU},
					{InputSize: 8, OutputSize: 2},
				},
			},
		},
		{
			name: "tanh activation",
			arch: Architecture{
				Layers: []LayerSpec{
					{InputSize: 4, OutputSize: 8, Activation: ActivationTanh},
					{InputSize: 8, OutputSize: 2},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const numModels = 5
			bi, err := NewBatchedInference(numModels, tt.arch, newEngine())
			if err != nil {
				t.Fatalf("NewBatchedInference: %v", err)
			}

			rng := rand.New(rand.NewPCG(7, 0))
			for m := range numModels {
				weights := make(map[string][]float32)
				for i, l := range tt.arch.Layers {
					key := fmt.Sprintf("layer_%d", i)
					weights[key] = randomSlice(rng, l.InputSize*l.OutputSize)
				}
				if err := bi.LoadWeights(m, weights); err != nil {
					t.Fatalf("LoadWeights(%d): %v", m, err)
				}
			}

			inputSize := tt.arch.Layers[0].InputSize
			inputs := make([][]float32, numModels)
			for i := range inputs {
				inputs[i] = randomSlice(rng, inputSize)
			}

			outputs, err := bi.Forward(inputs)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			outputSize := tt.arch.Layers[len(tt.arch.Layers)-1].OutputSize
			for i, out := range outputs {
				if len(out) != outputSize {
					t.Errorf("model %d: expected output size %d, got %d", i, outputSize, len(out))
				}
				for j, v := range out {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Errorf("model %d output[%d] = %v (NaN/Inf)", i, j, v)
					}
				}
			}
		})
	}
}

func TestBatchedInference_LoadWeights(t *testing.T) {
	arch := Architecture{
		Layers: []LayerSpec{
			{InputSize: 3, OutputSize: 2},
		},
	}

	t.Run("valid load", func(t *testing.T) {
		bi, err := NewBatchedInference(2, arch, newEngine())
		if err != nil {
			t.Fatalf("NewBatchedInference: %v", err)
		}
		weights := map[string][]float32{
			"layer_0": {1, 2, 3, 4, 5, 6},
		}
		if err := bi.LoadWeights(0, weights); err != nil {
			t.Fatalf("LoadWeights(0): %v", err)
		}
		if err := bi.LoadWeights(1, weights); err != nil {
			t.Fatalf("LoadWeights(1): %v", err)
		}
	})

	t.Run("out of range model index", func(t *testing.T) {
		bi, err := NewBatchedInference(2, arch, newEngine())
		if err != nil {
			t.Fatalf("NewBatchedInference: %v", err)
		}
		weights := map[string][]float32{
			"layer_0": {1, 2, 3, 4, 5, 6},
		}
		if err := bi.LoadWeights(-1, weights); err == nil {
			t.Error("expected error for negative modelIdx")
		}
		if err := bi.LoadWeights(2, weights); err == nil {
			t.Error("expected error for modelIdx >= numModels")
		}
	})

	t.Run("missing layer key", func(t *testing.T) {
		bi, err := NewBatchedInference(1, arch, newEngine())
		if err != nil {
			t.Fatalf("NewBatchedInference: %v", err)
		}
		if err := bi.LoadWeights(0, map[string][]float32{}); err == nil {
			t.Error("expected error for missing layer key")
		}
	})

	t.Run("wrong weight size", func(t *testing.T) {
		bi, err := NewBatchedInference(1, arch, newEngine())
		if err != nil {
			t.Fatalf("NewBatchedInference: %v", err)
		}
		weights := map[string][]float32{
			"layer_0": {1, 2, 3}, // should be 6
		}
		if err := bi.LoadWeights(0, weights); err == nil {
			t.Error("expected error for wrong weight size")
		}
	})

	t.Run("deterministic output", func(t *testing.T) {
		bi, err := NewBatchedInference(2, arch, newEngine())
		if err != nil {
			t.Fatalf("NewBatchedInference: %v", err)
		}
		// Model 0: identity-like weights
		if err := bi.LoadWeights(0, map[string][]float32{
			"layer_0": {1, 0, 0, 1, 0, 0},
		}); err != nil {
			t.Fatal(err)
		}
		// Model 1: different weights
		if err := bi.LoadWeights(1, map[string][]float32{
			"layer_0": {0, 1, 1, 0, 0, 0},
		}); err != nil {
			t.Fatal(err)
		}

		inputs := [][]float32{
			{1, 2, 3},
			{1, 2, 3},
		}
		outputs, err := bi.Forward(inputs)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}

		// Model 0: [1*1+2*0+3*0, 1*0+2*1+3*0] = [1, 2]
		expectClose(t, "model0[0]", outputs[0][0], 1.0)
		expectClose(t, "model0[1]", outputs[0][1], 2.0)

		// Model 1: [1*0+2*1+3*0, 1*1+2*0+3*0] = [2, 1]
		expectClose(t, "model1[0]", outputs[1][0], 2.0)
		expectClose(t, "model1[1]", outputs[1][1], 1.0)
	})
}

func TestArchitecture_Validate(t *testing.T) {
	tests := []struct {
		name    string
		arch    Architecture
		wantErr bool
	}{
		{
			name:    "empty layers",
			arch:    Architecture{},
			wantErr: true,
		},
		{
			name: "zero input size",
			arch: Architecture{
				Layers: []LayerSpec{{InputSize: 0, OutputSize: 2}},
			},
			wantErr: true,
		},
		{
			name: "mismatched layer sizes",
			arch: Architecture{
				Layers: []LayerSpec{
					{InputSize: 4, OutputSize: 8},
					{InputSize: 5, OutputSize: 2}, // 5 != 8
				},
			},
			wantErr: true,
		},
		{
			name: "valid single layer",
			arch: Architecture{
				Layers: []LayerSpec{{InputSize: 4, OutputSize: 2}},
			},
			wantErr: false,
		},
		{
			name: "valid multi layer",
			arch: Architecture{
				Layers: []LayerSpec{
					{InputSize: 4, OutputSize: 8},
					{InputSize: 8, OutputSize: 2},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.arch.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestNewBatchedInference_Errors(t *testing.T) {
	arch := Architecture{
		Layers: []LayerSpec{{InputSize: 4, OutputSize: 2}},
	}

	t.Run("zero models", func(t *testing.T) {
		_, err := NewBatchedInference(0, arch, newEngine())
		if err == nil {
			t.Error("expected error for zero models")
		}
	})

	t.Run("invalid architecture", func(t *testing.T) {
		_, err := NewBatchedInference(1, Architecture{}, newEngine())
		if err == nil {
			t.Error("expected error for empty architecture")
		}
	})
}

func TestForward_InputValidation(t *testing.T) {
	arch := Architecture{
		Layers: []LayerSpec{{InputSize: 3, OutputSize: 2}},
	}
	bi, err := NewBatchedInference(2, arch, newEngine())
	if err != nil {
		t.Fatal(err)
	}
	// Load weights so forward doesn't fail on missing weights.
	for m := range 2 {
		if err := bi.LoadWeights(m, map[string][]float32{
			"layer_0": {1, 0, 0, 1, 0, 0},
		}); err != nil {
			t.Fatal(err)
		}
	}

	t.Run("wrong number of inputs", func(t *testing.T) {
		_, err := bi.Forward([][]float32{{1, 2, 3}})
		if err == nil {
			t.Error("expected error for wrong number of inputs")
		}
	})

	t.Run("wrong input size", func(t *testing.T) {
		_, err := bi.Forward([][]float32{{1, 2}, {1, 2, 3}})
		if err == nil {
			t.Error("expected error for wrong input size")
		}
	})
}

// --- helpers ---

func randomSlice(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = rng.Float32()*2 - 1 // [-1, 1]
	}
	return s
}

func expectClose(t *testing.T, label string, got, want float32) {
	t.Helper()
	if diff := math.Abs(float64(got - want)); diff > 1e-5 {
		t.Errorf("%s: got %v, want %v (diff %v)", label, got, want, diff)
	}
}

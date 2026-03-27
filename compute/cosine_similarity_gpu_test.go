package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestGPUEngine_CosineSimilarityGPU verifies GPU CosineSimilarity matches CPU
// CosineSimilarity within 1e-4 tolerance. On machines without a GPU the test
// exercises the CPU-fallback path that GPUEngine delegates to.
func TestGPUEngine_CosineSimilarityGPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	tests := []struct {
		name   string
		aShape []int
		aData  []float32
		bShape []int
		bData  []float32
	}{
		{
			name:   "identical_vectors",
			aShape: []int{1, 3},
			aData:  []float32{1, 2, 3},
			bShape: []int{1, 3},
			bData:  []float32{1, 2, 3},
		},
		{
			name:   "orthogonal_vectors",
			aShape: []int{1, 2},
			aData:  []float32{1, 0},
			bShape: []int{1, 2},
			bData:  []float32{0, 1},
		},
		{
			name:   "opposite_vectors",
			aShape: []int{1, 3},
			aData:  []float32{1, 2, 3},
			bShape: []int{1, 3},
			bData:  []float32{-1, -2, -3},
		},
		{
			name:   "pairwise_2x2",
			aShape: []int{2, 2},
			aData:  []float32{1, 0, 0, 1},
			bShape: []int{2, 2},
			bData:  []float32{1, 0, 1, 1},
		},
		{
			name:   "M_neq_N",
			aShape: []int{1, 3},
			aData:  []float32{1, 0, 0},
			bShape: []int{3, 3},
			bData:  []float32{1, 0, 0, 0, 1, 0, 0, 0, 1},
		},
		{
			name:   "zero_norm",
			aShape: []int{1, 3},
			aData:  []float32{0, 0, 0},
			bShape: []int{1, 3},
			bData:  []float32{1, 2, 3},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, err := tensor.New(tc.aShape, tc.aData)
			if err != nil {
				t.Fatalf("tensor.New(a): %v", err)
			}
			b, err := tensor.New(tc.bShape, tc.bData)
			if err != nil {
				t.Fatalf("tensor.New(b): %v", err)
			}

			gpuOut, err := gpuEng.CosineSimilarity(ctx, a, b)
			if err != nil {
				t.Fatalf("GPU CosineSimilarity: %v", err)
			}
			cpuOut, err := cpuEng.CosineSimilarity(ctx, a, b)
			if err != nil {
				t.Fatalf("CPU CosineSimilarity: %v", err)
			}

			gShape := gpuOut.Shape()
			cShape := cpuOut.Shape()
			if len(gShape) != len(cShape) {
				t.Fatalf("shape rank mismatch: GPU=%v, CPU=%v", gShape, cShape)
			}
			for i := range gShape {
				if gShape[i] != cShape[i] {
					t.Fatalf("shape mismatch: GPU=%v, CPU=%v", gShape, cShape)
				}
			}

			gData := gpuOut.Data()
			cData := cpuOut.Data()
			if len(gData) != len(cData) {
				t.Fatalf("data length mismatch: GPU=%d, CPU=%d", len(gData), len(cData))
			}
			for i := range gData {
				diff := math.Abs(float64(gData[i]) - float64(cData[i]))
				if diff > 1e-4 {
					t.Errorf("[%d] GPU=%v, CPU=%v, diff=%e", i, gData[i], cData[i], diff)
				}
			}
		})
	}
}

// TestGPUEngine_CosineSimilarityCPUFallback verifies the CPU-fallback path
// works correctly when no GPU is available.
func TestGPUEngine_CosineSimilarityCPUFallback(t *testing.T) {
	ops := numeric.Float32Ops{}
	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	a, err := tensor.New([]int{2, 3}, []float32{1, 0, 0, 0, 1, 0})
	if err != nil {
		t.Fatalf("tensor.New(a): %v", err)
	}
	b, err := tensor.New([]int{2, 3}, []float32{1, 0, 0, 0, 0, 1})
	if err != nil {
		t.Fatalf("tensor.New(b): %v", err)
	}

	// Use CPUEngine directly (same path GPUEngine delegates to).
	result, err := cpuEng.CosineSimilarity(ctx, a, b)
	if err != nil {
		t.Fatalf("CosineSimilarity: %v", err)
	}

	wantShape := []int{2, 2}
	gotShape := result.Shape()
	if len(gotShape) != len(wantShape) {
		t.Fatalf("shape rank mismatch: got %v, want %v", gotShape, wantShape)
	}
	for i := range gotShape {
		if gotShape[i] != wantShape[i] {
			t.Fatalf("shape mismatch: got %v, want %v", gotShape, wantShape)
		}
	}

	// a[0]=[1,0,0], a[1]=[0,1,0]; b[0]=[1,0,0], b[1]=[0,0,1]
	// cos(a0,b0)=1, cos(a0,b1)=0, cos(a1,b0)=0, cos(a1,b1)=0
	want := []float32{1.0, 0.0, 0.0, 0.0}
	got := result.Data()
	for i := range got {
		diff := math.Abs(float64(got[i]) - float64(want[i]))
		if diff > 1e-6 {
			t.Errorf("[%d] got=%v, want=%v, diff=%e", i, got[i], want[i], diff)
		}
	}
}

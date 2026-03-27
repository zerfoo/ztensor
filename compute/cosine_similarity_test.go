package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestCPUEngine_CosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		aShape   []int
		aData    []float32
		bShape   []int
		bData    []float32
		wantShape []int
		want     []float32
	}{
		{
			name:   "identical vectors",
			aShape: []int{1, 3},
			aData:  []float32{1, 2, 3},
			bShape: []int{1, 3},
			bData:  []float32{1, 2, 3},
			wantShape: []int{1, 1},
			want:   []float32{1.0},
		},
		{
			name:   "orthogonal vectors",
			aShape: []int{1, 2},
			aData:  []float32{1, 0},
			bShape: []int{1, 2},
			bData:  []float32{0, 1},
			wantShape: []int{1, 1},
			want:   []float32{0.0},
		},
		{
			name:   "opposite vectors",
			aShape: []int{1, 3},
			aData:  []float32{1, 2, 3},
			bShape: []int{1, 3},
			bData:  []float32{-1, -2, -3},
			wantShape: []int{1, 1},
			want:   []float32{-1.0},
		},
		{
			name:   "pairwise 2x2",
			aShape: []int{2, 2},
			aData:  []float32{1, 0, 0, 1},
			bShape: []int{2, 2},
			bData:  []float32{1, 0, 1, 1},
			wantShape: []int{2, 2},
			// a[0]=[1,0], a[1]=[0,1]; b[0]=[1,0], b[1]=[1,1]
			// cos(a0,b0)=1, cos(a0,b1)=1/sqrt(2)
			// cos(a1,b0)=0, cos(a1,b1)=1/sqrt(2)
			want: []float32{1.0, float32(1.0 / math.Sqrt(2)), 0.0, float32(1.0 / math.Sqrt(2))},
		},
		{
			name:   "M != N",
			aShape: []int{1, 3},
			aData:  []float32{1, 0, 0},
			bShape: []int{3, 3},
			bData:  []float32{1, 0, 0, 0, 1, 0, 0, 0, 1},
			wantShape: []int{1, 3},
			want:   []float32{1.0, 0.0, 0.0},
		},
		{
			name:   "zero-norm vector returns 0",
			aShape: []int{1, 3},
			aData:  []float32{0, 0, 0},
			bShape: []int{1, 3},
			bData:  []float32{1, 2, 3},
			wantShape: []int{1, 1},
			want:   []float32{0.0},
		},
		{
			name:   "both zero-norm vectors returns 0",
			aShape: []int{1, 2},
			aData:  []float32{0, 0},
			bShape: []int{1, 2},
			bData:  []float32{0, 0},
			wantShape: []int{1, 1},
			want:   []float32{0.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := NewCPUEngine[float32](numeric.Float32Ops{})
			ctx := context.Background()

			a, err := tensor.New(tt.aShape, tt.aData)
			if err != nil {
				t.Fatalf("failed to create tensor a: %v", err)
			}
			b, err := tensor.New(tt.bShape, tt.bData)
			if err != nil {
				t.Fatalf("failed to create tensor b: %v", err)
			}

			result, err := eng.CosineSimilarity(ctx, a, b)
			if err != nil {
				t.Fatalf("CosineSimilarity returned error: %v", err)
			}

			gotShape := result.Shape()
			if len(gotShape) != len(tt.wantShape) {
				t.Fatalf("shape mismatch: got %v, want %v", gotShape, tt.wantShape)
			}
			for i := range gotShape {
				if gotShape[i] != tt.wantShape[i] {
					t.Fatalf("shape mismatch: got %v, want %v", gotShape, tt.wantShape)
				}
			}

			got := result.Data()
			if len(got) != len(tt.want) {
				t.Fatalf("data length mismatch: got %d, want %d", len(got), len(tt.want))
			}
			for i, v := range got {
				if math.Abs(float64(v)-float64(tt.want[i])) > 1e-6 {
					t.Errorf("result[%d] = %v, want %v", i, v, tt.want[i])
				}
			}
		})
	}
}

func TestCPUEngine_CosineSimilarity_errors(t *testing.T) {
	eng := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	t.Run("nil input a", func(t *testing.T) {
		b, _ := tensor.New[float32]([]int{1, 3}, nil)
		_, err := eng.CosineSimilarity(ctx, nil, b)
		if err == nil {
			t.Fatal("expected error for nil input a")
		}
	})

	t.Run("nil input b", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{1, 3}, nil)
		_, err := eng.CosineSimilarity(ctx, a, nil)
		if err == nil {
			t.Fatal("expected error for nil input b")
		}
	})

	t.Run("non-2D input", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{3}, nil)
		b, _ := tensor.New[float32]([]int{1, 3}, nil)
		_, err := eng.CosineSimilarity(ctx, a, b)
		if err == nil {
			t.Fatal("expected error for non-2D input")
		}
	})

	t.Run("dimension mismatch", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{1, 3}, nil)
		b, _ := tensor.New[float32]([]int{1, 4}, nil)
		_, err := eng.CosineSimilarity(ctx, a, b)
		if err == nil {
			t.Fatal("expected error for dimension mismatch")
		}
	})
}

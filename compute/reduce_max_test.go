package compute

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestCPUEngine_ReduceMax(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	ctx := context.Background()

	tests := []struct {
		name      string
		shape     []int
		data      []int
		axis      int
		keepDims  bool
		wantShape []int
		wantData  []int
	}{
		{
			name:      "3D axis=0",
			shape:     []int{2, 2, 3},
			data:      []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			axis:      0,
			keepDims:  false,
			wantShape: []int{2, 3},
			wantData:  []int{7, 8, 9, 10, 11, 12},
		},
		{
			name:      "3D axis=1",
			shape:     []int{2, 2, 3},
			data:      []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			axis:      1,
			keepDims:  false,
			wantShape: []int{2, 3},
			wantData:  []int{4, 5, 6, 10, 11, 12},
		},
		{
			name:      "3D axis=2",
			shape:     []int{2, 2, 3},
			data:      []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			axis:      2,
			keepDims:  false,
			wantShape: []int{2, 2},
			wantData:  []int{3, 6, 9, 12},
		},
		{
			name:      "3D axis=1 keepDims",
			shape:     []int{2, 2, 3},
			data:      []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			axis:      1,
			keepDims:  true,
			wantShape: []int{2, 1, 3},
			wantData:  []int{4, 5, 6, 10, 11, 12},
		},
		{
			name:      "2D axis=0",
			shape:     []int{2, 3},
			data:      []int{1, 5, 3, 4, 2, 6},
			axis:      0,
			keepDims:  false,
			wantShape: []int{3},
			wantData:  []int{4, 5, 6},
		},
		{
			name:      "2D axis=1",
			shape:     []int{2, 3},
			data:      []int{1, 5, 3, 4, 2, 6},
			axis:      1,
			keepDims:  false,
			wantShape: []int{2},
			wantData:  []int{5, 6},
		},
		{
			name:      "negative axis reduces all",
			shape:     []int{2, 3},
			data:      []int{1, 5, 3, 4, 2, 6},
			axis:      -1,
			keepDims:  false,
			wantShape: []int{1},
			wantData:  []int{6},
		},
		{
			name:      "negative axis keepDims",
			shape:     []int{2, 3},
			data:      []int{1, 5, 3, 4, 2, 6},
			axis:      -1,
			keepDims:  true,
			wantShape: []int{1, 1},
			wantData:  []int{6},
		},
		{
			name:      "1D axis=0",
			shape:     []int{4},
			data:      []int{3, 1, 4, 2},
			axis:      0,
			keepDims:  false,
			wantShape: []int{1},
			wantData:  []int{4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := tensor.New[int](tt.shape, tt.data)
			if err != nil {
				t.Fatalf("failed to create tensor: %v", err)
			}
			result, err := engine.ReduceMax(ctx, a, tt.axis, tt.keepDims)
			if err != nil {
				t.Fatalf("ReduceMax returned error: %v", err)
			}
			if !reflect.DeepEqual(result.Shape(), tt.wantShape) {
				t.Errorf("shape: got %v, want %v", result.Shape(), tt.wantShape)
			}
			if !reflect.DeepEqual(result.Data(), tt.wantData) {
				t.Errorf("data: got %v, want %v", result.Data(), tt.wantData)
			}
		})
	}
}

func TestCPUEngine_ReduceMax_NilInput(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	_, err := engine.ReduceMax(context.Background(), nil, 0, false)
	if err == nil {
		t.Fatal("expected error for nil input")
	}
}

func TestCPUEngine_ReduceMax_AxisOOB(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})
	_, err := engine.ReduceMax(context.Background(), a, 5, false)
	if err == nil {
		t.Fatal("expected error for out-of-bounds axis")
	}
}

func TestCPUEngine_ReduceMax_Float32(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	a, _ := tensor.New[float32]([]int{2, 2, 2}, []float32{1.5, 3.0, 2.0, 4.5, 5.0, 0.5, 3.5, 2.5})
	result, err := engine.ReduceMax(context.Background(), a, 0, false)
	if err != nil {
		t.Fatalf("ReduceMax returned error: %v", err)
	}
	expected := []float32{5.0, 3.0, 3.5, 4.5}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("got %v, want %v", result.Data(), expected)
	}
}

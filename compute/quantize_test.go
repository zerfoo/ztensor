package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestComputeAmax(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	tests := []struct {
		name string
		data []float32
		want float32
	}{
		{"positive values", []float32{1.0, 3.5, 2.0, 0.5}, 3.5},
		{"single element", []float32{7.0}, 7.0},
		{"large tensor", []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, 1.0},
		{"first is max", []float32{9.0, 1.0, 2.0}, 9.0},
		{"last is max", []float32{1.0, 2.0, 9.0}, 9.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tn, err := tensor.New[float32]([]int{len(tt.data)}, tt.data)
			if err != nil {
				t.Fatal(err)
			}
			got, err := ComputeAmax(ctx, ops, tn)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("ComputeAmax = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestComputeAmax_Negative(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	tests := []struct {
		name string
		data []float32
		want float32
	}{
		{"all negative", []float32{-1.0, -5.0, -2.0}, 5.0},
		{"mixed sign, negative max", []float32{3.0, -7.0, 2.0, -1.0}, 7.0},
		{"mixed sign, positive max", []float32{-3.0, 8.0, -2.0, 1.0}, 8.0},
		{"negative one element", []float32{-42.0}, 42.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tn, err := tensor.New[float32]([]int{len(tt.data)}, tt.data)
			if err != nil {
				t.Fatal(err)
			}
			got, err := ComputeAmax(ctx, ops, tn)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("ComputeAmax = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestComputeAmax_AllZero(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	tn, err := tensor.New[float32]([]int{4}, []float32{0, 0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	got, err := ComputeAmax(ctx, ops, tn)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Errorf("ComputeAmax all-zero = %v, want 0", got)
	}
}

func TestComputeAmax_Empty(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	tn, err := tensor.New[float32]([]int{0}, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ComputeAmax(ctx, ops, tn)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Errorf("ComputeAmax empty = %v, want 0", got)
	}
}

func TestComputeAmax_Nil(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	_, err := ComputeAmax[float32](ctx, ops, nil)
	if err == nil {
		t.Fatal("expected error for nil tensor")
	}
}

func TestComputeAmax_2D(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// 2x3 matrix — amax should scan all elements
	tn, err := tensor.New[float32]([]int{2, 3}, []float32{1, -6, 3, 4, 2, -1})
	if err != nil {
		t.Fatal(err)
	}
	got, err := ComputeAmax(ctx, ops, tn)
	if err != nil {
		t.Fatal(err)
	}
	if got != 6.0 {
		t.Errorf("ComputeAmax 2D = %v, want 6.0", got)
	}
}

func TestScaleForFP8(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	tests := []struct {
		name string
		data []float32
		want float32
	}{
		{"amax=1", []float32{0.5, -1.0, 0.3}, 448.0},
		{"amax=4", []float32{2.0, -4.0, 3.0}, 112.0},
		{"amax=448", []float32{448.0, 0.0}, 1.0},
		{"amax=0.5", []float32{0.5, -0.1, 0.2}, 896.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tn, err := tensor.New[float32]([]int{len(tt.data)}, tt.data)
			if err != nil {
				t.Fatal(err)
			}
			got, err := ScaleForFP8(ctx, ops, tn)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("ScaleForFP8 = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestScaleForFP8_AllZero(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	tn, err := tensor.New[float32]([]int{3}, []float32{0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	got, err := ScaleForFP8(ctx, ops, tn)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(float64(got), 1) {
		t.Errorf("ScaleForFP8 all-zero = %v, want +Inf", got)
	}
}

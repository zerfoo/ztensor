package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestHadamardTransform1D(t *testing.T) {
	eng := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// Input: [1, 0, 0, 0] -> H * e0 = first column of H4 = [0.5, 0.5, 0.5, 0.5]
	input, err := tensor.New[float32]([]int{4}, []float32{1, 0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}

	result, err := eng.HadamardTransform(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	expected := float32(1.0 / math.Sqrt(4.0)) // 0.5
	data := result.Data()
	for i, v := range data {
		if diff := float32(math.Abs(float64(v - expected))); diff > 1e-4 {
			t.Fatalf("element %d: got %f, want %f (diff %f)", i, v, expected, diff)
		}
	}
}

func TestHadamardTransform2D(t *testing.T) {
	eng := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// Two rows: e0 and e1 of R^4
	input, err := tensor.New[float32]([]int{2, 4}, []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	})
	if err != nil {
		t.Fatal(err)
	}

	result, err := eng.HadamardTransform(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	// H4 first column (row 0) and second column (row 1), normalized.
	// H4 = 1/2 * [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
	// H * e0 = [0.5, 0.5, 0.5, 0.5]
	// H * e1 = [0.5, -0.5, 0.5, -0.5]
	want := [][]float32{
		{0.5, 0.5, 0.5, 0.5},
		{0.5, -0.5, 0.5, -0.5},
	}

	data := result.Data()
	for row := 0; row < 2; row++ {
		for col := 0; col < 4; col++ {
			got := data[row*4+col]
			exp := want[row][col]
			if diff := float32(math.Abs(float64(got - exp))); diff > 1e-4 {
				t.Fatalf("row %d col %d: got %f, want %f", row, col, got, exp)
			}
		}
	}
}

func TestHadamardTransformRoundtrip(t *testing.T) {
	eng := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// Applying the normalized Hadamard transform twice should return the original vector.
	original := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	input, err := tensor.New[float32]([]int{8}, append([]float32(nil), original...))
	if err != nil {
		t.Fatal(err)
	}

	first, err := eng.HadamardTransform(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	second, err := eng.HadamardTransform(ctx, first)
	if err != nil {
		t.Fatal(err)
	}

	data := second.Data()
	for i, v := range data {
		if diff := float32(math.Abs(float64(v - original[i]))); diff > 1e-4 {
			t.Fatalf("roundtrip element %d: got %f, want %f (diff %f)", i, v, original[i], diff)
		}
	}
}

func TestHadamardTransformRoundtripBatch(t *testing.T) {
	eng := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	original := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	input, err := tensor.New[float32]([]int{2, 4}, append([]float32(nil), original...))
	if err != nil {
		t.Fatal(err)
	}

	first, err := eng.HadamardTransform(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	second, err := eng.HadamardTransform(ctx, first)
	if err != nil {
		t.Fatal(err)
	}

	data := second.Data()
	for i, v := range data {
		if diff := float32(math.Abs(float64(v - original[i]))); diff > 1e-4 {
			t.Fatalf("roundtrip element %d: got %f, want %f", i, v, original[i])
		}
	}
}

func TestHadamardTransformAgainstMatrix(t *testing.T) {
	eng := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// Compare FWHT result against explicit matrix multiplication with HadamardMatrix.
	dim := 8
	H, err := HadamardMatrix[float32](dim)
	if err != nil {
		t.Fatal(err)
	}

	input, err := tensor.New[float32]([]int{1, dim}, []float32{3, 1, 4, 1, 5, 9, 2, 6})
	if err != nil {
		t.Fatal(err)
	}

	// FWHT
	fwhtResult, err := eng.HadamardTransform(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	// Explicit matmul: input * H^T = input * H (H is symmetric)
	matmulResult, err := eng.MatMul(ctx, input, H)
	if err != nil {
		t.Fatal(err)
	}

	fwhtData := fwhtResult.Data()
	matmulData := matmulResult.Data()
	for i := range fwhtData {
		if diff := float32(math.Abs(float64(fwhtData[i] - matmulData[i]))); diff > 1e-4 {
			t.Fatalf("element %d: FWHT=%f, MatMul=%f (diff %f)", i, fwhtData[i], matmulData[i], diff)
		}
	}
}

func TestHadamardTransformErrors(t *testing.T) {
	eng := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// nil input
	if _, err := eng.HadamardTransform(ctx, nil); err == nil {
		t.Fatal("expected error for nil input")
	}

	// 3D input
	input3d, _ := tensor.New[float32]([]int{2, 2, 2}, nil)
	if _, err := eng.HadamardTransform(ctx, input3d); err == nil {
		t.Fatal("expected error for 3D input")
	}

	// Non-power-of-2 dim
	input3, _ := tensor.New[float32]([]int{3}, nil)
	if _, err := eng.HadamardTransform(ctx, input3); err == nil {
		t.Fatal("expected error for non-power-of-2 dim")
	}

	// Dim > 512
	input1024, _ := tensor.New[float32]([]int{1024}, nil)
	if _, err := eng.HadamardTransform(ctx, input1024); err == nil {
		t.Fatal("expected error for dim > 512")
	}
}

func TestHadamardTransformDest(t *testing.T) {
	eng := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	input, _ := tensor.New[float32]([]int{4}, []float32{1, 0, 0, 0})
	dst, _ := tensor.New[float32]([]int{4}, nil)

	result, err := eng.HadamardTransform(ctx, input, dst)
	if err != nil {
		t.Fatal(err)
	}
	if result != dst {
		t.Fatal("expected result to be the provided dst tensor")
	}

	expected := float32(0.5)
	for i, v := range result.Data() {
		if diff := float32(math.Abs(float64(v - expected))); diff > 1e-4 {
			t.Fatalf("element %d: got %f, want %f", i, v, expected)
		}
	}
}

func TestHadamardTransformFloat64(t *testing.T) {
	eng := NewCPUEngine(numeric.Float64Ops{})
	ctx := context.Background()

	original := []float64{1, 2, 3, 4}
	input, _ := tensor.New[float64]([]int{4}, append([]float64(nil), original...))

	first, err := eng.HadamardTransform(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	second, err := eng.HadamardTransform(ctx, first)
	if err != nil {
		t.Fatal(err)
	}

	for i, v := range second.Data() {
		if diff := math.Abs(v - original[i]); diff > 1e-10 {
			t.Fatalf("roundtrip element %d: got %f, want %f", i, v, original[i])
		}
	}
}

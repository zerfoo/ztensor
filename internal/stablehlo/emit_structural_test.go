package stablehlo

import (
	"strings"
	"testing"
)

func TestEmitMatMul2D(t *testing.T) {
	namer := &SSANamer{}
	line, result, err := EmitMatMul(namer, "%a", "%b", []int{4, 3}, []int{3, 5}, DTypeF32)
	if err != nil {
		t.Fatal(err)
	}
	if result != "%v0" {
		t.Errorf("expected result %%v0, got %s", result)
	}
	want := `%v0 = stablehlo.dot_general %a, %b, batching_dims = [] x [], contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x5xf32>) -> tensor<4x5xf32>`
	if line != want {
		t.Errorf("mismatch:\ngot:  %s\nwant: %s", line, want)
	}
}

func TestEmitMatMulBatched(t *testing.T) {
	namer := &SSANamer{}
	line, result, err := EmitMatMul(namer, "%a", "%b", []int{2, 4, 3}, []int{2, 3, 5}, DTypeF32)
	if err != nil {
		t.Fatal(err)
	}
	if result != "%v0" {
		t.Errorf("expected result %%v0, got %s", result)
	}
	want := `%v0 = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x4x3xf32>, tensor<2x3x5xf32>) -> tensor<2x4x5xf32>`
	if line != want {
		t.Errorf("mismatch:\ngot:  %s\nwant: %s", line, want)
	}
}

func TestEmitMatMulContractionMismatch(t *testing.T) {
	namer := &SSANamer{}
	_, _, err := EmitMatMul(namer, "%a", "%b", []int{4, 3}, []int{7, 5}, DTypeF32)
	if err == nil {
		t.Fatal("expected error for contraction dimension mismatch")
	}
}

func TestEmitMatMulRank1(t *testing.T) {
	namer := &SSANamer{}
	_, _, err := EmitMatMul(namer, "%a", "%b", []int{4}, []int{4}, DTypeF32)
	if err == nil {
		t.Fatal("expected error for rank-1 inputs")
	}
}

func TestEmitTranspose(t *testing.T) {
	namer := &SSANamer{}
	line, result, err := EmitTranspose(namer, "%a", []int{2, 3, 4}, []int{2, 0, 1}, DTypeF32)
	if err != nil {
		t.Fatal(err)
	}
	if result != "%v0" {
		t.Errorf("expected result %%v0, got %s", result)
	}
	want := `%v0 = stablehlo.transpose %a, permutation = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>`
	if line != want {
		t.Errorf("mismatch:\ngot:  %s\nwant: %s", line, want)
	}
}

func TestEmitTransposeInvalidPerm(t *testing.T) {
	namer := &SSANamer{}
	_, _, err := EmitTranspose(namer, "%a", []int{2, 3}, []int{0}, DTypeF32)
	if err == nil {
		t.Fatal("expected error for perm length mismatch")
	}
}

func TestEmitReshape(t *testing.T) {
	namer := &SSANamer{}
	line, result, err := EmitReshape(namer, "%a", []int{2, 3, 4}, []int{6, 4}, DTypeF32)
	if err != nil {
		t.Fatal(err)
	}
	if result != "%v0" {
		t.Errorf("expected result %%v0, got %s", result)
	}
	want := `%v0 = stablehlo.reshape %a : (tensor<2x3x4xf32>) -> tensor<6x4xf32>`
	if line != want {
		t.Errorf("mismatch:\ngot:  %s\nwant: %s", line, want)
	}
}

func TestEmitReshapeElementMismatch(t *testing.T) {
	namer := &SSANamer{}
	_, _, err := EmitReshape(namer, "%a", []int{2, 3}, []int{7}, DTypeF32)
	if err == nil {
		t.Fatal("expected error for element count mismatch")
	}
}

func TestEmitConcat(t *testing.T) {
	namer := &SSANamer{}
	line, result, err := EmitConcat(namer,
		[]string{"%a", "%b"},
		[][]int{{2, 3}, {2, 5}},
		1, DTypeF32,
	)
	if err != nil {
		t.Fatal(err)
	}
	if result != "%v0" {
		t.Errorf("expected result %%v0, got %s", result)
	}
	want := `%v0 = stablehlo.concatenate %a, %b, dimension = 1 : tensor<2x8xf32>`
	if line != want {
		t.Errorf("mismatch:\ngot:  %s\nwant: %s", line, want)
	}
}

func TestEmitConcatThreeInputs(t *testing.T) {
	namer := &SSANamer{}
	line, _, err := EmitConcat(namer,
		[]string{"%a", "%b", "%c"},
		[][]int{{4, 2}, {4, 3}, {4, 1}},
		1, DTypeF64,
	)
	if err != nil {
		t.Fatal(err)
	}
	want := `%v0 = stablehlo.concatenate %a, %b, %c, dimension = 1 : tensor<4x6xf64>`
	if line != want {
		t.Errorf("mismatch:\ngot:  %s\nwant: %s", line, want)
	}
}

func TestEmitSlice(t *testing.T) {
	namer := &SSANamer{}
	line, result, err := EmitSlice(namer, "%a",
		[]int{8, 6},
		[]int{1, 0}, []int{5, 4}, nil,
		DTypeF32,
	)
	if err != nil {
		t.Fatal(err)
	}
	if result != "%v0" {
		t.Errorf("expected result %%v0, got %s", result)
	}
	want := `%v0 = stablehlo.slice %a, starts = [1, 0], limits = [5, 4], strides = [1, 1] : (tensor<8x6xf32>) -> tensor<4x4xf32>`
	if line != want {
		t.Errorf("mismatch:\ngot:  %s\nwant: %s", line, want)
	}
}

func TestEmitSliceWithStrides(t *testing.T) {
	namer := &SSANamer{}
	line, _, err := EmitSlice(namer, "%a",
		[]int{10},
		[]int{0}, []int{10}, []int{2},
		DTypeF32,
	)
	if err != nil {
		t.Fatal(err)
	}
	want := `%v0 = stablehlo.slice %a, starts = [0], limits = [10], strides = [2] : (tensor<10xf32>) -> tensor<5xf32>`
	if line != want {
		t.Errorf("mismatch:\ngot:  %s\nwant: %s", line, want)
	}
}

func TestEmitSliceInvalidRange(t *testing.T) {
	namer := &SSANamer{}
	_, _, err := EmitSlice(namer, "%a", []int{4}, []int{3}, []int{1}, nil, DTypeF32)
	if err == nil {
		t.Fatal("expected error for invalid range (start > limit)")
	}
}

func TestEmitGather(t *testing.T) {
	namer := &SSANamer{}
	line, result, err := EmitGather(namer, "%data", "%indices",
		[]int{10, 8},  // operand shape
		[]int{3, 1},   // indices shape
		[]int{1, 8},   // slice sizes
		[]int{1},      // offset dims
		[]int{0},      // collapsed slice dims
		[]int{0},      // start index map
		1,             // index vector dim
		DTypeF32,
	)
	if err != nil {
		t.Fatal(err)
	}
	if result != "%v0" {
		t.Errorf("expected result %%v0, got %s", result)
	}
	// Output shape from InferStructuralShape(Gather): indices[:-1] + sliceSizes = [3] + [1, 8] = [3, 1, 8]
	if !strings.Contains(line, "stablehlo.gather") {
		t.Errorf("expected stablehlo.gather in output, got: %s", line)
	}
	if !strings.Contains(line, "offset_dims = [1]") {
		t.Errorf("expected offset_dims = [1], got: %s", line)
	}
	if !strings.Contains(line, "collapsed_slice_dims = [0]") {
		t.Errorf("expected collapsed_slice_dims = [0], got: %s", line)
	}
	if !strings.Contains(line, "start_index_map = [0]") {
		t.Errorf("expected start_index_map = [0], got: %s", line)
	}
	if !strings.Contains(line, "index_vector_dim = 1") {
		t.Errorf("expected index_vector_dim = 1, got: %s", line)
	}
	if !strings.Contains(line, "slice_sizes = [1, 8]") {
		t.Errorf("expected slice_sizes = [1, 8], got: %s", line)
	}
}

func TestEmitMatMulF16(t *testing.T) {
	namer := &SSANamer{}
	line, _, err := EmitMatMul(namer, "%a", "%b", []int{8, 16}, []int{16, 32}, DTypeF16)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(line, "f16") {
		t.Errorf("expected f16 dtype in output, got: %s", line)
	}
	if !strings.Contains(line, "tensor<8x32xf16>") {
		t.Errorf("expected output type tensor<8x32xf16>, got: %s", line)
	}
}

func TestSSANamerCounterAdvances(t *testing.T) {
	namer := &SSANamer{}

	_, r0, _ := EmitReshape(namer, "%a", []int{6}, []int{2, 3}, DTypeF32)
	_, r1, _ := EmitReshape(namer, "%b", []int{6}, []int{3, 2}, DTypeF32)
	_, r2, _ := EmitTranspose(namer, "%c", []int{2, 3}, []int{1, 0}, DTypeF32)

	if r0 != "%v0" || r1 != "%v1" || r2 != "%v2" {
		t.Errorf("expected %%v0, %%v1, %%v2 but got %s, %s, %s", r0, r1, r2)
	}
}

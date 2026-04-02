package stablehlo

import (
	"slices"
	"testing"
)

func TestInferMatMul2D(t *testing.T) {
	got, err := InferStructuralShape("MatMul", [][]int{{2, 3}, {3, 4}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 4}) {
		t.Errorf("got %v, want [2 4]", got)
	}
}

func TestInferMatMulBatched3D(t *testing.T) {
	got, err := InferStructuralShape("MatMul", [][]int{{5, 2, 3}, {5, 3, 4}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{5, 2, 4}) {
		t.Errorf("got %v, want [5 2 4]", got)
	}
}

func TestInferMatMulBatched4D(t *testing.T) {
	got, err := InferStructuralShape("MatMul", [][]int{{2, 3, 4, 5}, {2, 3, 5, 6}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 3, 4, 6}) {
		t.Errorf("got %v, want [2 3 4 6]", got)
	}
}

func TestInferMatMulSquare(t *testing.T) {
	got, err := InferStructuralShape("MatMul", [][]int{{3, 3}, {3, 3}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{3, 3}) {
		t.Errorf("got %v, want [3 3]", got)
	}
}

func TestInferMatMulContractionMismatch(t *testing.T) {
	_, err := InferStructuralShape("MatMul", [][]int{{2, 3}, {4, 5}}, nil)
	if err == nil {
		t.Fatal("expected error for contraction dimension mismatch")
	}
}

func TestInferMatMulBatchMismatch(t *testing.T) {
	_, err := InferStructuralShape("MatMul", [][]int{{2, 3, 4}, {5, 4, 6}}, nil)
	if err == nil {
		t.Fatal("expected error for batch dimension mismatch")
	}
}

func TestInferMatMulRank1(t *testing.T) {
	_, err := InferStructuralShape("MatMul", [][]int{{3}, {3, 4}}, nil)
	if err == nil {
		t.Fatal("expected error for rank-1 input")
	}
}

func TestInferMatMulWrongInputCount(t *testing.T) {
	_, err := InferStructuralShape("MatMul", [][]int{{2, 3}}, nil)
	if err == nil {
		t.Fatal("expected error for single input")
	}
}

func TestInferMatMulBatchRankMismatch(t *testing.T) {
	_, err := InferStructuralShape("MatMul", [][]int{{2, 3, 4}, {3, 4, 5}}, nil)
	if err == nil {
		t.Fatal("expected error for batch rank mismatch (batch dims [2] vs [3])")
	}
}

func TestInferTranspose(t *testing.T) {
	got, err := InferStructuralShape("Transpose", [][]int{{2, 3, 4}}, map[string]any{"perm": []int{2, 0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{4, 2, 3}) {
		t.Errorf("got %v, want [4 2 3]", got)
	}
}

func TestInferTranspose2D(t *testing.T) {
	got, err := InferStructuralShape("Transpose", [][]int{{3, 7}}, map[string]any{"perm": []int{1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{7, 3}) {
		t.Errorf("got %v, want [7 3]", got)
	}
}

func TestInferTransposeIdentity(t *testing.T) {
	got, err := InferStructuralShape("Transpose", [][]int{{2, 3, 4}}, map[string]any{"perm": []int{0, 1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 3, 4}) {
		t.Errorf("got %v, want [2 3 4]", got)
	}
}

func TestInferTransposeMissingPerm(t *testing.T) {
	_, err := InferStructuralShape("Transpose", [][]int{{2, 3}}, nil)
	if err == nil {
		t.Fatal("expected error for missing perm attr")
	}
}

func TestInferTransposePermLengthMismatch(t *testing.T) {
	_, err := InferStructuralShape("Transpose", [][]int{{2, 3, 4}}, map[string]any{"perm": []int{1, 0}})
	if err == nil {
		t.Fatal("expected error for perm length mismatch")
	}
}

func TestInferTransposeDuplicateAxis(t *testing.T) {
	_, err := InferStructuralShape("Transpose", [][]int{{2, 3, 4}}, map[string]any{"perm": []int{0, 0, 1}})
	if err == nil {
		t.Fatal("expected error for duplicate axis in perm")
	}
}

func TestInferTransposeOutOfRange(t *testing.T) {
	_, err := InferStructuralShape("Transpose", [][]int{{2, 3}}, map[string]any{"perm": []int{0, 5}})
	if err == nil {
		t.Fatal("expected error for perm axis out of range")
	}
}

func TestInferReshape(t *testing.T) {
	got, err := InferStructuralShape("Reshape", [][]int{{2, 3, 4}}, map[string]any{"shape": []int{6, 4}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{6, 4}) {
		t.Errorf("got %v, want [6 4]", got)
	}
}

func TestInferReshapeFlatten(t *testing.T) {
	got, err := InferStructuralShape("Reshape", [][]int{{2, 3, 4}}, map[string]any{"shape": []int{24}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{24}) {
		t.Errorf("got %v, want [24]", got)
	}
}

func TestInferReshapeToScalar(t *testing.T) {
	got, err := InferStructuralShape("Reshape", [][]int{{1}}, map[string]any{"shape": []int{}})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("got %v, want [] (scalar)", got)
	}
}

func TestInferReshapeFromScalar(t *testing.T) {
	got, err := InferStructuralShape("Reshape", [][]int{{}}, map[string]any{"shape": []int{1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{1, 1}) {
		t.Errorf("got %v, want [1 1]", got)
	}
}

func TestInferReshapeElementCountMismatch(t *testing.T) {
	_, err := InferStructuralShape("Reshape", [][]int{{2, 3}}, map[string]any{"shape": []int{7}})
	if err == nil {
		t.Fatal("expected error for element count mismatch")
	}
}

func TestInferReshapeMissingShape(t *testing.T) {
	_, err := InferStructuralShape("Reshape", [][]int{{2, 3}}, nil)
	if err == nil {
		t.Fatal("expected error for missing shape attr")
	}
}

func TestInferReshapeNotAliased(t *testing.T) {
	target := []int{6, 4}
	got, err := InferStructuralShape("Reshape", [][]int{{2, 3, 4}}, map[string]any{"shape": target})
	if err != nil {
		t.Fatal(err)
	}
	got[0] = 999
	if target[0] != 6 {
		t.Error("InferStructuralShape(Reshape) returned a slice that aliases the target attr")
	}
}

func TestInferConcat(t *testing.T) {
	got, err := InferStructuralShape("Concat", [][]int{{2, 3}, {2, 5}}, map[string]any{"axis": 1})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 8}) {
		t.Errorf("got %v, want [2 8]", got)
	}
}

func TestInferConcatAxis0(t *testing.T) {
	got, err := InferStructuralShape("Concat", [][]int{{2, 3}, {4, 3}}, map[string]any{"axis": 0})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{6, 3}) {
		t.Errorf("got %v, want [6 3]", got)
	}
}

func TestInferConcatMultiple(t *testing.T) {
	got, err := InferStructuralShape("Concat", [][]int{{2, 3}, {2, 4}, {2, 1}}, map[string]any{"axis": 1})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 8}) {
		t.Errorf("got %v, want [2 8]", got)
	}
}

func TestInferConcat3D(t *testing.T) {
	got, err := InferStructuralShape("Concat", [][]int{{2, 3, 4}, {2, 3, 6}}, map[string]any{"axis": 2})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 3, 10}) {
		t.Errorf("got %v, want [2 3 10]", got)
	}
}

func TestInferConcatDimMismatch(t *testing.T) {
	_, err := InferStructuralShape("Concat", [][]int{{2, 3}, {4, 5}}, map[string]any{"axis": 1})
	if err == nil {
		t.Fatal("expected error for non-concat dimension mismatch")
	}
}

func TestInferConcatRankMismatch(t *testing.T) {
	_, err := InferStructuralShape("Concat", [][]int{{2, 3}, {2, 3, 4}}, map[string]any{"axis": 0})
	if err == nil {
		t.Fatal("expected error for rank mismatch")
	}
}

func TestInferConcatAxisOutOfRange(t *testing.T) {
	_, err := InferStructuralShape("Concat", [][]int{{2, 3}, {2, 4}}, map[string]any{"axis": 5})
	if err == nil {
		t.Fatal("expected error for axis out of range")
	}
}

func TestInferConcatSingleInput(t *testing.T) {
	_, err := InferStructuralShape("Concat", [][]int{{2, 3}}, map[string]any{"axis": 0})
	if err == nil {
		t.Fatal("expected error for single input")
	}
}

func TestInferSlice(t *testing.T) {
	got, err := InferStructuralShape("Slice", [][]int{{10, 20}}, map[string]any{
		"start": []int{2, 5},
		"end":   []int{5, 15},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{3, 10}) {
		t.Errorf("got %v, want [3 10]", got)
	}
}

func TestInferSlice3D(t *testing.T) {
	got, err := InferStructuralShape("Slice", [][]int{{8, 6, 4}}, map[string]any{
		"start": []int{0, 1, 2},
		"end":   []int{8, 4, 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{8, 3, 2}) {
		t.Errorf("got %v, want [8 3 2]", got)
	}
}

func TestInferSliceFullDim(t *testing.T) {
	got, err := InferStructuralShape("Slice", [][]int{{5, 10}}, map[string]any{
		"start": []int{0, 0},
		"end":   []int{5, 10},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{5, 10}) {
		t.Errorf("got %v, want [5 10]", got)
	}
}

func TestInferSliceInvalidRange(t *testing.T) {
	_, err := InferStructuralShape("Slice", [][]int{{10, 20}}, map[string]any{
		"start": []int{5, 0},
		"end":   []int{3, 10},
	})
	if err == nil {
		t.Fatal("expected error for start > end")
	}
}

func TestInferSliceOutOfBounds(t *testing.T) {
	_, err := InferStructuralShape("Slice", [][]int{{10, 20}}, map[string]any{
		"start": []int{0, 0},
		"end":   []int{11, 20},
	})
	if err == nil {
		t.Fatal("expected error for end > dim size")
	}
}

func TestInferSliceMissingAttrs(t *testing.T) {
	_, err := InferStructuralShape("Slice", [][]int{{10}}, map[string]any{"start": []int{0}})
	if err == nil {
		t.Fatal("expected error for missing end attr")
	}

	_, err = InferStructuralShape("Slice", [][]int{{10}}, map[string]any{"end": []int{5}})
	if err == nil {
		t.Fatal("expected error for missing start attr")
	}
}

func TestInferSliceLengthMismatch(t *testing.T) {
	_, err := InferStructuralShape("Slice", [][]int{{10, 20}}, map[string]any{
		"start": []int{0},
		"end":   []int{5, 10},
	})
	if err == nil {
		t.Fatal("expected error for start/end length mismatch")
	}
}

func TestInferGather(t *testing.T) {
	// operand [10, 20], indices [3, 1], sliceSizes [5] -> [3, 5]
	got, err := InferStructuralShape("Gather", [][]int{{10, 20}, {3, 1}}, map[string]any{
		"sliceSizes": []int{5},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{3, 5}) {
		t.Errorf("got %v, want [3 5]", got)
	}
}

func TestInferGatherBatched(t *testing.T) {
	// operand [100, 64], indices [4, 8, 1], sliceSizes [64] -> [4, 8, 64]
	got, err := InferStructuralShape("Gather", [][]int{{100, 64}, {4, 8, 1}}, map[string]any{
		"sliceSizes": []int{64},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{4, 8, 64}) {
		t.Errorf("got %v, want [4 8 64]", got)
	}
}

func TestInferGatherMissingSliceSizes(t *testing.T) {
	_, err := InferStructuralShape("Gather", [][]int{{10, 20}, {3, 1}}, nil)
	if err == nil {
		t.Fatal("expected error for missing sliceSizes attr")
	}
}

func TestInferGatherWrongInputCount(t *testing.T) {
	_, err := InferStructuralShape("Gather", [][]int{{10, 20}}, map[string]any{"sliceSizes": []int{5}})
	if err == nil {
		t.Fatal("expected error for single input (missing indices)")
	}
}

func TestInferReduceSum(t *testing.T) {
	got, err := InferStructuralShape("ReduceSum", [][]int{{2, 3, 4}}, map[string]any{"axis": 1})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 4}) {
		t.Errorf("got %v, want [2 4]", got)
	}
}

func TestInferReduceMax(t *testing.T) {
	got, err := InferStructuralShape("ReduceMax", [][]int{{2, 3, 4}}, map[string]any{"axis": 0})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{3, 4}) {
		t.Errorf("got %v, want [3 4]", got)
	}
}

func TestInferReduceMean(t *testing.T) {
	got, err := InferStructuralShape("ReduceMean", [][]int{{2, 3, 4}}, map[string]any{"axis": 2})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 3}) {
		t.Errorf("got %v, want [2 3]", got)
	}
}

func TestInferReduceKeepDims(t *testing.T) {
	got, err := InferStructuralShape("ReduceSum", [][]int{{2, 3, 4}}, map[string]any{
		"axis":     1,
		"keepDims": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 1, 4}) {
		t.Errorf("got %v, want [2 1 4]", got)
	}
}

func TestInferReduceKeepDimsAxis0(t *testing.T) {
	got, err := InferStructuralShape("ReduceMax", [][]int{{5, 3}}, map[string]any{
		"axis":     0,
		"keepDims": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{1, 3}) {
		t.Errorf("got %v, want [1 3]", got)
	}
}

func TestInferReduceKeepDimsFalse(t *testing.T) {
	got, err := InferStructuralShape("ReduceSum", [][]int{{2, 3, 4}}, map[string]any{
		"axis":     1,
		"keepDims": false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 4}) {
		t.Errorf("got %v, want [2 4]", got)
	}
}

func TestInferReduce2DToScalar(t *testing.T) {
	// Reducing axis 0 of a rank-1 tensor produces a scalar.
	got, err := InferStructuralShape("ReduceSum", [][]int{{5}}, map[string]any{"axis": 0})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("got %v, want [] (scalar)", got)
	}
}

func TestInferReduceAxisOutOfRange(t *testing.T) {
	_, err := InferStructuralShape("ReduceSum", [][]int{{2, 3}}, map[string]any{"axis": 3})
	if err == nil {
		t.Fatal("expected error for axis out of range")
	}
}

func TestInferReduceMissingAxis(t *testing.T) {
	_, err := InferStructuralShape("ReduceSum", [][]int{{2, 3}}, nil)
	if err == nil {
		t.Fatal("expected error for missing axis attr")
	}
}

func TestInferReduceWrongInputCount(t *testing.T) {
	_, err := InferStructuralShape("ReduceSum", [][]int{{2, 3}, {2, 3}}, map[string]any{"axis": 0})
	if err == nil {
		t.Fatal("expected error for two inputs")
	}
}

func TestInferStructuralShapeUnsupportedOp(t *testing.T) {
	_, err := InferStructuralShape("FooBarBaz", [][]int{{2, 3}}, nil)
	if err == nil {
		t.Fatal("expected error for unsupported op")
	}
}

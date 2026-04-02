package stablehlo

import (
	"strings"
	"testing"
)

func TestEmitProgram_SingleAdd(t *testing.T) {
	ops := []ProgramOp{
		{
			OpName:      "Add",
			InputSlots:  []int{0, 1},
			OutputSlot:  2,
			InputShapes: [][]int{{2, 3}, {2, 3}},
			OutputShape: []int{2, 3},
			Dtype:       DTypeF32,
		},
	}
	mlir, err := EmitProgram(ops, []int{0, 1}, [][]int{{2, 3}, {2, 3}}, DTypeF32)
	if err != nil {
		t.Fatal(err)
	}

	// Verify module structure.
	if !strings.HasPrefix(mlir, "module {") {
		t.Errorf("missing module header:\n%s", mlir)
	}
	if !strings.HasSuffix(mlir, "}") {
		t.Errorf("missing module closing brace:\n%s", mlir)
	}
	if !strings.Contains(mlir, "func.func @main") {
		t.Errorf("missing func.func @main:\n%s", mlir)
	}

	// Verify function signature.
	if !strings.Contains(mlir, "%arg0: tensor<2x3xf32>") {
		t.Errorf("missing %%arg0 declaration:\n%s", mlir)
	}
	if !strings.Contains(mlir, "%arg1: tensor<2x3xf32>") {
		t.Errorf("missing %%arg1 declaration:\n%s", mlir)
	}
	if !strings.Contains(mlir, "-> tensor<2x3xf32>") {
		t.Errorf("missing return type:\n%s", mlir)
	}

	// Verify add op.
	if !strings.Contains(mlir, "stablehlo.add %arg0, %arg1 : tensor<2x3xf32>") {
		t.Errorf("missing stablehlo.add:\n%s", mlir)
	}

	// Verify return.
	if !strings.Contains(mlir, "return %v0 : tensor<2x3xf32>") {
		t.Errorf("missing return:\n%s", mlir)
	}
}

func TestEmitProgram_MatMulAddSoftmax(t *testing.T) {
	// matmul(slot0, slot1) -> slot2
	// add(slot2, slot3)     -> slot4
	// softmax(slot4)        -> slot5
	ops := []ProgramOp{
		{
			OpName:      "MatMul",
			InputSlots:  []int{0, 1},
			OutputSlot:  2,
			InputShapes: [][]int{{2, 3}, {3, 4}},
			OutputShape: []int{2, 4},
			Dtype:       DTypeF32,
		},
		{
			OpName:      "Add",
			InputSlots:  []int{2, 3},
			OutputSlot:  4,
			InputShapes: [][]int{{2, 4}, {2, 4}},
			OutputShape: []int{2, 4},
			Dtype:       DTypeF32,
		},
		{
			OpName:      "Softmax",
			InputSlots:  []int{4},
			OutputSlot:  5,
			InputShapes: [][]int{{2, 4}},
			OutputShape: []int{2, 4},
			Dtype:       DTypeF32,
		},
	}

	inputSlots := []int{0, 1, 3}
	inputShapes := [][]int{{2, 3}, {3, 4}, {2, 4}}

	mlir, err := EmitProgram(ops, inputSlots, inputShapes, DTypeF32)
	if err != nil {
		t.Fatal(err)
	}

	// Verify module structure.
	if !strings.HasPrefix(mlir, "module {") {
		t.Errorf("missing module header:\n%s", mlir)
	}
	if !strings.Contains(mlir, "func.func @main") {
		t.Errorf("missing func.func @main:\n%s", mlir)
	}

	// Verify all three inputs in the signature.
	if !strings.Contains(mlir, "%arg0: tensor<2x3xf32>") {
		t.Errorf("missing %%arg0:\n%s", mlir)
	}
	if !strings.Contains(mlir, "%arg1: tensor<3x4xf32>") {
		t.Errorf("missing %%arg1:\n%s", mlir)
	}
	if !strings.Contains(mlir, "%arg2: tensor<2x4xf32>") {
		t.Errorf("missing %%arg2:\n%s", mlir)
	}

	// Verify ops present.
	if !strings.Contains(mlir, "stablehlo.dot_general") {
		t.Errorf("missing dot_general:\n%s", mlir)
	}
	if !strings.Contains(mlir, "stablehlo.add") {
		t.Errorf("missing stablehlo.add:\n%s", mlir)
	}
	// Softmax decomposes into reduce, broadcast, subtract, exp, reduce, broadcast, divide.
	if !strings.Contains(mlir, "stablehlo.exponential") {
		t.Errorf("missing stablehlo.exponential (from Softmax):\n%s", mlir)
	}
	if !strings.Contains(mlir, "stablehlo.divide") {
		t.Errorf("missing stablehlo.divide (from Softmax):\n%s", mlir)
	}

	// Verify return type matches softmax output.
	if !strings.Contains(mlir, "-> tensor<2x4xf32>") {
		t.Errorf("return type should be tensor<2x4xf32>:\n%s", mlir)
	}
	if !strings.Contains(mlir, "return") {
		t.Errorf("missing return statement:\n%s", mlir)
	}
}

func TestEmitProgram_ValidMLIRStructure(t *testing.T) {
	// Verify the output has correct MLIR structural elements.
	ops := []ProgramOp{
		{
			OpName:      "Mul",
			InputSlots:  []int{0, 1},
			OutputSlot:  2,
			InputShapes: [][]int{{4, 4}, {4, 4}},
			OutputShape: []int{4, 4},
			Dtype:       DTypeF32,
		},
		{
			OpName:      "Exp",
			InputSlots:  []int{2},
			OutputSlot:  3,
			InputShapes: [][]int{{4, 4}},
			OutputShape: []int{4, 4},
			Dtype:       DTypeF32,
		},
	}

	mlir, err := EmitProgram(ops, []int{0, 1}, [][]int{{4, 4}, {4, 4}}, DTypeF32)
	if err != nil {
		t.Fatal(err)
	}

	// Count braces -- must be balanced.
	opens := strings.Count(mlir, "{")
	closes := strings.Count(mlir, "}")
	if opens != closes {
		t.Errorf("unbalanced braces: %d open vs %d close:\n%s", opens, closes, mlir)
	}

	// Must have exactly one module and one func.
	if strings.Count(mlir, "module {") != 1 {
		t.Errorf("expected exactly one module block:\n%s", mlir)
	}
	if strings.Count(mlir, "func.func @main") != 1 {
		t.Errorf("expected exactly one func.func @main:\n%s", mlir)
	}

	// Must have exactly one return.
	if strings.Count(mlir, "return ") != 1 {
		t.Errorf("expected exactly one return:\n%s", mlir)
	}

	// Lines should be indented.
	lines := strings.Split(mlir, "\n")
	for _, line := range lines {
		if strings.Contains(line, "stablehlo.") && !strings.HasPrefix(line, "    ") {
			t.Errorf("op line not indented with 4 spaces: %q", line)
		}
	}
}

func TestEmitProgram_ErrorCases(t *testing.T) {
	t.Run("no ops", func(t *testing.T) {
		_, err := EmitProgram(nil, nil, nil, DTypeF32)
		if err == nil {
			t.Fatal("expected error for empty ops")
		}
	})

	t.Run("mismatched input slots and shapes", func(t *testing.T) {
		ops := []ProgramOp{{
			OpName: "Add", InputSlots: []int{0, 1}, OutputSlot: 2,
			InputShapes: [][]int{{2}, {2}}, OutputShape: []int{2}, Dtype: DTypeF32,
		}}
		_, err := EmitProgram(ops, []int{0, 1, 2}, [][]int{{2}, {2}}, DTypeF32)
		if err == nil {
			t.Fatal("expected error for mismatched slots/shapes")
		}
	})

	t.Run("undefined slot reference", func(t *testing.T) {
		ops := []ProgramOp{{
			OpName: "Add", InputSlots: []int{0, 99}, OutputSlot: 2,
			InputShapes: [][]int{{2}, {2}}, OutputShape: []int{2}, Dtype: DTypeF32,
		}}
		_, err := EmitProgram(ops, []int{0}, [][]int{{2}}, DTypeF32)
		if err == nil {
			t.Fatal("expected error for undefined slot 99")
		}
		if !strings.Contains(err.Error(), "slot 99") {
			t.Errorf("error should mention slot 99: %v", err)
		}
	})
}

func TestEmitProgram_Reshape(t *testing.T) {
	ops := []ProgramOp{
		{
			OpName:      "Reshape",
			InputSlots:  []int{0},
			OutputSlot:  1,
			InputShapes: [][]int{{2, 6}},
			OutputShape: []int{3, 4},
			Dtype:       DTypeF32,
			Attrs:       map[string]any{"shape": []int{3, 4}},
		},
	}
	mlir, err := EmitProgram(ops, []int{0}, [][]int{{2, 6}}, DTypeF32)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(mlir, "stablehlo.reshape") {
		t.Errorf("missing reshape op:\n%s", mlir)
	}
}

func TestEmitProgram_ReduceSum(t *testing.T) {
	ops := []ProgramOp{
		{
			OpName:      "ReduceSum",
			InputSlots:  []int{0},
			OutputSlot:  1,
			InputShapes: [][]int{{2, 3}},
			OutputShape: []int{2},
			Dtype:       DTypeF32,
			Attrs:       map[string]any{"axis": 1},
		},
	}
	mlir, err := EmitProgram(ops, []int{0}, [][]int{{2, 3}}, DTypeF32)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(mlir, "stablehlo.reduce") {
		t.Errorf("missing reduce op:\n%s", mlir)
	}
	if !strings.Contains(mlir, "stablehlo.add") {
		t.Errorf("missing add body for ReduceSum:\n%s", mlir)
	}
}

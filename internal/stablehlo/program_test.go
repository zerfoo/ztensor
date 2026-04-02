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

func TestEmitKVCacheProgram_Prefill(t *testing.T) {
	// Simulate a simple prefill: matmul produces logits, a separate op produces KV cache.
	// Slots: 0=input_tokens, 1=weights, 2=kv_weights, 3=matmul_out(logits), 4=kv_out
	ops := []ProgramOp{
		{
			OpName:      "MatMul",
			InputSlots:  []int{0, 1},
			OutputSlot:  3,
			InputShapes: [][]int{{1, 2048}, {2048, 32000}},
			OutputShape: []int{1, 32000},
			Dtype:       DTypeF32,
		},
		{
			OpName:      "MatMul",
			InputSlots:  []int{0, 2},
			OutputSlot:  4,
			InputShapes: [][]int{{1, 2048}, {2048, 128}},
			OutputShape: []int{1, 128},
			Dtype:       DTypeF32,
		},
	}

	kvSlots := []KVCacheSlot{
		{InputSlot: 5, OutputSlot: 4, Shape: []int{32, 2048, 128}, SeqAxis: 1},
	}

	mlir, err := EmitKVCacheProgram(ops, []int{0, 1, 2}, [][]int{{1, 2048}, {2048, 32000}, {2048, 128}}, kvSlots, DTypeF32, false)
	if err != nil {
		t.Fatal(err)
	}

	// Should have tuple return type: (logits, kv_cache).
	if !strings.Contains(mlir, "-> (") {
		t.Errorf("expected tuple return type:\n%s", mlir)
	}

	// KV cache input should appear as a function argument.
	if !strings.Contains(mlir, "%arg3:") {
		t.Errorf("expected KV cache input arg (%%arg3):\n%s", mlir)
	}

	// Return should have two values.
	lines := strings.Split(mlir, "\n")
	var returnLine string
	for _, l := range lines {
		if strings.Contains(l, "return ") {
			returnLine = l
			break
		}
	}
	if returnLine == "" {
		t.Fatal("no return statement found")
	}
	// Should return two comma-separated values.
	returnParts := strings.SplitN(returnLine, ":", 2)
	if len(returnParts) < 2 {
		t.Fatalf("malformed return line: %s", returnLine)
	}
	returnValues := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(returnParts[0]), "return"))
	commaCount := strings.Count(returnValues, ",")
	if commaCount != 1 {
		t.Errorf("expected 2 return values (1 comma), got %d commas in %q", commaCount, returnValues)
	}
}

func TestEmitKVCacheProgram_Decode(t *testing.T) {
	// Decode program: single token + KV cache -> logits + updated KV cache.
	// Slots: 0=token, 1=weights, 2=matmul_out(logits), 3=kv_step
	ops := []ProgramOp{
		{
			OpName:      "MatMul",
			InputSlots:  []int{0, 1},
			OutputSlot:  2,
			InputShapes: [][]int{{1, 128}, {128, 32000}},
			OutputShape: []int{1, 32000},
			Dtype:       DTypeF32,
		},
		{
			OpName:      "MatMul",
			InputSlots:  []int{0, 1},
			OutputSlot:  3,
			InputShapes: [][]int{{1, 128}, {128, 128}},
			OutputShape: []int{1, 128},
			Dtype:       DTypeF32,
		},
	}

	kvSlots := []KVCacheSlot{
		{InputSlot: 10, OutputSlot: 3, Shape: []int{32, 64, 128}, SeqAxis: 1},
	}

	mlir, err := EmitKVCacheProgram(ops, []int{0, 1}, [][]int{{1, 128}, {128, 32000}}, kvSlots, DTypeF32, true)
	if err != nil {
		t.Fatal(err)
	}

	// Decode should emit a concatenate for KV cache.
	if !strings.Contains(mlir, "stablehlo.concatenate") {
		t.Errorf("decode program should contain stablehlo.concatenate:\n%s", mlir)
	}

	// KV cache arg should be present.
	if !strings.Contains(mlir, "%arg2: tensor<32x64x128xf32>") {
		t.Errorf("expected KV cache input arg with shape 32x64x128:\n%s", mlir)
	}

	// Return should have (logits, updated_kv) = 2 values.
	lines := strings.Split(mlir, "\n")
	var returnLine string
	for _, l := range lines {
		if strings.Contains(l, "return ") {
			returnLine = l
			break
		}
	}
	if returnLine == "" {
		t.Fatal("no return statement found")
	}
	returnParts := strings.SplitN(returnLine, ":", 2)
	if len(returnParts) < 2 {
		t.Fatalf("malformed return line: %s", returnLine)
	}
	returnValues := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(returnParts[0]), "return"))
	commaCount := strings.Count(returnValues, ",")
	if commaCount != 1 {
		t.Errorf("expected 2 return values (1 comma), got %d commas in %q", commaCount, returnValues)
	}

	// The updated KV shape should be seq_len+1 along the seq axis.
	if !strings.Contains(mlir, "tensor<32x65x128xf32>") {
		t.Errorf("expected updated KV cache shape 32x65x128 (seq_len 64+1):\n%s", mlir)
	}
}

func TestEmitKVCacheProgram_MultiLayer(t *testing.T) {
	// Two KV cache layers (like a 2-layer transformer).
	ops := []ProgramOp{
		{
			OpName:      "Add",
			InputSlots:  []int{0, 1},
			OutputSlot:  4,
			InputShapes: [][]int{{1, 64}, {1, 64}},
			OutputShape: []int{1, 64},
			Dtype:       DTypeF32,
		},
		{
			OpName:      "Add",
			InputSlots:  []int{0, 1},
			OutputSlot:  5,
			InputShapes: [][]int{{1, 64}, {1, 64}},
			OutputShape: []int{1, 64},
			Dtype:       DTypeF32,
		},
	}

	kvSlots := []KVCacheSlot{
		{InputSlot: 10, OutputSlot: 4, Shape: []int{8, 32, 64}, SeqAxis: 1},
		{InputSlot: 11, OutputSlot: 5, Shape: []int{8, 32, 64}, SeqAxis: 1},
	}

	mlir, err := EmitKVCacheProgram(ops, []int{0, 1}, [][]int{{1, 64}, {1, 64}}, kvSlots, DTypeF32, false)
	if err != nil {
		t.Fatal(err)
	}

	// Should have 4 function args: 2 regular + 2 KV cache.
	for _, arg := range []string{"%arg0:", "%arg1:", "%arg2:", "%arg3:"} {
		if !strings.Contains(mlir, arg) {
			t.Errorf("expected arg %s in signature:\n%s", arg, mlir)
		}
	}

	// Return should have 3 values: primary + 2 KV outputs.
	lines := strings.Split(mlir, "\n")
	var returnLine string
	for _, l := range lines {
		if strings.Contains(l, "return ") {
			returnLine = l
			break
		}
	}
	if returnLine == "" {
		t.Fatal("no return statement found")
	}
	returnParts := strings.SplitN(returnLine, ":", 2)
	returnValues := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(returnParts[0]), "return"))
	commaCount := strings.Count(returnValues, ",")
	if commaCount != 2 {
		t.Errorf("expected 3 return values (2 commas), got %d commas in %q", commaCount, returnValues)
	}
}

func TestEmitKVCacheProgram_ErrorCases(t *testing.T) {
	baseOps := []ProgramOp{
		{OpName: "Add", InputSlots: []int{0, 1}, OutputSlot: 2,
			InputShapes: [][]int{{2}, {2}}, OutputShape: []int{2}, Dtype: DTypeF32},
	}

	t.Run("no ops", func(t *testing.T) {
		_, err := EmitKVCacheProgram(nil, nil, nil, []KVCacheSlot{{InputSlot: 0, OutputSlot: 1, Shape: []int{2}}}, DTypeF32, false)
		if err == nil {
			t.Fatal("expected error for empty ops")
		}
	})

	t.Run("no kv slots", func(t *testing.T) {
		_, err := EmitKVCacheProgram(baseOps, []int{0, 1}, [][]int{{2}, {2}}, nil, DTypeF32, false)
		if err == nil {
			t.Fatal("expected error for nil KV slots")
		}
	})

	t.Run("mismatched input slots and shapes", func(t *testing.T) {
		_, err := EmitKVCacheProgram(baseOps, []int{0}, [][]int{{2}, {2}},
			[]KVCacheSlot{{InputSlot: 3, OutputSlot: 2, Shape: []int{2}}}, DTypeF32, false)
		if err == nil {
			t.Fatal("expected error for mismatched slots/shapes")
		}
	})

	t.Run("undefined kv output slot", func(t *testing.T) {
		_, err := EmitKVCacheProgram(baseOps, []int{0, 1}, [][]int{{2}, {2}},
			[]KVCacheSlot{{InputSlot: 3, OutputSlot: 99, Shape: []int{2}}}, DTypeF32, false)
		if err == nil {
			t.Fatal("expected error for undefined KV output slot")
		}
	})
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

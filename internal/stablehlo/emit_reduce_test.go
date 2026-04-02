package stablehlo

import (
	"strings"
	"testing"
)

func TestEmitReduceSum(t *testing.T) {
	namer := &SSANamer{}
	name, mlir := EmitReduceSum(namer, "%input", []int{2, 3}, 1, false, "f32")

	if name != "%v1" {
		t.Errorf("result name = %q, want %%v1", name)
	}
	if !strings.Contains(mlir, "stablehlo.reduce") {
		t.Error("missing stablehlo.reduce")
	}
	if !strings.Contains(mlir, "stablehlo.add") {
		t.Error("missing stablehlo.add in reduction body")
	}
	if !strings.Contains(mlir, "dimensions = array<i64: 1>") {
		t.Error("missing dimensions attribute")
	}
	if !strings.Contains(mlir, "tensor<2xf32>") {
		t.Errorf("missing output type tensor<2xf32> in:\n%s", mlir)
	}
}

func TestEmitReduceMax(t *testing.T) {
	namer := &SSANamer{}
	name, mlir := EmitReduceMax(namer, "%input", []int{4, 5}, 0, false, "f32")

	if name != "%v1" {
		t.Errorf("result name = %q, want %%v1", name)
	}
	if !strings.Contains(mlir, "stablehlo.maximum") {
		t.Error("missing stablehlo.maximum in reduction body")
	}
	if !strings.Contains(mlir, "0xFF800000") {
		t.Error("missing -inf init value for max reduction")
	}
	if !strings.Contains(mlir, "dimensions = array<i64: 0>") {
		t.Error("missing dimensions attribute")
	}
	if !strings.Contains(mlir, "tensor<5xf32>") {
		t.Errorf("missing output type tensor<5xf32> in:\n%s", mlir)
	}
}

func TestEmitReduceSumKeepDims(t *testing.T) {
	namer := &SSANamer{}
	name, mlir := EmitReduceSum(namer, "%input", []int{2, 3}, 1, true, "f32")

	if name != "%v2" {
		t.Errorf("result name = %q, want %%v2", name)
	}
	if !strings.Contains(mlir, "stablehlo.reshape") {
		t.Error("missing reshape for keepDims")
	}
	if !strings.Contains(mlir, "tensor<2x1xf32>") {
		t.Errorf("missing keepDims output type tensor<2x1xf32> in:\n%s", mlir)
	}
}

func TestEmitReduceMean(t *testing.T) {
	namer := &SSANamer{}
	name, mlir := EmitReduceMean(namer, "%input", []int{2, 6}, 1, false, "f32")

	if name != "%v3" {
		t.Errorf("result name = %q, want %%v3", name)
	}
	if !strings.Contains(mlir, "stablehlo.add") {
		t.Error("missing sum reduction for mean")
	}
	if !strings.Contains(mlir, "dense<6.0>") {
		t.Error("missing divisor constant for mean")
	}
	if !strings.Contains(mlir, "stablehlo.divide") {
		t.Error("missing divide for mean")
	}
}

func TestEmitSoftmax(t *testing.T) {
	namer := &SSANamer{}
	name, mlir := EmitSoftmax(namer, "%input", []int{2, 3}, 1, "f32")

	if name == "" {
		t.Fatal("result name is empty")
	}

	// Softmax should decompose into exactly 5 logical operations:
	// 1. ReduceMax  (reduce + reshape for keepDims)
	// 2. Subtract
	// 3. Exp
	// 4. ReduceSum  (reduce + reshape for keepDims)
	// 5. Divide
	ops := []struct {
		name string
		op   string
	}{
		{"ReduceMax", "stablehlo.maximum"},
		{"Subtract", "stablehlo.subtract"},
		{"Exp", "stablehlo.exponential"},
		{"ReduceSum", "stablehlo.add"},
		{"Divide", "stablehlo.divide"},
	}
	for _, op := range ops {
		if !strings.Contains(mlir, op.op) {
			t.Errorf("missing %s (%s) in Softmax decomposition", op.name, op.op)
		}
	}

	// Count the 5 high-level ops by counting the distinct operation types.
	highLevelOps := 0
	if strings.Contains(mlir, "stablehlo.maximum") {
		highLevelOps++
	}
	if strings.Contains(mlir, "stablehlo.subtract") {
		highLevelOps++
	}
	if strings.Contains(mlir, "stablehlo.exponential") {
		highLevelOps++
	}
	if strings.Contains(mlir, "stablehlo.add") {
		highLevelOps++
	}
	if strings.Contains(mlir, "stablehlo.divide") {
		highLevelOps++
	}
	if highLevelOps != 5 {
		t.Errorf("Softmax decomposition has %d high-level ops, want 5", highLevelOps)
	}
}

func TestEmitSoftmaxMLIRStructure(t *testing.T) {
	namer := &SSANamer{}
	_, mlir := EmitSoftmax(namer, "%input", []int{2, 3}, 1, "f32")

	// Verify the MLIR contains two reduce regions (one for max, one for sum).
	reduceCount := strings.Count(mlir, `"stablehlo.reduce"`)
	if reduceCount != 2 {
		t.Errorf("expected 2 stablehlo.reduce ops, got %d", reduceCount)
	}

	// Verify two reshape ops (keepDims for max and sum).
	reshapeCount := strings.Count(mlir, "stablehlo.reshape")
	if reshapeCount != 2 {
		t.Errorf("expected 2 reshape ops for keepDims, got %d", reshapeCount)
	}

	// Verify the output types contain the broadcast shapes.
	if !strings.Contains(mlir, "tensor<2x1xf32>") {
		t.Error("missing keepDims shape tensor<2x1xf32>")
	}
	if !strings.Contains(mlir, "tensor<2x3xf32>") {
		t.Error("missing full shape tensor<2x3xf32>")
	}
}

func TestEmitReduceRegionBody(t *testing.T) {
	namer := &SSANamer{}
	_, mlir := EmitReduceSum(namer, "%x", []int{3, 4}, 0, false, "f64")

	// Verify region body structure.
	if !strings.Contains(mlir, "^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>)") {
		t.Errorf("missing region block args in:\n%s", mlir)
	}
	if !strings.Contains(mlir, "stablehlo.return %0 : tensor<f64>") {
		t.Errorf("missing stablehlo.return in:\n%s", mlir)
	}
}

func TestEmitReduce3D(t *testing.T) {
	namer := &SSANamer{}
	_, mlir := EmitReduceSum(namer, "%t", []int{2, 3, 4}, 2, false, "f32")

	if !strings.Contains(mlir, "dimensions = array<i64: 2>") {
		t.Error("wrong dimension for axis=2")
	}
	if !strings.Contains(mlir, "tensor<2x3xf32>") {
		t.Errorf("wrong output shape, expected tensor<2x3xf32> in:\n%s", mlir)
	}
}

func TestReduceShape(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		axis     int
		keepDims bool
		want     []int
	}{
		{"remove dim", []int{2, 3}, 1, false, []int{2}},
		{"keep dim", []int{2, 3}, 1, true, []int{2, 1}},
		{"remove first", []int{4, 5}, 0, false, []int{5}},
		{"keep first", []int{4, 5}, 0, true, []int{1, 5}},
		{"3D middle", []int{2, 3, 4}, 1, false, []int{2, 4}},
		{"3D middle keep", []int{2, 3, 4}, 1, true, []int{2, 1, 4}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := reduceShape(tt.shape, tt.axis, tt.keepDims)
			if len(got) != len(tt.want) {
				t.Fatalf("reduceShape(%v, %d, %v) = %v, want %v", tt.shape, tt.axis, tt.keepDims, got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("reduceShape(%v, %d, %v)[%d] = %d, want %d", tt.shape, tt.axis, tt.keepDims, i, got[i], tt.want[i])
				}
			}
		})
	}
}

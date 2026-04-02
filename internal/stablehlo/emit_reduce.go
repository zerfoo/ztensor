package stablehlo

import (
	"fmt"
	"strings"
)

// EmitReduceSum emits a StableHLO reduce with an add body.
//
// The generated MLIR has the form:
//
//	%result = "stablehlo.reduce"(%input, %init) ({
//	^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
//	  %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
//	  stablehlo.return %0 : tensor<f32>
//	}) {dimensions = array<i64: axis>} : (inputType, tensor<dtype>) -> outputType
func EmitReduceSum(namer *SSANamer, input string, inputShape []int, axis int, keepDims bool, dtype string) (string, string) {
	return emitReduce(namer, input, inputShape, axis, keepDims, dtype, "add")
}

// EmitReduceMax emits a StableHLO reduce with a maximum body.
func EmitReduceMax(namer *SSANamer, input string, inputShape []int, axis int, keepDims bool, dtype string) (string, string) {
	return emitReduce(namer, input, inputShape, axis, keepDims, dtype, "maximum")
}

// EmitReduceMean emits a ReduceSum followed by a DivScalar to compute the mean.
// Returns the final result SSA name and the emitted MLIR text for both ops.
func EmitReduceMean(namer *SSANamer, input string, inputShape []int, axis int, keepDims bool, dtype string) (string, string) {
	sumName, sumMLIR := emitReduce(namer, input, inputShape, axis, keepDims, dtype, "add")

	// Compute the output shape of the reduction.
	outShape := reduceShape(inputShape, axis, keepDims)
	outType := FormatTensorType(outShape, dtype)

	// Emit a constant for the axis size and divide.
	count := inputShape[axis]
	constName := namer.NextName()
	divName := namer.NextName()

	var b strings.Builder
	b.WriteString(sumMLIR)
	fmt.Fprintf(&b, "%s = stablehlo.constant dense<%d.0> : tensor<%s>\n", constName, count, dtype)
	fmt.Fprintf(&b, "%s = stablehlo.divide %s, %s : %s\n", divName, sumName, constName, outType)
	return divName, b.String()
}

// EmitSoftmax decomposes Softmax into 5 StableHLO operations:
//  1. max = ReduceMax(input, axis, keepDims=true)
//  2. shifted = Sub(input, max)
//  3. exp = Exp(shifted)
//  4. sum = ReduceSum(exp, axis, keepDims=true)
//  5. result = Div(exp, sum)
//
// Returns the final result SSA name and the emitted MLIR text.
func EmitSoftmax(namer *SSANamer, input string, inputShape []int, axis int, dtype string) (string, string) {
	inputType := FormatTensorType(inputShape, dtype)

	var b strings.Builder

	// 1. max = ReduceMax(input, axis, keepDims=true)
	maxName, maxMLIR := EmitReduceMax(namer, input, inputShape, axis, true, dtype)
	b.WriteString(maxMLIR)

	// 2. shifted = Sub(input, max) -- broadcast max back to input shape
	shiftedName := namer.NextName()
	fmt.Fprintf(&b, "%s = stablehlo.subtract %s, %s : %s\n", shiftedName, input, maxName, inputType)

	// 3. exp = Exp(shifted)
	expName := namer.NextName()
	fmt.Fprintf(&b, "%s = stablehlo.exponential %s : %s\n", expName, shiftedName, inputType)

	// 4. sum = ReduceSum(exp, axis, keepDims=true)
	sumName, sumMLIR := EmitReduceSum(namer, expName, inputShape, axis, true, dtype)
	b.WriteString(sumMLIR)

	// 5. result = Div(exp, sum)
	resultName := namer.NextName()
	fmt.Fprintf(&b, "%s = stablehlo.divide %s, %s : %s\n", resultName, expName, sumName, inputType)

	return resultName, b.String()
}

// emitReduce generates a StableHLO reduce op with the given reduction body op.
// bodyOp should be "add" for sum or "maximum" for max.
func emitReduce(namer *SSANamer, input string, inputShape []int, axis int, keepDims bool, dtype, bodyOp string) (string, string) {
	scalarType := fmt.Sprintf("tensor<%s>", dtype)
	inputType := FormatTensorType(inputShape, dtype)
	outShape := reduceShape(inputShape, axis, false) // reduce always removes the dim
	outType := FormatTensorType(outShape, dtype)

	initName := namer.NextName()
	reduceName := namer.NextName()

	var b strings.Builder

	// Emit the init value (zero for add, -inf for maximum).
	initVal := "0.0"
	if bodyOp == "maximum" {
		initVal = "0xFF800000"
	}
	fmt.Fprintf(&b, "%s = stablehlo.constant dense<%s> : %s\n", initName, initVal, scalarType)

	// Emit the reduce op with inline region body.
	fmt.Fprintf(&b, "%s = \"stablehlo.reduce\"(%s, %s) ({\n", reduceName, input, initName)
	fmt.Fprintf(&b, "^bb0(%%arg0: %s, %%arg1: %s):\n", scalarType, scalarType)
	fmt.Fprintf(&b, "  %%0 = stablehlo.%s %%arg0, %%arg1 : %s\n", bodyOp, scalarType)
	fmt.Fprintf(&b, "  stablehlo.return %%0 : %s\n", scalarType)
	fmt.Fprintf(&b, "}) {dimensions = array<i64: %d>} : (%s, %s) -> %s\n", axis, inputType, scalarType, outType)

	// If keepDims, reshape to insert size-1 dimension at axis.
	if keepDims {
		keepShape := reduceShape(inputShape, axis, true)
		keepType := FormatTensorType(keepShape, dtype)
		reshapeName := namer.NextName()
		fmt.Fprintf(&b, "%s = stablehlo.reshape %s : %s -> %s\n", reshapeName, reduceName, outType, keepType)
		return reshapeName, b.String()
	}

	return reduceName, b.String()
}

// reduceShape computes the output shape after reducing along axis.
// If keepDims is true, the reduced axis becomes size 1.
// If keepDims is false, the reduced axis is removed.
func reduceShape(shape []int, axis int, keepDims bool) []int {
	if keepDims {
		out := make([]int, len(shape))
		copy(out, shape)
		out[axis] = 1
		return out
	}
	out := make([]int, 0, len(shape)-1)
	for i, d := range shape {
		if i != axis {
			out = append(out, d)
		}
	}
	return out
}

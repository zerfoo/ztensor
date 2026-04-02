// Package stablehlo provides StableHLO MLIR text emission for the PJRT backend.
package stablehlo

import (
	"fmt"
	"slices"
)

// InferShape computes the output shape for a given operation name, input shapes,
// and optional attributes. It returns an error if the shapes are incompatible.
func InferShape(opName string, inputShapes [][]int, attrs map[string]any) ([]int, error) {
	switch opName {
	// Element-wise binary ops with numpy-style broadcasting.
	case "Add", "Sub", "Mul", "Div":
		return inferBroadcastBinary(opName, inputShapes)

	// Scalar broadcast ops: output = shape of the non-scalar input.
	case "MulScalar", "DivScalar", "AddScalar":
		return inferScalarBroadcast(opName, inputShapes)

	// Unary ops: output shape = input shape.
	case "Exp", "Log", "Sin", "Cos", "Tanh", "Sqrt", "Rsqrt", "Neg", "Abs":
		return inferUnary(opName, inputShapes)

	// Binary same-shape ops: Pow takes two inputs of the same shape.
	case "Pow":
		return inferBroadcastBinary(opName, inputShapes)

	default:
		return nil, fmt.Errorf("stablehlo.InferShape: unsupported op %q", opName)
	}
}

// inferBroadcastBinary computes the broadcast output shape for two input shapes
// using numpy-style broadcasting: dimensions are aligned from the trailing end,
// and each pair must be equal or one of them must be 1.
func inferBroadcastBinary(opName string, inputShapes [][]int) ([]int, error) {
	if len(inputShapes) != 2 {
		return nil, fmt.Errorf("stablehlo.InferShape(%s): expected 2 input shapes, got %d", opName, len(inputShapes))
	}
	return broadcastShapes(opName, inputShapes[0], inputShapes[1])
}

// broadcastShapes returns the broadcast-compatible output shape for a and b,
// or an error if broadcasting is impossible.
func broadcastShapes(opName string, a, b []int) ([]int, error) {
	rank := max(len(a), len(b))
	out := make([]int, rank)

	for i := range rank {
		// Index from the trailing end.
		ai := len(a) - rank + i
		bi := len(b) - rank + i

		da := 1
		if ai >= 0 {
			da = a[ai]
		}
		db := 1
		if bi >= 0 {
			db = b[bi]
		}

		switch {
		case da == db:
			out[i] = da
		case da == 1:
			out[i] = db
		case db == 1:
			out[i] = da
		default:
			return nil, fmt.Errorf("stablehlo.InferShape(%s): incompatible shapes %v and %v at dimension %d (%d vs %d)",
				opName, a, b, i, da, db)
		}
	}
	return out, nil
}

// inferScalarBroadcast handles ops like MulScalar where one input is a scalar
// and the output takes the shape of the non-scalar (or first) input.
func inferScalarBroadcast(opName string, inputShapes [][]int) ([]int, error) {
	if len(inputShapes) < 1 || len(inputShapes) > 2 {
		return nil, fmt.Errorf("stablehlo.InferShape(%s): expected 1-2 input shapes, got %d", opName, len(inputShapes))
	}
	// The first input is the tensor; the second (if present) is the scalar.
	return slices.Clone(inputShapes[0]), nil
}

// inferUnary validates that exactly one input is provided and returns its shape.
func inferUnary(opName string, inputShapes [][]int) ([]int, error) {
	if len(inputShapes) != 1 {
		return nil, fmt.Errorf("stablehlo.InferShape(%s): expected 1 input shape, got %d", opName, len(inputShapes))
	}
	return slices.Clone(inputShapes[0]), nil
}

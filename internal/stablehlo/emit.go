package stablehlo

import "fmt"

// Emitter generates StableHLO MLIR text from operation inputs.
// Each emit method takes SSA input names, tensor shapes, and a dtype,
// and returns the emitted MLIR line(s) plus the output SSA name.
type Emitter struct {
	Namer *SSANamer
}

// NewEmitter creates an Emitter with a fresh SSANamer.
func NewEmitter() *Emitter {
	return &Emitter{Namer: &SSANamer{}}
}

// EmitBinaryElementwise emits a binary element-wise op (add, subtract, multiply, divide, power).
// Both inputs must have the same shape and dtype.
func (e *Emitter) EmitBinaryElementwise(opName, lhs, rhs string, shape []int, dtype string) (mlir, outName string) {
	outName = e.Namer.NextName()
	ty := FormatTensorType(shape, dtype)
	mlir = fmt.Sprintf("%s = %s %s, %s : %s", outName, opName, lhs, rhs, ty)
	return mlir, outName
}

// EmitAdd emits stablehlo.add.
func (e *Emitter) EmitAdd(lhs, rhs string, shape []int, dtype string) (string, string) {
	return e.EmitBinaryElementwise(OpAdd, lhs, rhs, shape, dtype)
}

// EmitSub emits stablehlo.subtract.
func (e *Emitter) EmitSub(lhs, rhs string, shape []int, dtype string) (string, string) {
	return e.EmitBinaryElementwise(OpSubtract, lhs, rhs, shape, dtype)
}

// EmitMul emits stablehlo.multiply.
func (e *Emitter) EmitMul(lhs, rhs string, shape []int, dtype string) (string, string) {
	return e.EmitBinaryElementwise(OpMultiply, lhs, rhs, shape, dtype)
}

// EmitDiv emits stablehlo.divide.
func (e *Emitter) EmitDiv(lhs, rhs string, shape []int, dtype string) (string, string) {
	return e.EmitBinaryElementwise(OpDivide, lhs, rhs, shape, dtype)
}

// EmitPow emits stablehlo.power.
func (e *Emitter) EmitPow(lhs, rhs string, shape []int, dtype string) (string, string) {
	return e.EmitBinaryElementwise(OpPower, lhs, rhs, shape, dtype)
}

// EmitUnary emits a unary element-wise op (exponential, log, sine, cosine, tanh, sqrt, rsqrt, negate).
func (e *Emitter) EmitUnary(opName, input string, shape []int, dtype string) (mlir, outName string) {
	outName = e.Namer.NextName()
	ty := FormatTensorType(shape, dtype)
	mlir = fmt.Sprintf("%s = %s %s : %s", outName, opName, input, ty)
	return mlir, outName
}

// EmitExp emits stablehlo.exponential.
func (e *Emitter) EmitExp(input string, shape []int, dtype string) (string, string) {
	return e.EmitUnary(OpExp, input, shape, dtype)
}

// EmitLog emits stablehlo.log.
func (e *Emitter) EmitLog(input string, shape []int, dtype string) (string, string) {
	return e.EmitUnary(OpLog, input, shape, dtype)
}

// EmitSin emits stablehlo.sine.
func (e *Emitter) EmitSin(input string, shape []int, dtype string) (string, string) {
	return e.EmitUnary(OpSin, input, shape, dtype)
}

// EmitCos emits stablehlo.cosine.
func (e *Emitter) EmitCos(input string, shape []int, dtype string) (string, string) {
	return e.EmitUnary(OpCos, input, shape, dtype)
}

// EmitTanh emits stablehlo.tanh.
func (e *Emitter) EmitTanh(input string, shape []int, dtype string) (string, string) {
	return e.EmitUnary(OpTanh, input, shape, dtype)
}

// EmitSqrt emits stablehlo.sqrt.
func (e *Emitter) EmitSqrt(input string, shape []int, dtype string) (string, string) {
	return e.EmitUnary(OpSqrt, input, shape, dtype)
}

// EmitRsqrt emits stablehlo.rsqrt.
func (e *Emitter) EmitRsqrt(input string, shape []int, dtype string) (string, string) {
	return e.EmitUnary(OpRsqrt, input, shape, dtype)
}

// EmitNeg emits stablehlo.negate.
func (e *Emitter) EmitNeg(input string, shape []int, dtype string) (string, string) {
	return e.EmitUnary(OpNegate, input, shape, dtype)
}

// EmitScalarOp emits a scalar operation as three MLIR instructions:
//  1. stablehlo.constant for the scalar value
//  2. stablehlo.broadcast_in_dim to broadcast to the tensor shape
//  3. The element-wise binary op
//
// Returns all three lines (newline-separated) and the final output SSA name.
func (e *Emitter) EmitScalarOp(elemOp, input string, scalar float64, shape []int, dtype string) (mlir, outName string) {
	ty := FormatTensorType(shape, dtype)
	scalarTy := FormatTensorType(nil, dtype)

	// 1. Constant
	constName := e.Namer.NextName()
	constLine := fmt.Sprintf("%s = %s dense<%v> : %s", constName, OpConstant, scalar, scalarTy)

	// 2. Broadcast
	bcastName := e.Namer.NextName()
	bcastLine := fmt.Sprintf("%s = %s %s, dims = [] : (%s) -> %s", bcastName, OpBroadcastIn, constName, scalarTy, ty)

	// 3. Element-wise op
	outName = e.Namer.NextName()
	opLine := fmt.Sprintf("%s = %s %s, %s : %s", outName, elemOp, input, bcastName, ty)

	mlir = constLine + "\n" + bcastLine + "\n" + opLine
	return mlir, outName
}

// EmitMulScalar emits stablehlo.constant + broadcast_in_dim + multiply.
func (e *Emitter) EmitMulScalar(input string, scalar float64, shape []int, dtype string) (string, string) {
	return e.EmitScalarOp(OpMultiply, input, scalar, shape, dtype)
}

// EmitAddScalar emits stablehlo.constant + broadcast_in_dim + add.
func (e *Emitter) EmitAddScalar(input string, scalar float64, shape []int, dtype string) (string, string) {
	return e.EmitScalarOp(OpAdd, input, scalar, shape, dtype)
}

// EmitDivScalar emits stablehlo.constant + broadcast_in_dim + divide.
func (e *Emitter) EmitDivScalar(input string, scalar float64, shape []int, dtype string) (string, string) {
	return e.EmitScalarOp(OpDivide, input, scalar, shape, dtype)
}

// EmitOp dispatches to the appropriate emit function based on the engine op name.
// For binary ops, inputs should be [lhs, rhs]. For unary ops, inputs should be [input].
// For scalar ops, inputs should be [input] and attrs must contain "scalar" (float64).
// Returns the emitted MLIR text and the output SSA name.
func (e *Emitter) EmitOp(opName string, inputs []string, shape []int, dtype string, attrs map[string]any) (string, string, error) {
	switch opName {
	case "Add":
		if len(inputs) != 2 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 2 inputs, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitAdd(inputs[0], inputs[1], shape, dtype)
		return mlir, out, nil
	case "Sub":
		if len(inputs) != 2 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 2 inputs, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitSub(inputs[0], inputs[1], shape, dtype)
		return mlir, out, nil
	case "Mul":
		if len(inputs) != 2 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 2 inputs, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitMul(inputs[0], inputs[1], shape, dtype)
		return mlir, out, nil
	case "Div":
		if len(inputs) != 2 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 2 inputs, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitDiv(inputs[0], inputs[1], shape, dtype)
		return mlir, out, nil
	case "Pow":
		if len(inputs) != 2 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 2 inputs, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitPow(inputs[0], inputs[1], shape, dtype)
		return mlir, out, nil
	case "Exp":
		if len(inputs) != 1 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 1 input, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitExp(inputs[0], shape, dtype)
		return mlir, out, nil
	case "Log":
		if len(inputs) != 1 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 1 input, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitLog(inputs[0], shape, dtype)
		return mlir, out, nil
	case "Sin":
		if len(inputs) != 1 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 1 input, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitSin(inputs[0], shape, dtype)
		return mlir, out, nil
	case "Cos":
		if len(inputs) != 1 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 1 input, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitCos(inputs[0], shape, dtype)
		return mlir, out, nil
	case "Tanh":
		if len(inputs) != 1 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 1 input, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitTanh(inputs[0], shape, dtype)
		return mlir, out, nil
	case "Sqrt":
		if len(inputs) != 1 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 1 input, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitSqrt(inputs[0], shape, dtype)
		return mlir, out, nil
	case "Rsqrt":
		if len(inputs) != 1 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 1 input, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitRsqrt(inputs[0], shape, dtype)
		return mlir, out, nil
	case "Neg":
		if len(inputs) != 1 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 1 input, got %d", opName, len(inputs))
		}
		mlir, out := e.EmitNeg(inputs[0], shape, dtype)
		return mlir, out, nil
	case "MulScalar":
		if len(inputs) != 1 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 1 input, got %d", opName, len(inputs))
		}
		s, err := scalarAttr(opName, attrs)
		if err != nil {
			return "", "", err
		}
		mlir, out := e.EmitMulScalar(inputs[0], s, shape, dtype)
		return mlir, out, nil
	case "AddScalar":
		if len(inputs) != 1 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 1 input, got %d", opName, len(inputs))
		}
		s, err := scalarAttr(opName, attrs)
		if err != nil {
			return "", "", err
		}
		mlir, out := e.EmitAddScalar(inputs[0], s, shape, dtype)
		return mlir, out, nil
	case "DivScalar":
		if len(inputs) != 1 {
			return "", "", fmt.Errorf("EmitOp(%s): expected 1 input, got %d", opName, len(inputs))
		}
		s, err := scalarAttr(opName, attrs)
		if err != nil {
			return "", "", err
		}
		mlir, out := e.EmitDivScalar(inputs[0], s, shape, dtype)
		return mlir, out, nil
	default:
		return "", "", fmt.Errorf("EmitOp: unsupported op %q", opName)
	}
}

func scalarAttr(opName string, attrs map[string]any) (float64, error) {
	if attrs == nil {
		return 0, fmt.Errorf("EmitOp(%s): attrs map is nil, need \"scalar\" key", opName)
	}
	v, ok := attrs["scalar"]
	if !ok {
		return 0, fmt.Errorf("EmitOp(%s): missing \"scalar\" in attrs", opName)
	}
	s, ok := v.(float64)
	if !ok {
		return 0, fmt.Errorf("EmitOp(%s): \"scalar\" attr is %T, want float64", opName, v)
	}
	return s, nil
}

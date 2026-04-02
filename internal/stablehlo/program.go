package stablehlo

import (
	"fmt"
	"strings"
)

// ProgramOp describes a single operation in a StableHLO program.
type ProgramOp struct {
	OpName      string         // Engine method name (e.g., "Add", "MatMul", "Softmax")
	InputSlots  []int          // indices into the slot table for inputs
	OutputSlot  int            // index for the output
	InputShapes [][]int        // shapes of each input
	OutputShape []int          // shape of the output
	Dtype       string         // "f32", "f16", etc.
	Attrs       map[string]any // op-specific attributes
}

// EmitProgram takes a sequence of operations and produces a complete StableHLO
// MLIR module. inputSlots identifies which slots are function arguments, and
// inputShapes provides their shapes. The last operation's output slot is used
// as the function return value.
//
// Slot indices map to SSA names: function arguments get %arg0, %arg1, etc.
// Intermediate results get %v0, %v1, etc. via the Emitter's SSANamer.
func EmitProgram(ops []ProgramOp, inputSlots []int, inputShapes [][]int, dtype string) (string, error) {
	if len(ops) == 0 {
		return "", fmt.Errorf("EmitProgram: no operations provided")
	}
	if len(inputSlots) != len(inputShapes) {
		return "", fmt.Errorf("EmitProgram: inputSlots length %d != inputShapes length %d", len(inputSlots), len(inputShapes))
	}

	// Build slot table: slot index -> SSA name.
	slotTable := make(map[int]string)
	for i, slot := range inputSlots {
		slotTable[slot] = fmt.Sprintf("%%arg%d", i)
	}

	// Build function signature.
	var argDecls []string
	for i, shape := range inputShapes {
		ty := FormatTensorType(shape, dtype)
		argDecls = append(argDecls, fmt.Sprintf("%%arg%d: %s", i, ty))
	}

	// The return type is the output shape of the last op.
	lastOp := ops[len(ops)-1]
	returnType := FormatTensorType(lastOp.OutputShape, lastOp.Dtype)

	emitter := NewEmitter()
	var bodyLines []string

	for i, op := range ops {
		// Resolve input SSA names from slot table.
		inputNames := make([]string, len(op.InputSlots))
		for j, slot := range op.InputSlots {
			name, ok := slotTable[slot]
			if !ok {
				return "", fmt.Errorf("EmitProgram: op %d (%s) references undefined slot %d", i, op.OpName, slot)
			}
			inputNames[j] = name
		}

		mlir, outName, err := dispatchProgramOp(emitter, op, inputNames)
		if err != nil {
			return "", fmt.Errorf("EmitProgram: op %d (%s): %w", i, op.OpName, err)
		}

		slotTable[op.OutputSlot] = outName
		bodyLines = append(bodyLines, mlir)
	}

	// Resolve the return value SSA name.
	returnName, ok := slotTable[lastOp.OutputSlot]
	if !ok {
		return "", fmt.Errorf("EmitProgram: return slot %d not found", lastOp.OutputSlot)
	}

	// Build the module.
	var b strings.Builder
	b.WriteString("module {\n")
	fmt.Fprintf(&b, "  func.func @main(%s) -> %s {\n", strings.Join(argDecls, ", "), returnType)
	for _, line := range bodyLines {
		// Each MLIR line may contain newlines (e.g., reduce ops emit multi-line blocks).
		for _, subLine := range strings.Split(line, "\n") {
			if subLine == "" {
				continue
			}
			fmt.Fprintf(&b, "    %s\n", subLine)
		}
	}
	fmt.Fprintf(&b, "    return %s : %s\n", returnName, returnType)
	b.WriteString("  }\n")
	b.WriteString("}")

	return b.String(), nil
}

// dispatchProgramOp dispatches a ProgramOp to the appropriate emitter.
// It handles element-wise ops via Emitter.EmitOp and structural/reduce ops
// via their dedicated emitters in emit_structural.go and emit_reduce.go.
func dispatchProgramOp(e *Emitter, op ProgramOp, inputNames []string) (string, string, error) {
	switch op.OpName {
	case "MatMul":
		if len(inputNames) != 2 || len(op.InputShapes) != 2 {
			return "", "", fmt.Errorf("MatMul requires 2 inputs, got %d names and %d shapes", len(inputNames), len(op.InputShapes))
		}
		return EmitMatMul(e.Namer, inputNames[0], inputNames[1], op.InputShapes[0], op.InputShapes[1], op.Dtype)

	case "Transpose":
		if len(inputNames) != 1 || len(op.InputShapes) != 1 {
			return "", "", fmt.Errorf("Transpose requires 1 input, got %d names and %d shapes", len(inputNames), len(op.InputShapes))
		}
		perm, err := intSliceAttr("perm", op.Attrs)
		if err != nil {
			return "", "", err
		}
		return EmitTranspose(e.Namer, inputNames[0], op.InputShapes[0], perm, op.Dtype)

	case "Reshape":
		if len(inputNames) != 1 || len(op.InputShapes) != 1 {
			return "", "", fmt.Errorf("Reshape requires 1 input, got %d names and %d shapes", len(inputNames), len(op.InputShapes))
		}
		targetShape, err := intSliceAttr("shape", op.Attrs)
		if err != nil {
			return "", "", err
		}
		return EmitReshape(e.Namer, inputNames[0], op.InputShapes[0], targetShape, op.Dtype)

	case "ReduceSum":
		if len(inputNames) != 1 || len(op.InputShapes) != 1 {
			return "", "", fmt.Errorf("ReduceSum requires 1 input, got %d names and %d shapes", len(inputNames), len(op.InputShapes))
		}
		axis, err := intAttr("axis", op.Attrs)
		if err != nil {
			return "", "", err
		}
		outName, mlir := EmitReduceSum(e.Namer, inputNames[0], op.InputShapes[0], axis, false, op.Dtype)
		return mlir, outName, nil

	case "ReduceMax":
		if len(inputNames) != 1 || len(op.InputShapes) != 1 {
			return "", "", fmt.Errorf("ReduceMax requires 1 input, got %d names and %d shapes", len(inputNames), len(op.InputShapes))
		}
		axis, err := intAttr("axis", op.Attrs)
		if err != nil {
			return "", "", err
		}
		outName, mlir := EmitReduceMax(e.Namer, inputNames[0], op.InputShapes[0], axis, false, op.Dtype)
		return mlir, outName, nil

	case "Softmax":
		if len(inputNames) != 1 || len(op.InputShapes) != 1 {
			return "", "", fmt.Errorf("Softmax requires 1 input, got %d names and %d shapes", len(inputNames), len(op.InputShapes))
		}
		axis := len(op.InputShapes[0]) - 1 // default: last axis
		if op.Attrs != nil {
			if a, ok := op.Attrs["axis"]; ok {
				if ai, ok := a.(int); ok {
					axis = ai
				}
			}
		}
		outName, mlir := EmitSoftmax(e.Namer, inputNames[0], op.InputShapes[0], axis, op.Dtype)
		return mlir, outName, nil

	default:
		// Delegate to the element-wise EmitOp dispatcher.
		return e.EmitOp(op.OpName, inputNames, op.OutputShape, op.Dtype, op.Attrs)
	}
}

// intSliceAttr extracts an []int attribute by key from an attrs map.
func intSliceAttr(key string, attrs map[string]any) ([]int, error) {
	if attrs == nil {
		return nil, fmt.Errorf("missing attr %q: attrs map is nil", key)
	}
	v, ok := attrs[key]
	if !ok {
		return nil, fmt.Errorf("missing attr %q", key)
	}
	s, ok := v.([]int)
	if !ok {
		return nil, fmt.Errorf("attr %q must be []int, got %T", key, v)
	}
	return s, nil
}

// intAttr extracts an int attribute by key from an attrs map.
func intAttr(key string, attrs map[string]any) (int, error) {
	if attrs == nil {
		return 0, fmt.Errorf("missing attr %q: attrs map is nil", key)
	}
	v, ok := attrs[key]
	if !ok {
		return 0, fmt.Errorf("missing attr %q", key)
	}
	i, ok := v.(int)
	if !ok {
		return 0, fmt.Errorf("attr %q must be int, got %T", key, v)
	}
	return i, nil
}

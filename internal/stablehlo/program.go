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

// KVCacheSlot describes a stateful KV cache slot that must be rewritten as
// explicit function I/O for PJRT's pure-functional execution model.
//
// In the original graph, the KV cache is fed back via StatefulInputNode.
// For PJRT, each KV cache tensor becomes both a function argument (the
// previous state) and a return value (the updated state).
type KVCacheSlot struct {
	InputSlot  int   // slot index where the KV cache is read (becomes a function arg)
	OutputSlot int   // slot index where the updated KV cache is produced (becomes a return value)
	Shape      []int // tensor shape (e.g., [num_heads, seq_len, head_dim])
	SeqAxis    int   // axis along which decode concatenation occurs
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

// EmitKVCacheProgram emits a StableHLO program with explicit KV cache I/O.
//
// KV cache tensors are added as both function arguments and return values.
// The function signature becomes:
//
//	func.func @main(%regular_args..., %kv_in_0, %kv_in_1, ...) ->
//	    (regular_output, %kv_out_0, %kv_out_1, ...)
//
// For decode programs (decode=true), each KV cache output is produced by
// concatenating the KV input with the new KV step along the sequence axis:
//
//	kv_out = concat(kv_in, kv_step, axis=seq_axis)
//
// For prefill programs (decode=false), KV cache outputs are passed through
// directly from the ops that produce them.
func EmitKVCacheProgram(ops []ProgramOp, inputSlots []int, inputShapes [][]int, kvSlots []KVCacheSlot, dtype string, decode bool) (string, error) {
	if len(ops) == 0 {
		return "", fmt.Errorf("EmitKVCacheProgram: no operations provided")
	}
	if len(inputSlots) != len(inputShapes) {
		return "", fmt.Errorf("EmitKVCacheProgram: inputSlots length %d != inputShapes length %d", len(inputSlots), len(inputShapes))
	}
	if len(kvSlots) == 0 {
		return "", fmt.Errorf("EmitKVCacheProgram: no KV cache slots provided")
	}

	// Build slot table: slot index -> SSA name.
	slotTable := make(map[int]string)
	argIdx := 0
	for i, slot := range inputSlots {
		_ = i
		slotTable[slot] = fmt.Sprintf("%%arg%d", argIdx)
		argIdx++
	}

	// Add KV cache input slots as additional function arguments.
	kvArgStart := argIdx
	for i, kv := range kvSlots {
		slotTable[kv.InputSlot] = fmt.Sprintf("%%arg%d", kvArgStart+i)
		argIdx++
	}

	// Build function argument declarations.
	var argDecls []string
	for i, shape := range inputShapes {
		ty := FormatTensorType(shape, dtype)
		argDecls = append(argDecls, fmt.Sprintf("%%arg%d: %s", i, ty))
	}
	for i, kv := range kvSlots {
		ty := FormatTensorType(kv.Shape, dtype)
		argDecls = append(argDecls, fmt.Sprintf("%%arg%d: %s", kvArgStart+i, ty))
	}

	// Emit all ops.
	emitter := NewEmitter()
	var bodyLines []string

	for i, op := range ops {
		inputNames := make([]string, len(op.InputSlots))
		for j, slot := range op.InputSlots {
			name, ok := slotTable[slot]
			if !ok {
				return "", fmt.Errorf("EmitKVCacheProgram: op %d (%s) references undefined slot %d", i, op.OpName, slot)
			}
			inputNames[j] = name
		}

		mlir, outName, err := dispatchProgramOp(emitter, op, inputNames)
		if err != nil {
			return "", fmt.Errorf("EmitKVCacheProgram: op %d (%s): %w", i, op.OpName, err)
		}

		slotTable[op.OutputSlot] = outName
		bodyLines = append(bodyLines, mlir)
	}

	// Build return values: primary output + KV cache outputs.
	lastOp := ops[len(ops)-1]
	primaryReturnName, ok := slotTable[lastOp.OutputSlot]
	if !ok {
		return "", fmt.Errorf("EmitKVCacheProgram: return slot %d not found", lastOp.OutputSlot)
	}
	primaryReturnType := FormatTensorType(lastOp.OutputShape, lastOp.Dtype)

	var returnNames []string
	var returnTypes []string
	returnNames = append(returnNames, primaryReturnName)
	returnTypes = append(returnTypes, primaryReturnType)

	// For each KV slot, resolve its output and optionally emit concat.
	for i, kv := range kvSlots {
		kvOutName, ok := slotTable[kv.OutputSlot]
		if !ok {
			return "", fmt.Errorf("EmitKVCacheProgram: KV output slot %d not found", kv.OutputSlot)
		}

		if decode {
			// Decode: concat(kv_in, kv_step, axis=seq_axis).
			kvInName := fmt.Sprintf("%%arg%d", kvArgStart+i)

			// Compute the concat output shape: kv_in.Shape with seq_axis doubled
			// (kv_in has full seq_len, kv_step has 1 step).
			kvStepShape := make([]int, len(kv.Shape))
			copy(kvStepShape, kv.Shape)
			kvStepShape[kv.SeqAxis] = 1 // the new step is a single position

			concatOutShape := make([]int, len(kv.Shape))
			copy(concatOutShape, kv.Shape)
			concatOutShape[kv.SeqAxis] = kv.Shape[kv.SeqAxis] + 1

			mlir, concatName, err := EmitConcat(
				emitter.Namer,
				[]string{kvInName, kvOutName},
				[][]int{kv.Shape, kvStepShape},
				kv.SeqAxis,
				dtype,
			)
			if err != nil {
				return "", fmt.Errorf("EmitKVCacheProgram: KV concat for slot %d: %w", i, err)
			}
			bodyLines = append(bodyLines, mlir)
			kvOutName = concatName
			returnTypes = append(returnTypes, FormatTensorType(concatOutShape, dtype))
		} else {
			// Prefill: pass through the KV output directly.
			// Find the shape from the op that produces this slot.
			kvOutShape := findSlotShape(ops, kv.OutputSlot)
			if kvOutShape == nil {
				return "", fmt.Errorf("EmitKVCacheProgram: cannot determine shape for KV output slot %d", kv.OutputSlot)
			}
			returnTypes = append(returnTypes, FormatTensorType(kvOutShape, dtype))
		}
		returnNames = append(returnNames, kvOutName)
	}

	// Build the module.
	returnTypeStr := "(" + strings.Join(returnTypes, ", ") + ")"
	returnValueStr := strings.Join(returnNames, ", ")
	returnAnnotation := strings.Join(returnTypes, ", ")

	var b strings.Builder
	b.WriteString("module {\n")
	fmt.Fprintf(&b, "  func.func @main(%s) -> %s {\n", strings.Join(argDecls, ", "), returnTypeStr)
	for _, line := range bodyLines {
		for _, subLine := range strings.Split(line, "\n") {
			if subLine == "" {
				continue
			}
			fmt.Fprintf(&b, "    %s\n", subLine)
		}
	}
	fmt.Fprintf(&b, "    return %s : %s\n", returnValueStr, returnAnnotation)
	b.WriteString("  }\n")
	b.WriteString("}")

	return b.String(), nil
}

// findSlotShape finds the output shape of the op that produces a given slot.
func findSlotShape(ops []ProgramOp, slot int) []int {
	for _, op := range ops {
		if op.OutputSlot == slot {
			return op.OutputShape
		}
	}
	return nil
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

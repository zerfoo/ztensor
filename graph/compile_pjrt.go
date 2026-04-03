package graph

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/internal/pjrt"
	"github.com/zerfoo/ztensor/internal/stablehlo"
	"github.com/zerfoo/ztensor/tensor"
)

// CompilePJRT compiles the graph into a PJRTPlan for execution on a PJRT
// backend. It traces the graph to obtain primitive operations, emits
// StableHLO MLIR, compiles it via the PJRT client, and transfers frozen
// weights to the device.
//
// For graphs with KV cache (StatefulInputNodes), both a prefill and a decode
// executable are compiled. For graphs without KV cache, only the prefill
// executable is produced.
func (g *Graph[T]) CompilePJRT(ctx context.Context, client *pjrt.Client, inputs ...*tensor.TensorNumeric[T]) (*PJRTPlan[T], error) {
	if client == nil {
		return nil, fmt.Errorf("CompilePJRT: client is nil")
	}

	// Step 1: Trace the graph to get primitive ops and slot metadata.
	plan, err := g.CompileTraced(ctx, inputs...)
	if err != nil {
		return nil, fmt.Errorf("CompilePJRT: trace failed: %w", err)
	}

	tracedMetas := plan.Instructions()
	slotShapes := plan.SlotShapes()
	inputSlots := plan.InputSlots()
	outputSlot := plan.OutputSlot()
	frozenSlots := plan.FrozenSlots()

	// Build a set of frozen slot indices for quick lookup.
	frozenSet := make(map[int]bool, len(frozenSlots))
	for _, fs := range frozenSlots {
		frozenSet[fs.SlotIdx] = true
	}

	// Determine the MLIR dtype from the Go generic type.
	dtype, err := mlirDtype[T]()
	if err != nil {
		return nil, fmt.Errorf("CompilePJRT: %w", err)
	}

	// Step 2: Convert traced InstructionMeta to stablehlo.ProgramOp.
	programOps := make([]stablehlo.ProgramOp, len(tracedMetas))
	for i, meta := range tracedMetas {
		inputShapes := make([][]int, len(meta.InputIdx))
		for j, idx := range meta.InputIdx {
			if idx < len(slotShapes) && slotShapes[idx] != nil {
				inputShapes[j] = slotShapes[idx]
			}
		}

		var outputShape []int
		if meta.OutputIdx < len(slotShapes) {
			outputShape = slotShapes[meta.OutputIdx]
		}

		programOps[i] = stablehlo.ProgramOp{
			OpName:      meta.OpName,
			InputSlots:  meta.InputIdx,
			OutputSlot:  meta.OutputIdx,
			InputShapes: inputShapes,
			OutputShape: outputShape,
			Dtype:       dtype,
			Attrs:       meta.ExtraArgs,
		}
	}

	// Collect input shapes for the StableHLO function signature.
	// This includes both dynamic inputs and frozen (weight) slots.
	allInputSlots := make([]int, 0, len(inputSlots)+len(frozenSlots))
	allInputShapes := make([][]int, 0, len(inputSlots)+len(frozenSlots))

	for _, idx := range inputSlots {
		allInputSlots = append(allInputSlots, idx)
		if idx < len(slotShapes) {
			allInputShapes = append(allInputShapes, slotShapes[idx])
		} else {
			allInputShapes = append(allInputShapes, nil)
		}
	}
	for _, fs := range frozenSlots {
		allInputSlots = append(allInputSlots, fs.SlotIdx)
		if fs.SlotIdx < len(slotShapes) {
			allInputShapes = append(allInputShapes, slotShapes[fs.SlotIdx])
		} else {
			allInputShapes = append(allInputShapes, nil)
		}
	}

	// Step 3: Identify KV cache slots from graph's kvPairs.
	kvPairs := g.KVPairs()
	var kvSlots []stablehlo.KVCacheSlot

	if len(kvPairs) > 0 {
		// We need to map the KV pair nodes to their traced slot indices.
		// The graph ran a Forward during CompileTraced, so memo has the
		// output tensors. We need the slot indices from the plan.
		for _, kv := range kvPairs {
			inputShape := kv.Input.OutputShape()
			outputShape := kv.Output.OutputShape()

			// Find the slot index for the input node's output. The input
			// node is a StatefulInputNode — its tensor was registered as
			// a slot during tracing. We search plan's slot shapes.
			inputSlot := findNodeSlot(inputSlots, slotShapes, inputShape)
			outputSlot := findOutputSlot(programOps, outputShape)

			// Default seq_axis to 1 (standard KV cache layout:
			// [num_heads, seq_len, head_dim]).
			seqAxis := 1
			if len(inputShape) >= 2 {
				seqAxis = len(inputShape) - 2
			}

			kvSlots = append(kvSlots, stablehlo.KVCacheSlot{
				InputSlot:  inputSlot,
				OutputSlot: outputSlot,
				Shape:      inputShape,
				SeqAxis:    seqAxis,
			})
		}
	}

	// Step 4: Emit StableHLO MLIR and compile.
	var prefillMLIR string
	var decodeMLIR string

	if len(kvSlots) > 0 {
		prefillMLIR, err = stablehlo.EmitKVCacheProgram(programOps, allInputSlots, allInputShapes, kvSlots, dtype, false)
		if err != nil {
			return nil, fmt.Errorf("CompilePJRT: emit prefill program: %w", err)
		}
		decodeMLIR, err = stablehlo.EmitKVCacheProgram(programOps, allInputSlots, allInputShapes, kvSlots, dtype, true)
		if err != nil {
			return nil, fmt.Errorf("CompilePJRT: emit decode program: %w", err)
		}
	} else {
		prefillMLIR, err = stablehlo.EmitProgram(programOps, allInputSlots, allInputShapes, dtype)
		if err != nil {
			return nil, fmt.Errorf("CompilePJRT: emit program: %w", err)
		}
	}

	// Step 5: Compile the MLIR programs via PJRT.
	prefillExec, err := client.Compile(prefillMLIR)
	if err != nil {
		return nil, fmt.Errorf("CompilePJRT: compile prefill: %w", err)
	}

	var decodeExec *pjrt.LoadedExecutable
	if decodeMLIR != "" {
		decodeExec, err = client.Compile(decodeMLIR)
		if err != nil {
			prefillExec.Close()
			return nil, fmt.Errorf("CompilePJRT: compile decode: %w", err)
		}
	}

	// Step 6: Transfer frozen weights to device.
	devices, err := client.AddressableDevices()
	if err != nil {
		prefillExec.Close()
		if decodeExec != nil {
			decodeExec.Close()
		}
		return nil, fmt.Errorf("CompilePJRT: get devices: %w", err)
	}
	if len(devices) == 0 {
		prefillExec.Close()
		if decodeExec != nil {
			decodeExec.Close()
		}
		return nil, fmt.Errorf("CompilePJRT: no addressable devices")
	}
	device := devices[0]

	weightBuffers := make([]*pjrt.Buffer, len(frozenSlots))
	for i, fs := range frozenSlots {
		data := fs.Data.Data()
		shape := fs.Data.Shape()
		buf, err := pjrt.BufferFromHost(client, data, shape, device)
		if err != nil {
			// Clean up already-transferred buffers.
			for j := 0; j < i; j++ {
				weightBuffers[j].Close()
			}
			prefillExec.Close()
			if decodeExec != nil {
				decodeExec.Close()
			}
			return nil, fmt.Errorf("CompilePJRT: transfer weight slot %d: %w", fs.SlotIdx, err)
		}
		weightBuffers[i] = buf
	}

	// Build the slot shape map.
	slotShapeMap := make(map[int][]int, len(slotShapes))
	for i, s := range slotShapes {
		if s != nil {
			slotShapeMap[i] = s
		}
	}

	return &PJRTPlan[T]{
		PrefillExec:   prefillExec,
		DecodeExec:    decodeExec,
		Client:        client,
		WeightBuffers: weightBuffers,
		KVSlots:       kvSlots,
		InputSlots:    inputSlots,
		OutputSlot:    outputSlot,
		SlotShapes:    slotShapeMap,
		Dtype:         dtype,
		FrozenSlots:   frozenSlotIdxs(frozenSlots),
	}, nil
}

// mlirDtype resolves the MLIR dtype string for the generic type T.
func mlirDtype[T tensor.Numeric]() (string, error) {
	var zero T
	goType := fmt.Sprintf("%T", zero)
	// Handle package-qualified type names.
	switch goType {
	case "float16.Float16":
		goType = "float16"
	case "float16.BFloat16":
		goType = "bfloat16"
	case "float8.Float8":
		goType = "float8"
	}
	dtype, ok := stablehlo.GoDTypeToMLIR(goType)
	if !ok {
		return "", fmt.Errorf("unsupported type %T for StableHLO", zero)
	}
	return dtype, nil
}

// findNodeSlot finds the slot index for a node by matching its shape against
// known input slot shapes. This is a heuristic — in practice, KV cache input
// nodes have unique shapes within the input set.
func findNodeSlot(inputSlots []int, slotShapes [][]int, targetShape []int) int {
	for _, idx := range inputSlots {
		if idx < len(slotShapes) && shapesEqual(slotShapes[idx], targetShape) {
			return idx
		}
	}
	// Fallback: return -1 to indicate not found.
	return -1
}

// findOutputSlot finds the last op whose output shape matches the target.
func findOutputSlot(ops []stablehlo.ProgramOp, targetShape []int) int {
	for i := len(ops) - 1; i >= 0; i-- {
		if shapesEqual(ops[i].OutputShape, targetShape) {
			return ops[i].OutputSlot
		}
	}
	return -1
}

// shapesEqual compares two shapes for equality.
func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// frozenSlotIdxs extracts slot indices from FrozenSlot.
func frozenSlotIdxs[T tensor.Numeric](frozen []FrozenSlot[T]) []int {
	idxs := make([]int, len(frozen))
	for i, fs := range frozen {
		idxs[i] = fs.SlotIdx
	}
	return idxs
}

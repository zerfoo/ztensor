package tensorrt

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/tensor"
)

// TabularConfig controls compilation of a tabular model graph to TensorRT.
type TabularConfig struct {
	// FP16 enables FP16 precision mode. Default false (FP32).
	FP16 bool

	// MaxWorkspaceBytes sets the maximum TensorRT workspace memory.
	// Default: 64 MB.
	MaxWorkspaceBytes int

	// MaxBatchSize is the maximum batch dimension for dynamic shapes.
	// Default: 64.
	MaxBatchSize int

	// OptBatchSize is the optimal batch size for the optimization profile.
	// Default: 1.
	OptBatchSize int
}

func (c *TabularConfig) defaults() {
	if c.MaxWorkspaceBytes == 0 {
		c.MaxWorkspaceBytes = 64 << 20 // 64 MB
	}
	if c.MaxBatchSize == 0 {
		c.MaxBatchSize = 64
	}
	if c.OptBatchSize == 0 {
		c.OptBatchSize = 1
	}
}

// TabularEngine wraps a compiled TensorRT engine for tabular model inference.
// It holds the serialized engine, runtime, execution context, and CUDA stream.
type TabularEngine struct {
	logger  *Logger
	runtime *Runtime
	engine  *Engine
	ctx     *ExecutionContext
	stream  *cuda.Stream

	inputName  string
	outputName string
	inputDims  []int32 // [batch, features] excluding batch for dynamic shape
	outputDims []int32

	// Device memory buffers
	inputDev  unsafe.Pointer
	outputDev unsafe.Pointer
	inputLen  int // in bytes
	outputLen int // in bytes
}

// CompileTabular compiles a graph.ExecutionPlan from a tabular model into a
// TensorRT engine. Tabular models are small feed-forward networks (MLP) with
// operations: MatMul, Add, ReLU, Sigmoid, Tanh, Softmax, ReduceSum.
//
// The plan must have been compiled from a graph with a single input tensor
// of shape [batch, features] and a single output tensor.
func CompileTabular[T tensor.Numeric](plan *graph.ExecutionPlan[T], cfg TabularConfig) (*TabularEngine, error) {
	if !Available() {
		return nil, fmt.Errorf("tensorrt: library not available")
	}
	cfg.defaults()

	logger := CreateLogger(SeverityWarning)
	if logger == nil {
		return nil, fmt.Errorf("tensorrt: failed to create logger")
	}

	builder, err := CreateBuilder(logger)
	if err != nil {
		logger.Destroy()
		return nil, err
	}
	defer builder.Destroy()

	network, err := builder.CreateNetwork()
	if err != nil {
		logger.Destroy()
		return nil, err
	}
	defer network.Destroy()

	config, err := builder.CreateBuilderConfig()
	if err != nil {
		logger.Destroy()
		return nil, err
	}
	defer config.Destroy()

	config.SetMemoryPoolLimit(cfg.MaxWorkspaceBytes)
	if cfg.FP16 {
		config.SetFlag(FlagFP16)
	}

	// Translate the execution plan into a TensorRT network.
	instructions := plan.Instructions()
	slotShapes := plan.SlotShapes()
	inputSlots := plan.InputSlots()
	outputSlot := plan.OutputSlot()
	frozenSlots := plan.FrozenSlots()

	if len(inputSlots) != 1 {
		logger.Destroy()
		return nil, fmt.Errorf("tensorrt: tabular models require exactly 1 input, got %d", len(inputSlots))
	}

	// Build frozen slot data map for weight constants.
	frozenData := make(map[int]*tensor.TensorNumeric[T])
	for _, fs := range frozenSlots {
		frozenData[fs.SlotIdx] = fs.Data
	}

	// Determine input shape from the slot shapes.
	inputShape := slotShapes[inputSlots[0]]
	if len(inputShape) < 1 || len(inputShape) > 2 {
		logger.Destroy()
		return nil, fmt.Errorf("tensorrt: input shape must be [batch, features] or [features], got %v", inputShape)
	}

	// Convert input shape to TRT dims. Use dynamic batch dimension (-1).
	var trtInputDims []int32
	var featureDim int32
	if len(inputShape) == 2 {
		trtInputDims = []int32{-1, int32(inputShape[1])}
		featureDim = int32(inputShape[1])
	} else {
		trtInputDims = []int32{int32(inputShape[0])}
		featureDim = int32(inputShape[0])
	}

	// Add input tensor to the network.
	input := network.AddInput("input", Float32, trtInputDims)
	if input == nil {
		logger.Destroy()
		return nil, fmt.Errorf("tensorrt: failed to add input tensor")
	}

	// Set up optimization profile for dynamic batch.
	if len(inputShape) == 2 {
		profile, err := builder.CreateOptimizationProfile()
		if err != nil {
			logger.Destroy()
			return nil, err
		}
		err = profile.SetDimensions("input",
			[]int32{1, featureDim},                     // min
			[]int32{int32(cfg.OptBatchSize), featureDim}, // opt
			[]int32{int32(cfg.MaxBatchSize), featureDim}, // max
		)
		if err != nil {
			logger.Destroy()
			return nil, err
		}
		if _, err := profile.AddToConfig(config); err != nil {
			logger.Destroy()
			return nil, err
		}
	}

	// Map slot index -> TensorRT tensor for wiring layers.
	slotTensors := make(map[int]*Tensor)
	slotTensors[inputSlots[0]] = input

	// Translate each instruction to TensorRT layers.
	for i, inst := range instructions {
		var outTensor *Tensor
		var layerErr error

		switch inst.OpName {
		case "MatMul", "MatMulTransposeB":
			outTensor, layerErr = addMatMulLayer(network, inst, slotTensors, frozenData, slotShapes, inst.OpName == "MatMulTransposeB")

		case "Add":
			outTensor, layerErr = addElementWiseLayer(network, inst, slotTensors, frozenData, slotShapes, ElementWiseSum)

		case "Sub":
			outTensor, layerErr = addElementWiseLayer(network, inst, slotTensors, frozenData, slotShapes, ElementWiseSub)

		case "Mul":
			outTensor, layerErr = addElementWiseLayer(network, inst, slotTensors, frozenData, slotShapes, ElementWiseProd)

		case "Div":
			outTensor, layerErr = addElementWiseLayer(network, inst, slotTensors, frozenData, slotShapes, ElementWiseDiv)

		case "Exp", "Log", "Tanh", "Sqrt", "Rsqrt":
			// These unary ops are not directly supported as TRT layers in our binding.
			// Use activation or elementwise workarounds where possible.
			if inst.OpName == "Tanh" {
				outTensor, layerErr = addActivationLayer(network, inst, slotTensors, ActivationTanh)
			} else {
				layerErr = fmt.Errorf("unsupported unary op %q", inst.OpName)
			}

		case "Softmax":
			outTensor, layerErr = addSoftmaxLayer(network, inst, slotTensors)

		case "ReduceSum":
			outTensor, layerErr = addReduceLayer(network, inst, slotTensors, ReduceSum)

		case "ReduceMean":
			outTensor, layerErr = addReduceLayer(network, inst, slotTensors, ReduceAvg)

		case "Reshape":
			outTensor, layerErr = addReshapeLayer(network, inst, slotTensors)

		default:
			// Check if this is a composite op with a known activation pattern.
			// Common tabular patterns: ReLU, Sigmoid activations in Dense layers.
			act, ok := opToActivation(inst.OpName)
			if ok {
				outTensor, layerErr = addActivationLayer(network, inst, slotTensors, act)
			} else {
				layerErr = fmt.Errorf("unsupported op %q", inst.OpName)
			}
		}

		if layerErr != nil {
			logger.Destroy()
			return nil, fmt.Errorf("tensorrt: instruction %d (%s): %w", i, inst.OpName, layerErr)
		}
		if outTensor == nil {
			logger.Destroy()
			return nil, fmt.Errorf("tensorrt: instruction %d (%s): produced nil tensor", i, inst.OpName)
		}
		slotTensors[inst.OutputIdx] = outTensor
	}

	// Mark the output.
	outputTensor, ok := slotTensors[outputSlot]
	if !ok {
		logger.Destroy()
		return nil, fmt.Errorf("tensorrt: output slot %d not found", outputSlot)
	}
	network.MarkOutput(outputTensor)

	// Build serialized engine.
	serialized, err := builder.BuildSerializedNetwork(network, config)
	if err != nil {
		logger.Destroy()
		return nil, err
	}

	// Deserialize into runtime + engine.
	rt, err := CreateRuntime(logger)
	if err != nil {
		logger.Destroy()
		return nil, err
	}

	engine, err := rt.DeserializeEngine(serialized)
	if err != nil {
		rt.Destroy()
		logger.Destroy()
		return nil, err
	}

	ctx, err := engine.CreateExecutionContext()
	if err != nil {
		engine.Destroy()
		rt.Destroy()
		logger.Destroy()
		return nil, err
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		ctx.Destroy()
		engine.Destroy()
		rt.Destroy()
		logger.Destroy()
		return nil, fmt.Errorf("tensorrt: %w", err)
	}

	// Determine output shape.
	outputShape := slotShapes[outputSlot]
	var trtOutputDims []int32
	for _, d := range outputShape {
		trtOutputDims = append(trtOutputDims, int32(d))
	}

	// Find output tensor name.
	outputName := ""
	for i := 0; i < engine.NumIOTensors(); i++ {
		name := engine.GetIOTensorName(i)
		if name != "input" {
			outputName = name
			break
		}
	}

	te := &TabularEngine{
		logger:     logger,
		runtime:    rt,
		engine:     engine,
		ctx:        ctx,
		stream:     stream,
		inputName:  "input",
		outputName: outputName,
		inputDims:  trtInputDims,
		outputDims: trtOutputDims,
	}

	return te, nil
}

// Infer runs inference on the compiled TensorRT engine with the given input data.
// inputData shape must be [batch, features]. Returns output data as a flat float32 slice.
func (te *TabularEngine) Infer(inputData []float32, batchSize int) ([]float32, error) {
	inputBytes := len(inputData) * 4

	// Compute output size.
	outputElems := batchSize
	for i, d := range te.outputDims {
		if i == 0 {
			continue // skip batch dim
		}
		outputElems *= int(d)
	}
	if len(te.outputDims) == 1 {
		outputElems = batchSize * int(te.outputDims[0])
	}
	outputBytes := outputElems * 4

	// Allocate device memory if needed or if size changed.
	if te.inputDev == nil || te.inputLen != inputBytes {
		if te.inputDev != nil {
			_ = cuda.Free(te.inputDev)
		}
		var err error
		te.inputDev, err = cuda.Malloc(inputBytes)
		if err != nil {
			return nil, fmt.Errorf("tensorrt: malloc input: %w", err)
		}
		te.inputLen = inputBytes
	}
	if te.outputDev == nil || te.outputLen != outputBytes {
		if te.outputDev != nil {
			_ = cuda.Free(te.outputDev)
		}
		var err error
		te.outputDev, err = cuda.Malloc(outputBytes)
		if err != nil {
			return nil, fmt.Errorf("tensorrt: malloc output: %w", err)
		}
		te.outputLen = outputBytes
	}

	// Set dynamic input shape.
	if len(te.inputDims) == 2 {
		dynamicDims := []int32{int32(batchSize), te.inputDims[1]}
		if err := te.ctx.SetInputShape("input", dynamicDims); err != nil {
			return nil, err
		}
	}

	// Copy input to device.
	if err := cuda.Memcpy(te.inputDev, unsafe.Pointer(&inputData[0]), inputBytes, cuda.MemcpyHostToDevice); err != nil {
		return nil, fmt.Errorf("tensorrt: memcpy H2D: %w", err)
	}

	// Bind tensors.
	if err := te.ctx.SetTensorAddress(te.inputName, te.inputDev); err != nil {
		return nil, err
	}
	if err := te.ctx.SetTensorAddress(te.outputName, te.outputDev); err != nil {
		return nil, err
	}

	// Run inference.
	if err := te.ctx.EnqueueV3(te.stream.Ptr()); err != nil {
		return nil, err
	}
	if err := te.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("tensorrt: synchronize: %w", err)
	}

	// Copy output back.
	result := make([]float32, outputElems)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), te.outputDev, outputBytes, cuda.MemcpyDeviceToHost); err != nil {
		return nil, fmt.Errorf("tensorrt: memcpy D2H: %w", err)
	}

	return result, nil
}

// Destroy releases all TensorRT and CUDA resources.
func (te *TabularEngine) Destroy() {
	if te.inputDev != nil {
		_ = cuda.Free(te.inputDev)
		te.inputDev = nil
	}
	if te.outputDev != nil {
		_ = cuda.Free(te.outputDev)
		te.outputDev = nil
	}
	if te.stream != nil {
		_ = te.stream.Destroy()
		te.stream = nil
	}
	if te.ctx != nil {
		te.ctx.Destroy()
		te.ctx = nil
	}
	if te.engine != nil {
		te.engine.Destroy()
		te.engine = nil
	}
	if te.runtime != nil {
		te.runtime.Destroy()
		te.runtime = nil
	}
	if te.logger != nil {
		te.logger.Destroy()
		te.logger = nil
	}
}

// --- Internal layer translation helpers ---

// opToActivation maps graph op names to TensorRT activation types.
func opToActivation(opName string) (ActivationType, bool) {
	switch opName {
	case "ReLU":
		return ActivationReLU, true
	case "Sigmoid":
		return ActivationSigmoid, true
	case "Tanh":
		return ActivationTanh, true
	default:
		return 0, false
	}
}

func addActivationLayer(network *NetworkDefinition, inst graph.InstructionMeta, slots map[int]*Tensor, act ActivationType) (*Tensor, error) {
	if len(inst.InputIdx) < 1 {
		return nil, fmt.Errorf("activation requires 1 input")
	}
	input, ok := slots[inst.InputIdx[0]]
	if !ok {
		return nil, fmt.Errorf("input slot %d not found", inst.InputIdx[0])
	}
	layer := network.AddActivation(input, act)
	if layer == nil {
		return nil, fmt.Errorf("AddActivation returned nil")
	}
	layer.SetName(fmt.Sprintf("%s_%d", inst.OpName, inst.OutputIdx))
	return layer.GetOutput(0), nil
}

func addMatMulLayer[T tensor.Numeric](network *NetworkDefinition, inst graph.InstructionMeta, slots map[int]*Tensor, frozen map[int]*tensor.TensorNumeric[T], shapes [][]int, transposeB bool) (*Tensor, error) {
	if len(inst.InputIdx) < 2 {
		return nil, fmt.Errorf("MatMul requires 2 inputs")
	}

	lhsIdx := inst.InputIdx[0]
	rhsIdx := inst.InputIdx[1]

	// Get or create LHS tensor.
	lhs, lhsOk := slots[lhsIdx]
	if !lhsOk {
		// LHS might be a frozen weight.
		if ft, ok := frozen[lhsIdx]; ok {
			lhs = addFrozenConstant(network, ft, shapes[lhsIdx])
		}
		if lhs == nil {
			return nil, fmt.Errorf("LHS slot %d not found", lhsIdx)
		}
	}

	// Get or create RHS tensor.
	rhs, rhsOk := slots[rhsIdx]
	if !rhsOk {
		if ft, ok := frozen[rhsIdx]; ok {
			rhs = addFrozenConstant(network, ft, shapes[rhsIdx])
		}
		if rhs == nil {
			return nil, fmt.Errorf("RHS slot %d not found", rhsIdx)
		}
	}

	op1 := MatrixOpNone
	if transposeB {
		op1 = MatrixOpTranspose
	}

	layer := network.AddMatrixMultiply(lhs, MatrixOpNone, rhs, op1)
	if layer == nil {
		return nil, fmt.Errorf("AddMatrixMultiply returned nil")
	}
	layer.SetName(fmt.Sprintf("%s_%d", inst.OpName, inst.OutputIdx))
	return layer.GetOutput(0), nil
}

func addElementWiseLayer[T tensor.Numeric](network *NetworkDefinition, inst graph.InstructionMeta, slots map[int]*Tensor, frozen map[int]*tensor.TensorNumeric[T], shapes [][]int, op ElementWiseOp) (*Tensor, error) {
	if len(inst.InputIdx) < 2 {
		return nil, fmt.Errorf("elementwise op requires 2 inputs")
	}

	lhsIdx := inst.InputIdx[0]
	rhsIdx := inst.InputIdx[1]

	lhs, lhsOk := slots[lhsIdx]
	if !lhsOk {
		if ft, ok := frozen[lhsIdx]; ok {
			lhs = addFrozenConstant(network, ft, shapes[lhsIdx])
		}
		if lhs == nil {
			return nil, fmt.Errorf("LHS slot %d not found", lhsIdx)
		}
	}

	rhs, rhsOk := slots[rhsIdx]
	if !rhsOk {
		if ft, ok := frozen[rhsIdx]; ok {
			rhs = addFrozenConstant(network, ft, shapes[rhsIdx])
		}
		if rhs == nil {
			return nil, fmt.Errorf("RHS slot %d not found", rhsIdx)
		}
	}

	layer := network.AddElementWise(lhs, rhs, op)
	if layer == nil {
		return nil, fmt.Errorf("AddElementWise returned nil")
	}
	layer.SetName(fmt.Sprintf("%s_%d", inst.OpName, inst.OutputIdx))
	return layer.GetOutput(0), nil
}

func addSoftmaxLayer(network *NetworkDefinition, inst graph.InstructionMeta, slots map[int]*Tensor) (*Tensor, error) {
	if len(inst.InputIdx) < 1 {
		return nil, fmt.Errorf("softmax requires 1 input")
	}
	input, ok := slots[inst.InputIdx[0]]
	if !ok {
		return nil, fmt.Errorf("input slot %d not found", inst.InputIdx[0])
	}

	axis := 1 // default: softmax over last dim
	if inst.ExtraArgs != nil {
		if a, ok := inst.ExtraArgs["axis"]; ok {
			switch v := a.(type) {
			case int:
				axis = v
			case float64:
				axis = int(v)
			}
		}
	}

	layer := network.AddSoftMax(input, axis)
	if layer == nil {
		return nil, fmt.Errorf("AddSoftMax returned nil")
	}
	layer.SetName(fmt.Sprintf("Softmax_%d", inst.OutputIdx))
	return layer.GetOutput(0), nil
}

func addReduceLayer(network *NetworkDefinition, inst graph.InstructionMeta, slots map[int]*Tensor, op ReduceOp) (*Tensor, error) {
	if len(inst.InputIdx) < 1 {
		return nil, fmt.Errorf("reduce requires 1 input")
	}
	input, ok := slots[inst.InputIdx[0]]
	if !ok {
		return nil, fmt.Errorf("input slot %d not found", inst.InputIdx[0])
	}

	axis := 0
	keepDims := false
	if inst.ExtraArgs != nil {
		if a, ok := inst.ExtraArgs["axis"]; ok {
			switch v := a.(type) {
			case int:
				axis = v
			case float64:
				axis = int(v)
			}
		}
		if kd, ok := inst.ExtraArgs["keepDims"]; ok {
			keepDims, _ = kd.(bool)
		}
	}

	// TensorRT uses a bitmask for axes.
	axisMask := uint32(1 << axis)
	layer := network.AddReduce(input, op, axisMask, keepDims)
	if layer == nil {
		return nil, fmt.Errorf("AddReduce returned nil")
	}
	layer.SetName(fmt.Sprintf("Reduce_%d", inst.OutputIdx))
	return layer.GetOutput(0), nil
}

func addReshapeLayer(network *NetworkDefinition, inst graph.InstructionMeta, slots map[int]*Tensor) (*Tensor, error) {
	if len(inst.InputIdx) < 1 {
		return nil, fmt.Errorf("reshape requires 1 input")
	}
	input, ok := slots[inst.InputIdx[0]]
	if !ok {
		return nil, fmt.Errorf("input slot %d not found", inst.InputIdx[0])
	}

	layer := network.AddShuffle(input)
	if layer == nil {
		return nil, fmt.Errorf("AddShuffle returned nil")
	}

	if inst.ExtraArgs != nil {
		if shapeRaw, ok := inst.ExtraArgs["shape"]; ok {
			var dims []int32
			switch s := shapeRaw.(type) {
			case []int:
				for _, v := range s {
					dims = append(dims, int32(v))
				}
			case []any:
				for _, v := range s {
					switch iv := v.(type) {
					case int:
						dims = append(dims, int32(iv))
					case float64:
						dims = append(dims, int32(iv))
					}
				}
			}
			if len(dims) > 0 {
				ShuffleSetReshapeDims(layer, dims)
			}
		}
	}

	layer.SetName(fmt.Sprintf("Reshape_%d", inst.OutputIdx))
	return layer.GetOutput(0), nil
}

// addFrozenConstant adds a constant weight tensor to the TensorRT network.
func addFrozenConstant[T tensor.Numeric](network *NetworkDefinition, t *tensor.TensorNumeric[T], shape []int) *Tensor {
	if t == nil {
		return nil
	}
	data := t.Data()
	if len(data) == 0 {
		return nil
	}

	dims := make([]int32, len(shape))
	for i, d := range shape {
		dims[i] = int32(d)
	}

	layer := network.AddConstant(dims, Float32, unsafe.Pointer(&data[0]), int64(len(data)))
	if layer == nil {
		return nil
	}
	return layer.GetOutput(0)
}

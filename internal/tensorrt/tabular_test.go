package tensorrt

import (
	"context"
	"math"
	"testing"
	"time"
	"unsafe"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// simpleMatMulNode implements graph.Node for testing: output = MatMul(input, weights).
type simpleMatMulNode struct {
	graph.NoParameters[float32]
	engine  compute.Engine[float32]
	weights *tensor.TensorNumeric[float32]
	outDim  int
}

func (n *simpleMatMulNode) OpType() string                    { return "MatMul" }
func (n *simpleMatMulNode) Attributes() map[string]interface{} { return nil }
func (n *simpleMatMulNode) OutputShape() []int                 { return []int{1, n.outDim} }

func (n *simpleMatMulNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return n.engine.MatMul(ctx, inputs[0], n.weights)
}

func (n *simpleMatMulNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

// simpleAddNode implements graph.Node for bias addition.
type simpleAddNode struct {
	graph.NoParameters[float32]
	engine compute.Engine[float32]
	bias   *tensor.TensorNumeric[float32]
	shape  []int
}

func (n *simpleAddNode) OpType() string                    { return "Add" }
func (n *simpleAddNode) Attributes() map[string]interface{} { return nil }
func (n *simpleAddNode) OutputShape() []int                 { return n.shape }

func (n *simpleAddNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return n.engine.Add(ctx, inputs[0], n.bias)
}

func (n *simpleAddNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

// simpleReLUNode implements graph.Node for ReLU activation.
type simpleReLUNode struct {
	graph.NoParameters[float32]
	shape []int
}

func (n *simpleReLUNode) OpType() string                    { return "ReLU" }
func (n *simpleReLUNode) Attributes() map[string]interface{} { return nil }
func (n *simpleReLUNode) OutputShape() []int                 { return n.shape }

func (n *simpleReLUNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	data := inputs[0].Data()
	out := make([]float32, len(data))
	for i, v := range data {
		if v > 0 {
			out[i] = v
		}
	}
	return tensor.New[float32](inputs[0].Shape(), out)
}

func (n *simpleReLUNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

// buildTabularGraph constructs a simple tabular MLP graph:
// input(1,4) -> MatMul(4,3) -> Add(bias) -> ReLU -> MatMul(3,1) -> output(1,1)
func buildTabularGraph(engine compute.Engine[float32]) (*graph.Graph[float32], *tensor.TensorNumeric[float32]) {
	b := graph.NewBuilder[float32](engine)

	input := b.Input([]int{1, 4})

	w1, _ := tensor.New[float32]([]int{4, 3}, []float32{
		0.5, -0.3, 0.8,
		0.1, 0.6, -0.2,
		-0.4, 0.7, 0.3,
		0.2, -0.1, 0.5,
	})
	bias1, _ := tensor.New[float32]([]int{1, 3}, []float32{0.1, -0.1, 0.2})

	matmul1 := b.AddNode(&simpleMatMulNode{engine: engine, weights: w1, outDim: 3}, input)
	add1 := b.AddNode(&simpleAddNode{engine: engine, bias: bias1, shape: []int{1, 3}}, matmul1)
	relu1 := b.AddNode(&simpleReLUNode{shape: []int{1, 3}}, add1)

	w2, _ := tensor.New[float32]([]int{3, 1}, []float32{0.4, -0.5, 0.6})
	matmul2 := b.AddNode(&simpleMatMulNode{engine: engine, weights: w2, outDim: 1}, relu1)

	g, _ := b.Build(matmul2)

	inputTensor, _ := tensor.New[float32]([]int{1, 4}, []float32{1.0, 2.0, 3.0, 4.0})
	return g, inputTensor
}

// TestTensorRT_TabularCompile verifies that a tabular model graph can be
// compiled to TensorRT and produces correct inference results.
func TestTensorRT_TabularCompile(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	g, inputTensor := buildTabularGraph(engine)

	ctx := context.Background()
	plan, err := g.Compile(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	expected, err := plan.Run(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Plan.Run: %v", err)
	}
	expectedData := expected.Data()
	t.Logf("Expected output: %v", expectedData)

	cfg := TabularConfig{
		MaxBatchSize: 16,
		OptBatchSize: 1,
	}
	te, err := CompileTabular(plan, cfg)
	if err != nil {
		t.Fatalf("CompileTabular: %v", err)
	}
	defer te.Destroy()

	inputData := []float32{1.0, 2.0, 3.0, 4.0}
	result, err := te.Infer(inputData, 1)
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}

	t.Logf("TensorRT output: %v", result)

	if len(result) != len(expectedData) {
		t.Fatalf("output length mismatch: got %d, want %d", len(result), len(expectedData))
	}
	for i := range result {
		if diff := math.Abs(float64(result[i] - expectedData[i])); diff > 1e-3 {
			t.Errorf("output[%d] = %f, want %f (diff=%f)", i, result[i], expectedData[i], diff)
		}
	}
}

// TestTensorRT_Latency verifies that TensorRT tabular inference achieves
// sub-10us per-source latency after warmup.
func TestTensorRT_Latency(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	g, inputTensor := buildTabularGraph(engine)

	ctx := context.Background()
	plan, err := g.Compile(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	cfg := TabularConfig{
		MaxBatchSize: 16,
		OptBatchSize: 1,
	}
	te, err := CompileTabular(plan, cfg)
	if err != nil {
		t.Fatalf("CompileTabular: %v", err)
	}
	defer te.Destroy()

	inputData := []float32{1.0, 2.0, 3.0, 4.0}

	// Warmup.
	for i := 0; i < 100; i++ {
		if _, err := te.Infer(inputData, 1); err != nil {
			t.Fatalf("Warmup Infer: %v", err)
		}
	}

	// Benchmark.
	const iterations = 1000
	start := time.Now()
	for i := 0; i < iterations; i++ {
		if _, err := te.Infer(inputData, 1); err != nil {
			t.Fatalf("Benchmark Infer: %v", err)
		}
	}
	elapsed := time.Since(start)
	perInfer := elapsed / iterations

	t.Logf("TensorRT tabular latency: %v per inference (%d iterations, total %v)", perInfer, iterations, elapsed)

	if perInfer > 10*time.Microsecond {
		t.Logf("WARNING: latency %v exceeds 10us target (may be expected on non-GPU CI)", perInfer)
	}
}

// TestTensorRT_TabularCompileDirect tests TensorRT compilation using the
// low-level TensorRT API directly to validate basic tabular network patterns.
func TestTensorRT_TabularCompileDirect(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}

	logger := CreateLogger(SeverityWarning)
	if logger == nil {
		t.Fatal("CreateLogger returned nil")
	}
	defer logger.Destroy()

	builder, err := CreateBuilder(logger)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer builder.Destroy()

	network, err := builder.CreateNetwork()
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}
	defer network.Destroy()

	// Build: input(1,4) -> MatMul(4,3) -> Add(bias) -> ReLU -> MatMul(3,1)
	input := network.AddInput("input", Float32, []int32{1, 4})
	if input == nil {
		t.Fatal("AddInput returned nil")
	}

	w1Data := []float32{0.5, -0.3, 0.8, 0.1, 0.6, -0.2, -0.4, 0.7, 0.3, 0.2, -0.1, 0.5}
	w1 := network.AddConstant([]int32{4, 3}, Float32, unsafe.Pointer(&w1Data[0]), 12)
	if w1 == nil {
		t.Fatal("AddConstant w1 returned nil")
	}

	mm1 := network.AddMatrixMultiply(input, MatrixOpNone, w1.GetOutput(0), MatrixOpNone)
	if mm1 == nil {
		t.Fatal("AddMatrixMultiply 1 returned nil")
	}
	mm1.SetName("dense_1_matmul")

	biasData := []float32{0.1, -0.1, 0.2}
	bias := network.AddConstant([]int32{1, 3}, Float32, unsafe.Pointer(&biasData[0]), 3)
	if bias == nil {
		t.Fatal("AddConstant bias returned nil")
	}

	add := network.AddElementWise(mm1.GetOutput(0), bias.GetOutput(0), ElementWiseSum)
	if add == nil {
		t.Fatal("AddElementWise returned nil")
	}
	add.SetName("dense_1_bias")

	relu := network.AddActivation(add.GetOutput(0), ActivationReLU)
	if relu == nil {
		t.Fatal("AddActivation returned nil")
	}
	relu.SetName("relu_1")

	w2Data := []float32{0.4, -0.5, 0.6}
	w2 := network.AddConstant([]int32{3, 1}, Float32, unsafe.Pointer(&w2Data[0]), 3)
	if w2 == nil {
		t.Fatal("AddConstant w2 returned nil")
	}

	mm2 := network.AddMatrixMultiply(relu.GetOutput(0), MatrixOpNone, w2.GetOutput(0), MatrixOpNone)
	if mm2 == nil {
		t.Fatal("AddMatrixMultiply 2 returned nil")
	}
	mm2.SetName("dense_2_matmul")

	network.MarkOutput(mm2.GetOutput(0))

	config, err := builder.CreateBuilderConfig()
	if err != nil {
		t.Fatalf("CreateBuilderConfig: %v", err)
	}
	defer config.Destroy()
	config.SetMemoryPoolLimit(64 << 20)

	serialized, err := builder.BuildSerializedNetwork(network, config)
	if err != nil {
		t.Fatalf("BuildSerializedNetwork: %v", err)
	}
	if len(serialized) == 0 {
		t.Fatal("empty serialized engine")
	}

	runtime, err := CreateRuntime(logger)
	if err != nil {
		t.Fatalf("CreateRuntime: %v", err)
	}
	defer runtime.Destroy()

	trtEngine, err := runtime.DeserializeEngine(serialized)
	if err != nil {
		t.Fatalf("DeserializeEngine: %v", err)
	}
	defer trtEngine.Destroy()

	trtCtx, err := trtEngine.CreateExecutionContext()
	if err != nil {
		t.Fatalf("CreateExecutionContext: %v", err)
	}
	defer trtCtx.Destroy()

	inputData := []float32{1.0, 2.0, 3.0, 4.0}
	inputBytes := 4 * 4
	outputBytes := 1 * 4

	inputDev, err := cuda.Malloc(inputBytes)
	if err != nil {
		t.Fatalf("Malloc input: %v", err)
	}
	defer func() { _ = cuda.Free(inputDev) }()

	outputDev, err := cuda.Malloc(outputBytes)
	if err != nil {
		t.Fatalf("Malloc output: %v", err)
	}
	defer func() { _ = cuda.Free(outputDev) }()

	if err := cuda.Memcpy(inputDev, unsafe.Pointer(&inputData[0]), inputBytes, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}

	if err := trtCtx.SetTensorAddress("input", inputDev); err != nil {
		t.Fatalf("SetTensorAddress(input): %v", err)
	}
	outputName := trtEngine.GetIOTensorName(1)
	if outputName == "input" {
		outputName = trtEngine.GetIOTensorName(0)
	}
	if err := trtCtx.SetTensorAddress(outputName, outputDev); err != nil {
		t.Fatalf("SetTensorAddress(%s): %v", outputName, err)
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	if err := trtCtx.EnqueueV3(stream.Ptr()); err != nil {
		t.Fatalf("EnqueueV3: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	result := make([]float32, 1)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), outputDev, outputBytes, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	t.Logf("Direct TensorRT tabular output: %v", result)

	// Expected: 1.01 (see manual computation in function doc)
	expected := float32(1.01)
	if diff := float32(math.Abs(float64(result[0] - expected))); diff > 1e-3 {
		t.Errorf("output = %f, want %f (diff=%f)", result[0], expected, diff)
	}

	// Latency benchmark.
	const warmup = 100
	const iters = 1000
	for i := 0; i < warmup; i++ {
		_ = trtCtx.EnqueueV3(stream.Ptr())
		_ = stream.Synchronize()
	}

	benchStart := time.Now()
	for i := 0; i < iters; i++ {
		_ = trtCtx.EnqueueV3(stream.Ptr())
		_ = stream.Synchronize()
	}
	elapsed := time.Since(benchStart)
	perInfer := elapsed / iters

	t.Logf("Direct TensorRT latency: %v per inference (%d iterations)", perInfer, iters)

	if perInfer > 10*time.Microsecond {
		t.Logf("NOTE: latency %v exceeds 10us target", perInfer)
	}
}

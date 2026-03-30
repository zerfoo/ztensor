# ztensor Design

ztensor is a GPU-accelerated tensor, compute engine, and computation graph library written in Go. It provides the numerical foundation for the Zerfoo ML framework: multi-type tensor storage, a pluggable compute engine interface, a compilable computation graph with automatic differentiation, and a multi-vendor GPU abstraction layer.

## Package Layout

```
tensor/      Tensor storage and numeric type system
compute/     Engine interface and CPU/GPU implementations
graph/       Computation graph, compilation, and CUDA graph capture
device/      Device abstraction and memory allocators
numeric/     Type-safe arithmetic operations
batched/     Batched multi-model inference
internal/    GPU bindings, SIMD assembly, code generation
  gpuapi/      GPU Runtime Abstraction Layer (GRAL)
  cuda/        CUDA runtime bindings (purego, zero CGo)
  cublas/      cuBLAS linear algebra bindings
  cudnn/       cuDNN deep-learning primitives
  tensorrt/    TensorRT integration
  hip/         AMD HIP runtime bindings
  rocblas/     AMD rocBLAS bindings
  miopen/      AMD MIOpen bindings
  opencl/      OpenCL runtime bindings
  clblast/     CLBlast linear algebra
  metal/       Apple Metal compute bindings
  fpga/        FPGA accelerator bindings
  sycl/        Intel oneAPI SYCL bindings
  nccl/        NCCL multi-GPU communication
  xblas/       ARM NEON and x86 AVX2 SIMD assembly kernels
  codegen/     Megakernel code generator
  workerpool/  Goroutine pool for parallel graph execution
gguf/        GGUF v3 file writer
log/         Structured leveled logging
metrics/     Evaluation metrics (Pearson, Spearman, MSE, RMSE, MAE)
testing/     Test utilities
types/       Shared type definitions (backward mode enum)
```

## tensor/

The `tensor` package defines the core data structure and type system.

### Numeric Type System

The `Numeric` interface constraint enumerates every element type the library supports:

- Native Go types: int, int8, int16, int32, int64, uint, uint8, uint32, uint64, float32, float64
- Custom minifloat types: float8.Float8 (E4M3FN), float16.Float16 (IEEE 754 half-precision), float16.BFloat16

The `Float` constraint restricts to float32 and float64 for operations that require native Go arithmetic operators. The `Addable` constraint covers all types that support built-in Go operators, excluding custom minifloat types that require conversion helpers.

### TensorNumeric[T]

`TensorNumeric[T Numeric]` is the concrete n-dimensional array type, parameterized by element type. It holds a shape, strides (row-major), and a `Storage[T]` backend. Tensors support views (shared storage with different strides) and release hooks for GPU memory cleanup.

### Storage Abstraction

The `Storage[T]` interface decouples tensor data from its physical location:

- **CPUStorage[T]**: wraps a Go slice; `Slice()` is zero-copy.
- **GPUStorage**: device-resident memory; `Slice()` performs a device-to-host copy.
- **MmapStorage**: memory-mapped file backing for zero-copy model loading.

### Quantized Storage Types

Quantized storage types store weights in compressed formats and dequantize on read:

| Type | Bits/Weight | Block Size | Format |
|------|------------|------------|--------|
| Q4Storage | 4 | 32 | float16 scale + packed 4-bit nibbles (GGML Q4_0) |
| Q5Storage | 5 | 32 | float16 scale + 5-bit values |
| Q8Storage | 8 | 32 | float16 scale + int8 values |
| Float16Storage | 16 | -- | IEEE 754 half-precision, bulk dequantize to float32 |
| BFloat16Storage | 16 | -- | Brain floating point, truncated mantissa |
| FP8Storage | 8 | -- | E4M3FN format for quantized inference |
| IQ2Storage | ~2.5 | -- | Importance-weighted 2-bit quantization |
| IQ3Storage | ~3.4 | -- | Importance-weighted 3-bit with grid lookup |
| IQ4Storage | ~4.5 | -- | Importance-weighted 4-bit quantization |
| TernaryStorage | 1.6 | -- | Ternary values (-1, 0, +1) for extreme quantization |
| W8A8Storage | 8 | -- | Weight 8-bit / activation 8-bit symmetric quantization |
| AWQStorage | 4 | -- | Activation-aware weight quantization |
| GPTQStorage | 4 | -- | Post-training quantization with group scaling |
| NF4Storage | 4 | -- | 4-bit NormalFloat (QLoRA format) |
| NVFP4Storage | 4 | -- | NVIDIA FP4 format |

All quantized types implement `Storage[float32]`; their `Slice()` method dequantizes to float32 on read. GPU engines can access the raw compressed bytes directly via a device pointer for quantized GEMM/GEMV kernels that dequantize on the fly.

### Quant Registry

The `quant_registry.go` file maintains a registry mapping GGUF quantization type IDs to constructor functions, enabling dynamic dispatch during model loading.

## compute/

The `compute` package defines the `Engine[T]` interface and provides CPU and GPU implementations.

### Engine[T] Interface

`Engine[T Numeric]` is the central abstraction for all tensor arithmetic. Every operation in the framework flows through an engine instance, enabling transparent switching between CPU and GPU execution without changing calling code. The interface includes:

- **Element-wise ops**: Add, Sub, Mul, Div, MulScalar, DivScalar, AddScalar
- **Math functions**: Exp, Log, Sin, Cos, Tanh, TanhPrime, Pow, Sqrt, Rsqrt
- **Matrix operations**: MatMul, Transpose, HadamardTransform
- **Reductions**: Sum, ReduceSum, ReduceMax, ReduceMean, Softmax
- **Data manipulation**: Gather, ScatterAdd, Split, Concat, Repeat, Reshape, Copy, Fill, Zero, Zeros
- **Utilities**: RandomUniform, OneHot, UnaryOp

All methods accept a `context.Context` and an optional variadic `dst` tensor for in-place buffer reuse, reducing allocation pressure during inference.

### Optional Engine Capabilities

Beyond the core interface, engines may implement optional capability interfaces discovered via type assertion:

| Interface | Purpose |
|-----------|---------|
| FusedRMSNormer | GPU-accelerated fused RMSNorm kernel |
| TransposeBMatMuler[T] | MatMul with implicit B transpose (eliminates an allocation) |
| StreamProvider | Exposes the GPU stream handle for CUDA graph capture |
| GPUStreamAccessor | Provides the gpuapi.Stream for async memory operations |
| GPUArgmaxer | On-device argmax without device-to-host logit copy |
| FP16ToF32Converter | Convert FP16 tensors to float32 on device |
| PagedGQAer | Paged grouped-query attention via block-table indirection |
| PoolResetter | O(1) arena reset at the start of each forward pass |
| WeightUploader | Pre-upload model weights to device memory at load time |

### CPUEngine

`CPUEngine[T]` implements `Engine[T]` using pure Go with optional SIMD acceleration (ARM NEON, x86 AVX2) for hot paths like matrix multiplication. It uses a `TensorArena` for power-of-2 bucketed buffer pooling to reduce GC pressure.

### GPUEngine

`GPUEngine` implements `Engine[float32]` using the GPU Runtime Abstraction Layer (GRAL). It delegates linear algebra to vendor BLAS libraries, deep-learning primitives to vendor DNN libraries, and custom fused operations to hand-written GPU kernels. It also implements the optional capability interfaces listed above.

### TensorArena

`TensorArena` is a power-of-2 bucketed pool for float32 backing arrays. Buffers are bucketed by capacity (2^0 through 2^31) with per-bucket mutex protection. `Get(n)` returns a zeroed slice of at least n elements; `Put(buf)` returns it for reuse. `Reset()` clears all pooled buffers.

### MemoryTracker

`MemoryTracker` provides atomic byte-level allocation tracking with an optional upper limit. When a limit is set, allocations that would exceed it return `ErrMemoryLimitExceeded`. All methods are lock-free using compare-and-swap.

### EngineProxy

`EngineProxy[T]` wraps an `Engine[T]` and allows swapping the underlying engine at runtime. This is used during CUDA graph capture to redirect operations through a recording engine, then replay them via the captured graph.

### Fused Operations

The compute package includes several fused operation implementations that combine multiple elementwise operations into a single pass, reducing memory bandwidth and kernel launch overhead:

- **FusedRMSNorm**: root-mean-square normalization in a single pass
- **FusedSiLUGate**: SiLU activation with gating (used in feed-forward networks)
- **FusedSwiGLU**: SwiGLU activation (SiLU-gated linear unit)
- **FusedAddRMSNorm**: residual addition combined with RMSNorm
- **FusedNormAdd**: normalization combined with residual addition
- **FusedQKNormRoPE**: query/key normalization with rotary position embedding
- **FusedRoPE**: standalone rotary position embedding
- **FusedScaledSoftmax**: softmax with integrated scaling factor
- **FlashDecode**: optimized single-token attention decode

### Quantized Compute

Specialized compute paths handle mixed-precision arithmetic:

- **W4A16**: 4-bit weight, 16-bit activation GEMV for quantized inference
- **TernaryGEMV**: ternary weight (-1, 0, +1) GEMV using bitwise operations

## graph/

The `graph` package provides a computation graph that supports forward/backward execution, compilation to flat instruction sequences, optimization passes, and CUDA graph capture.

### Graph[T]

`Graph[T Numeric]` holds a topologically sorted list of nodes, a dependency map, designated input and output nodes, and a memo table for caching intermediate results during forward execution. Key features:

- **Forward execution**: evaluates nodes in topological order, caching results in the memo table. When a `TensorReleaser` pool is set, intermediate tensors are released as soon as all consumers have executed.
- **Backward execution**: traverses nodes in reverse topological order, accumulating gradients via the engine's Add operation.
- **Parallel mode**: when enabled, independent nodes execute concurrently using a goroutine pool.
- **KV cache feedback**: stateful input nodes can be linked to output nodes so that each forward pass feeds the output back as input to the next pass.
- **Parameter access**: `Parameters()` returns all trainable parameters sorted by name; `LoadParameters()` loads values by name from a map.

### Node[T] Interface

Every operation in the graph implements `Node[T]`:

```go
type Node[T tensor.Numeric] interface {
    OpType() string
    Attributes() map[string]interface{}
    Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
    Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error)
    Parameters() []*Parameter[T]
    OutputShape() []int
}
```

### Compilation

`Compile()` transforms a graph into an `ExecutionPlan` -- a flat instruction sequence with indexed slot storage. This eliminates map lookups and memo operations from the forward path. The execution plan includes:

- **Slot array**: indexed tensor storage replacing the map-based memo table
- **Instruction list**: each instruction holds pre-resolved input/output slot indices and a direct Forward function reference
- **Frozen slots**: parameter and constant tensors are placed in frozen slots that persist across runs
- **Buffer layout**: pre-computed shapes for all slots, enabling arena pre-allocation
- **Last-use tracking**: per-slot last-consumer indices for intra-pass buffer reuse

### Optimization Passes

- **FoldConstantTransposes**: pre-applies transpose operations to constant tensors at compile time, eliminating runtime transpose nodes.

### CUDA Graph Capture

`CUDAGraphExecutor` captures and replays CUDA graphs for compiled execution plans. The plan is split into three regions:

1. **Pre-capture**: instructions that perform host-device transfers or have dynamic state (embedding lookups, attention masks, position IDs, slicing, shape ops)
2. **Capture region**: GPU-only, position-independent instructions that can be recorded into a CUDA graph
3. **Post-capture**: any remaining non-capturable instructions

Non-capturable operations are identified by op type (EmbeddingLookup, Gather, AutoAttentionMask, AutoPositionIds, Slice, ConstantOfShape, Shape) and run outside the captured graph. The capture region is recorded once, then replayed on each forward pass with near-zero CPU overhead.

### BufferArena

`BufferArena[T]` pre-allocates one tensor per slot shape. Frozen slots (parameters, constants) are not zeroed on reset. Non-frozen buffers are cleared between runs to avoid stale data.

### Builder

`Builder[T]` provides a fluent API for constructing graphs by adding nodes and wiring dependencies. It performs topological sorting and validation before producing a `Graph[T]`.

### Checkpointing

The graph package supports saving and loading graph state (parameter values) for training checkpoints.

## device/

The `device` package provides a hardware abstraction layer for compute devices and memory allocation.

### Device Interface

```go
type Device interface {
    ID() string
    GetAllocator() Allocator
    Type() Type
}
```

Supported device types: CPU, CUDA, ROCm, OpenCL, Metal, FPGA, SYCL.

### Device Registry

A global registry maps device IDs (e.g., "cpu", "cuda:0", "rocm:0") to `Device` instances. The CPU device is registered automatically at package init. GPU devices are registered when their runtime is initialized.

### Allocator Interface

```go
type Allocator interface {
    Allocate(size int) (any, error)
    Free(ptr any) error
}
```

The CPU allocator creates Go slices and relies on the garbage collector. GPU allocators delegate to their respective runtime's malloc/free functions.

## numeric/

The `numeric` package defines the `Arithmetic[T]` interface, which abstracts all scalar mathematical operations needed by the compute engine:

- Basic arithmetic: Add, Sub, Mul, Div
- Activation functions and gradients: Tanh, Sigmoid, ReLU, LeakyReLU (with corresponding gradient functions)
- Conversion: FromFloat32, FromFloat64, One
- Comparison and utilities: IsZero, Abs, GreaterThan
- Math functions: Sum, Exp, Log, Pow, Sqrt

Concrete implementations are provided for each supported numeric type (float32, float64, float16, bfloat16, float8, int8, uint8). This allows the `Engine[T]` to be completely agnostic to the specific numeric type it operates on.

## batched/

The `batched` package enables multi-model batched inference. When many models share the same architecture (identical layer sizes and activations) but have different weights, this package stacks their weights into a single 3D tensor and executes one batched GEMM call instead of N sequential matrix multiplications. This is designed for scenarios with hundreds or thousands of per-source models.

An `Architecture` struct describes the shared model structure (layer specs with input/output sizes and activation types). All models in a batch must conform to the same architecture.

## internal/

### GPU Runtime Abstraction Layer (GRAL)

The `internal/gpuapi` package defines the internal interfaces that decouple the compute engine from vendor-specific GPU APIs. These interfaces are not exported to users; the public `Engine[T]` interface in `compute/` is the user-facing API.

**Runtime** abstracts device and memory management:
- Device enumeration and selection (SetDevice, GetDeviceCount)
- Memory allocation and deallocation (Malloc, Free)
- Synchronous and asynchronous memory copies (Memcpy, MemcpyAsync, MemcpyPeer)
- Stream creation for asynchronous command queues

**BLAS** abstracts linear algebra operations (GEMM, GEMV, batched GEMM).

**DNN** abstracts deep-learning primitives (convolution, pooling, normalization).

**KernelRunner** abstracts custom GPU kernel dispatch.

**MemPool** abstracts GPU memory pooling and arena allocation.

Each vendor provides adapter implementations:

| Vendor | Runtime | BLAS | DNN | Kernels |
|--------|---------|------|-----|---------|
| NVIDIA CUDA | cuda_runtime | cublas | cudnn | cuda_kernels |
| AMD ROCm | rocm_runtime (HIP) | rocblas | miopen | rocm_kernels |
| OpenCL | opencl_runtime | clblast | opencl_dnn | opencl_kernels |
| Apple Metal | metal_runtime | metal_blas | metal_dnn | metal_kernels |
| Intel SYCL | sycl_runtime | sycl_blas | sycl_dnn | sycl_kernels |
| FPGA | fpga_runtime | fpga_blas | fpga_dnn | fpga_kernels |

All GPU bindings use purego/dlopen for dynamic library loading at runtime. No CGo is required for a CPU-only build; GPU support is loaded dynamically when available.

### CUDA Subsystem (internal/cuda/)

Hand-written CUDA kernel bindings loaded via purego. Includes custom kernels for fused operations, quantized GEMM/GEMV, rotary position embeddings, and attention.

### SIMD (internal/xblas/)

Hand-written ARM NEON and x86 AVX2 assembly kernels for CPU hot paths (matrix multiplication, vector operations).

### Code Generation (internal/codegen/)

Megakernel code generator that fuses multiple graph operations into a single GPU kernel launch, reducing kernel dispatch overhead.

### Multi-GPU Communication (internal/nccl/)

NCCL bindings for multi-GPU collective operations (all-reduce, all-gather) used by distributed training.

## gguf/

A GGUF v3 file writer for producing model files. Buffers metadata key-value pairs and tensor data, then writes the complete file in a single call. Used by conversion tools to produce model files compatible with the inference pipeline.

## log/

Structured, leveled logging with Debug, Info, Warn, and Error levels. Two implementations:
- **StdLogger**: writes to an io.Writer with configurable level filtering and text or JSON output format.
- **nopLogger**: zero-allocation no-op logger for when logging is disabled.

## metrics/

Evaluation metrics for model performance: Pearson correlation, Spearman correlation, MSE, RMSE, and MAE. Used by training and evaluation pipelines to assess model quality.

## Memory Architecture

### CPU Memory

CPU tensors use Go slices managed by the garbage collector. The `TensorArena` provides a bucketed pool for reusing intermediate buffers during inference, reducing GC pressure. The `MemoryTracker` enforces optional byte-level allocation limits with lock-free atomic operations.

### GPU Memory

GPU tensors hold device pointers allocated via the GRAL Runtime.Malloc interface. The GPU memory arena (`internal/gpuapi/cuda_arena.go`) provides sub-allocation from large pre-allocated blocks. The `MemPool` interface abstracts vendor-specific memory pooling.

Quantized tensors can hold both a CPU-side block array and an optional GPU device pointer. When weights are pre-uploaded via `WeightUploader`, the GPU pointer is set and subsequent operations read directly from device memory without per-operation host-to-device copies.

### Arena Reset

At the start of each inference pass, `PoolResetter.ResetPool()` reclaims all per-pass intermediate allocations in O(1) by resetting the arena's free pointer. This avoids per-tensor deallocation overhead.

## CUDA Graph Capture Lifecycle

CUDA graph capture records a sequence of GPU operations once, then replays them with minimal CPU overhead on subsequent forward passes. The lifecycle:

1. **Warm-up**: run the forward pass once to determine tensor shapes and identify capturable operations.
2. **Pre-capture**: execute non-capturable operations (embedding lookups, mask generation, position IDs) that involve host-device transfers.
3. **Begin capture**: call cudaStreamBeginCapture on the engine's GPU stream.
4. **Record**: execute all capturable instructions. GPU operations are recorded into the graph rather than executed immediately.
5. **End capture**: call cudaStreamEndCapture to produce a cudaGraph_t.
6. **Instantiate**: call cudaGraphInstantiate to produce a cudaGraphExec_t ready for replay.
7. **Replay**: on each subsequent forward pass, execute pre-capture ops, then call cudaGraphLaunch to replay the captured graph. Post-capture ops execute after replay.

Operations are classified as non-capturable when they perform CPU work (reading tensor data, allocating CPU tensors) or host-device memory copies during execution. The `EngineProxy` allows swapping the underlying engine during capture to redirect operations through the recording path.

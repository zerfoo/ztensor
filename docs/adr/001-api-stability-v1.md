# ADR-001: API Stability Contract for ztensor v1.0.0

**Status:** Accepted
**Date:** 2026-03-29
**Authors:** David Ndungu

## Context

The `ztensor` module (`github.com/zerfoo/ztensor`) is the tensor, compute, and graph foundation for the Zerfoo ML framework. Downstream consumers — primarily `github.com/zerfoo/zerfoo` — depend heavily on its exported surface. Before tagging v1.0.0 we need a clear contract defining which APIs are covered by the Go compatibility promise (no breaking changes without a v2 major version bump) and which may evolve in minor releases.

## Decision

### Stable Surface (v1 compatibility guarantee)

The following packages and their exported symbols are **stable**. Breaking changes to these APIs require a v2 major version.

#### `compute` — Engine interface and CPU/GPU implementations

| Symbol | Kind | Description |
|--------|------|-------------|
| `Engine[T]` | interface | Core computation engine — all methods |
| `CPUEngine[T]` | struct | CPU engine implementation |
| `NewCPUEngine[T]` | constructor | |
| `GPUEngine[T]` | struct | GPU (CUDA) engine implementation |
| `NewGPUEngine[T]` | constructor | |
| `EngineProxy[T]` | struct | Engine wrapper/proxy |
| `NewEngineProxy[T]` | constructor | |
| `DType` | type | Data type enum |
| `DTypeF32`, etc. | constants | DType values |
| `DefaultMaxAllocBytes` | constant | Default memory limit |
| `ErrMemoryLimitExceeded` | var | Sentinel error |

The following `compute` symbols are exported but **not** covered by the v1 stability guarantee (they support specialised kernel paths, GPU internals, or testing infrastructure):

| Symbol | Kind | Reason unstable |
|--------|------|-----------------|
| `FusedAddRMSNormProvider[T]` | interface | Fusion provider — kernel interface may evolve |
| `FusedNormAddProvider[T]` | interface | Fusion provider |
| `FusedQKNormRoPEProvider[T]` | interface | Fusion provider |
| `FusedRMSNormer` | interface | Fusion provider |
| `FusedRoPEProvider[T]` | interface | Fusion provider |
| `FusedScaledSoftmaxProvider[T]` | interface | Fusion provider |
| `FusedSwiGLUProvider[T]` | interface | Fusion provider |
| `FusedRMSNorm` | func | Standalone fused op |
| `FusedRoPE` | func | Standalone fused op |
| `FusedSiLUGate` | func | Standalone fused op |
| `FlashDecode` | func | Flash attention kernel entry point |
| `FlashDecodeSplitKV` | func | Flash attention kernel entry point |
| `GPUArgmaxer` | interface | GPU-specific capability |
| `GPUStreamAccessor` | interface | GPU-specific capability |
| `PagedGQAer` | interface | GPU-specific capability |
| `PoolResetter` | interface | GPU-specific capability |
| `StreamProvider` | interface | GPU-specific capability |
| `WeightUploader` | interface | GPU-specific capability |
| `FP16ToF32Converter` | interface | GPU-specific capability |
| `TransposeBMatMuler[T]` | interface | GPU-specific capability |
| `W4A16MatMuler[T]` | interface | Quantisation kernel interface |
| `W4A16Precision` | struct | Quantisation detail |
| `W4A16Info[T]` | func | Quantisation detail |
| `IsW4A16[T]` | func | Quantisation detail |
| `MatMulW4A16[T]` | func | Quantisation kernel |
| `TryW4A16MatMul[T]` | func | Quantisation kernel |
| `DequantW4ToFP16` | func | Quantisation kernel |
| `ComputeAmax[T]` | func | FP8 helper |
| `ScaleForFP8[T]` | func | FP8 helper |
| `QuantFormat[T]` | func | Quantisation helper |
| `HadamardMatrix[T]` | func | Specialised math |
| `TernaryGEMV` | func | Ternary kernel |
| `TernaryGEMVGPU` | func | Ternary GPU kernel |
| `HardwareProfile` | struct | Hardware profiling |
| `ProfileHardware` | func | Hardware profiling |
| `MemoryTracker` | struct | Memory tracking |
| `NewMemoryTracker` | constructor | Memory tracking |
| `TensorArena` | struct | Arena allocator |
| `TensorPool[T]` | struct | Pool allocator |
| `NewTensorPool[T]` | constructor | Pool allocator |
| `FailableTensor[T]` | struct | Testing utility |
| `NewFailableTensor[T]` | constructor | Testing utility |
| `FailableZeroer[T]` | struct | Testing utility |
| `NewFailableZeroer[T]` | constructor | Testing utility |
| `TestableEngine[T]` | struct | Testing utility |
| `NewTestableEngine[T]` | constructor | Testing utility |
| `TraceRecorder[T]` | interface | Tracing/debugging |
| `TracedOp` | struct | Tracing/debugging |
| `Tracer[T]` | struct | Tracing/debugging |
| `NewTracer[T]` | constructor | Tracing/debugging |

#### `tensor` — Tensor types and storage

| Symbol | Kind | Description |
|--------|------|-------------|
| `Numeric` | interface constraint | Core type constraint for all numeric types |
| `Float` | interface constraint | Floating-point subset of Numeric |
| `Addable` | interface constraint | Types supporting addition |
| `TensorNumeric[T]` | struct | Primary tensor type |
| `New[T]` | constructor | Create tensor from data |
| `NewFromBytes[T]` | constructor | Create tensor from bytes |
| `NewWithStorage[T]` | constructor | Create tensor with custom storage |
| `ToCPU[T]` | func | Transfer tensor to CPU |
| `ToGPU[T]` | func | Transfer tensor to GPU |
| `ToGPUDevice[T]` | func | Transfer tensor to specific GPU |
| `Tensor` | interface | Non-generic tensor interface |
| `NewFromType` | constructor | Create tensor from reflect.Type |
| `TensorBool` | struct | Boolean tensor |
| `NewBool` | constructor | |
| `TensorString` | struct | String tensor |
| `NewString` | constructor | |
| `Storage[T]` | interface | Storage backend interface |
| `CPUStorage[T]` | struct | CPU storage |
| `NewCPUStorage[T]` | constructor | |
| `GPUStorage[T]` | struct | GPU storage |
| `NewGPUStorage[T]`, `NewGPUStorageFromSlice[T]`, etc. | constructors | GPU storage constructors |
| `Equals[T]` | func | Tensor equality |
| `AssertClose[T]` | func | Testing helper |
| `AssertEquals[T]` | func | Testing helper |
| `BroadcastShapes` | func | Shape broadcasting |
| `BroadcastIndex` | func | Index broadcasting |
| `SameShape`, `ShapesEqual` | funcs | Shape comparison |
| `Product` | func | Shape product |
| `ConvertInt64ToInt`, `ConvertIntToInt64` | funcs | Index conversion |

The following `tensor` symbols are exported but **not** covered by the v1 stability guarantee (quantisation storage types, mmap internals, GGML type enums):

| Symbol | Kind | Reason unstable |
|--------|------|-----------------|
| `GGMLType` | type + constants | GGML format detail |
| `Q4Storage`, `Q4KStorage`, `Q5KStorage`, `Q5_0Storage`, `Q6KStorage`, `Q8Storage` | structs | Quantisation storage — format may evolve |
| `IQ2XXSStorage`, `IQ3SStorage`, `IQ4NLStorage` | structs | Quantisation storage |
| `AWQStorage`, `GPTQStorage`, `NF4Storage`, `NVFloat4Storage` | structs | Quantisation storage |
| `W8A8Storage` | struct | Quantisation storage |
| `Float16Storage`, `BFloat16Storage` | structs | Precision storage |
| `FP8E4M3Storage`, `FP8E5M2Storage` | structs | FP8 storage |
| `TernaryStorage` | struct | Ternary storage |
| `MmapStorage` | struct | Mmap-backed storage |
| `Dequantizer` | interface | Quantisation registry |
| `RegisterQuantType`, `GetQuantType`, `ListQuantTypes` | funcs | Quantisation registry |
| `DequantizeQ4K`, `DequantizeQ5K`, `DequantizeQ5_0`, `DequantizeQ6K`, `DequantizeIQ3S`, `DequantizeIQ4NL` | funcs | Dequantisation |
| `QuantizeQ4`, `QuantizeQ8`, `QuantizeAWQ`, `QuantizeGPTQ`, `QuantizeW8A8` | funcs | Quantisation |
| `GemmW8A8`, `GemmW8A8NT`, `GemmF32W8A8NT` | funcs | Quantised GEMM kernels |
| `IQ4NLTable` | var | Lookup table |
| `Mmap`, `MmapFile`, `Munmap` | funcs | Memory-mapped I/O |
| `MadviseSequential`, `MadviseRandom`, `MadviseWillNeed`, `MadviseDontNeed` | funcs | Madvise helpers |
| `Float32ToBytes`, `Int8ToBytes`, `Uint8ToBytes` | funcs | Byte conversion |
| `Q4GPUDataOffset`, `Q4GPUScaleOffset` | funcs | GPU quantisation layout |
| `MergeQ4Storage`, `MergeQ4KStorage`, `MergeQ6KStorage`, `MergeIQ4NLStorage` | funcs | Storage merging |

#### `device` — Device abstraction

| Symbol | Kind | Description |
|--------|------|-------------|
| `Device` | interface | Device abstraction — all methods |
| `Get` | func | Device lookup by ID |
| `Type` | type | Device type enum |
| `CPU` | constant | CPU device type |
| `Allocator` | interface | Memory allocator |
| `NewCPUAllocator` | constructor | |
| `NewCUDAAllocator` | constructor | |

All symbols in `device` are stable.

#### `numeric` — Arithmetic operations

| Symbol | Kind | Description |
|--------|------|-------------|
| `Arithmetic[T]` | interface | Core arithmetic interface |
| `Float32Ops` | struct | float32 arithmetic |
| `Float64Ops` | struct | float64 arithmetic |
| `Float16Ops` | struct | float16 arithmetic |
| `BFloat16Ops` | struct | bfloat16 arithmetic |
| `Float8Ops` | struct | float8 arithmetic |
| `Int8Ops` | struct | int8 arithmetic |
| `IntOps` | struct | int arithmetic |
| `Uint8Ops` | struct | uint8 arithmetic |
| `QuantizationConfig` | struct | Quantisation parameters |
| `ComputeQuantizationParams` | func | Compute quantisation scale/zero-point |
| `NewQuantizationConfig` | func | |
| `Pack4BitSlice`, `Unpack4BitSlice` | funcs | 4-bit packing |
| `Pack4BitWeights`, `Unpack4BitWeights` | funcs | 4-bit packing |

The following `numeric` symbols are testing utilities and **not** covered by the v1 stability guarantee:

| Symbol | Kind | Reason unstable |
|--------|------|-----------------|
| `TestArithmeticOp[T]` | func | Testing helper |
| `TestUnaryOp[T]` | func | Testing helper |
| `TestLeakyReLUOp[T]` | func | Testing helper |
| `TestSumOp[T]` | func | Testing helper |
| `ArithmeticTestCase[T]` | struct | Testing helper |
| `UnaryTestCase[T]` | struct | Testing helper |
| `LeakyReLUTestCase[T]` | struct | Testing helper |
| `SumTestCase[T]` | struct | Testing helper |
| `Float16TestData` | func | Testing helper |
| `Float8TestData` | func | Testing helper |

### Explicitly Unstable Packages

The following packages are **not** covered by the v1 stability guarantee. They may change in minor versions.

| Package | Reason |
|---------|--------|
| `internal/*` | Go internal convention — not importable outside module |
| `graph/` | Computation graph compilation pipeline is still evolving |
| `graph/kv/` | KV cache management — API actively changing |
| `batched/` | Batched multi-model inference — new, API not yet settled |
| `gguf/` | GGUF writer — low-level format utility |
| `log/` | Logging abstraction — may be replaced |
| `metrics/` | Metrics and correlation functions |
| `metrics/runtime/` | Runtime metrics collection |
| `types/` | Shared types (e.g. `BackwardMode`) — may be reorganised |

These packages are used by `github.com/zerfoo/zerfoo` and must remain exported, but their APIs may change in ztensor v1.x minor releases.

## Consequences

1. **v1.0.0 tag** — Once tagged, the five stable packages (`compute`, `tensor`, `device`, `numeric`, and their stable symbols as listed above) follow Go module compatibility: no breaking changes until v2.
2. **Unstable packages** — Consumers of `graph`, `batched`, `gguf`, `log`, `metrics`, and `types` should pin to exact ztensor versions and expect potential breakage on minor upgrades.
3. **Documentation** — Unstable symbols in stable packages carry a doc comment: `// This API is not covered by the v1 stability guarantee.`
4. **Future promotion** — As unstable packages mature (especially `graph/`), they may be promoted to stable in a future minor release. Promotion is additive and never breaking.

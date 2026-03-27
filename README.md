# ztensor

[![CI](https://github.com/zerfoo/ztensor/actions/workflows/ci.yml/badge.svg)](https://github.com/zerfoo/ztensor/actions/workflows/ci.yml)
[![CI](https://github.com/zerfoo/ztensor/actions/workflows/ci.yml/badge.svg)](https://github.com/zerfoo/ztensor/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/zerfoo/ztensor.svg)](https://pkg.go.dev/github.com/zerfoo/ztensor)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

GPU-accelerated tensor, compute engine, and computation graph library for Go. Zero CGo.

Part of the [Zerfoo](https://github.com/zerfoo) ML ecosystem.

## Features

- **Multi-type tensors** with compile-time type safety via Go generics (`float32`, `float64`, `float16`, `bfloat16`, `float8`, integer types)
- **GPU backends** — CUDA (cuBLAS, cuDNN, TensorRT, custom kernels), ROCm (HIP, rocBLAS, MIOpen), and OpenCL (CLBlast), all loaded dynamically via purego (zero CGo)
- **Computation graphs** with fusion passes and CUDA graph capture for optimized inference
- **CPU SIMD** — ARM NEON and x86 AVX2 hand-written assembly for GEMM, RMSNorm, RoPE, SiLU, softmax
- **Memory management** — arena-based GPU memory pools with O(1) per-pass reclamation
- **Quantized storage** — FP8 E4M3/E5M2, FP16, BFloat16 tensor storage with automatic dequantization

## Installation

```bash
go get github.com/zerfoo/ztensor
```

No CGo required. GPU backends are discovered and loaded at runtime via `dlopen`/purego.

## Quick Start

```go
package main

import (
    "context"
    "fmt"

    "github.com/zerfoo/ztensor/compute"
    "github.com/zerfoo/ztensor/numeric"
    "github.com/zerfoo/ztensor/tensor"
)

func main() {
    ctx := context.Background()

    // Create a CPU compute engine for float32
    eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})

    // Create two tensors
    a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
    b, _ := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})

    // Matrix multiplication
    c, _ := eng.MatMul(ctx, a, b)
    fmt.Println(c.Shape()) // [2, 2]
    fmt.Println(c.Data())  // [22 28 49 64]

    // Element-wise operations
    x, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
    y, _ := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})
    sum, _ := eng.Add(ctx, x, y)
    fmt.Println(sum.Data()) // [6 8 10 12]
}
```

## GPU Backend Example

GPU libraries are loaded at runtime via purego — no CGo, no build tags, no linking. If CUDA/ROCm/OpenCL is not available, the engine constructor returns an error and you fall back to CPU.

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/zerfoo/ztensor/compute"
    "github.com/zerfoo/ztensor/numeric"
    "github.com/zerfoo/ztensor/tensor"
)

func main() {
    ctx := context.Background()

    // Try CUDA first, fall back to CPU
    eng, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
    if err != nil {
        fmt.Println("CUDA not available, using CPU:", err)
        cpuEng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
        run(ctx, cpuEng)
        return
    }
    run(ctx, eng)
}

func run(ctx context.Context, eng compute.Engine[float32]) {
    a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
    b, _ := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
    c, _ := eng.MatMul(ctx, a, b)
    fmt.Println(c.Data()) // [22 28 49 64]
}
```

Other GPU backends follow the same pattern:

```go
// ROCm (AMD GPUs)
eng, err := compute.NewROCmEngine[float32](numeric.Float32Ops{})

// OpenCL (cross-vendor)
eng, err := compute.NewOpenCLEngine[float32](numeric.Float32Ops{})
```

## Type Safety with Generics

The `tensor.Numeric` type constraint ensures compile-time type safety across all supported numeric types:

```go
// Works with any Numeric type
func dotProduct[T tensor.Numeric](eng compute.Engine[T], a, b *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
    return eng.MatMul(context.Background(), a, b)
}
```

Supported types include `float32`, `float64`, `float16.Float16`, `float16.BFloat16`, `float8.Float8`, and all Go integer types.

## Use Cases

- **ML inference engines** — ztensor powers the [zerfoo](https://github.com/zerfoo/zerfoo) inference runtime for transformer models
- **Scientific computing** — GPU-accelerated linear algebra with automatic backend selection
- **GPU compute from Go** — use CUDA/ROCm/OpenCL from pure Go without CGo or build tags
- **Custom ML operators** — build neural network layers on top of the `compute.Engine` interface

## Package Overview

| Package | Description |
|---------|-------------|
| `tensor/` | Multi-type tensor storage — CPU, GPU, quantized (FP8, FP16, BFloat16) |
| `compute/` | Compute engine interface with CPU, CUDA, ROCm, and OpenCL implementations |
| `graph/` | Computation graph compiler with operator fusion and CUDA graph capture |
| `numeric/` | Type-safe `Arithmetic[T]` interface for all numeric types |
| `device/` | Device abstraction and memory allocators |
| `types/` | Shared type definitions |
| `log/` | Structured logging interface |
| `metrics/` | Performance metrics and profiling |
| `internal/cuda/` | Zero-CGo CUDA runtime bindings via purego, 25+ custom kernels |
| `internal/xblas/` | ARM NEON and x86 AVX2 SIMD assembly (GEMM, RMSNorm, RoPE, SiLU, softmax) |
| `internal/gpuapi/` | GPU Runtime Abstraction Layer — unified adapter for CUDA, ROCm, OpenCL |
| `internal/codegen/` | Megakernel code generator |

## Dependencies

ztensor depends on:

- [float16](https://github.com/zerfoo/float16) — IEEE 754 half-precision and BFloat16 arithmetic
- [float8](https://github.com/zerfoo/float8) — FP8 E4M3FN arithmetic for quantized inference

ztensor is used by:

- [zerfoo](https://github.com/zerfoo/zerfoo) — ML inference, training, and serving framework

## License

Apache 2.0

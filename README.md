# ztensor

GPU-accelerated tensor, compute engine, and computation graph library for Go.

Part of the [Zerfoo](https://github.com/zerfoo) ML ecosystem.

## Install

```sh
go get github.com/zerfoo/ztensor
```

## What's included

- **tensor/** - Multi-type tensor storage (CPU, GPU, quantized, FP16, BF16, FP8)
- **compute/** - Engine interface with CPU, CUDA, ROCm, and OpenCL backends
- **graph/** - Computation graph compiler with fusion passes and CUDA graph capture
- **numeric/** - Type-safe arithmetic operations for all numeric types
- **device/** - Device abstraction and memory allocators
- **internal/cuda/** - Zero-CGo CUDA bindings via purego with 25+ custom kernels
- **internal/xblas/** - ARM NEON and x86 AVX2 SIMD assembly for GEMM, RMSNorm, RoPE, SiLU, softmax

## Quick start

```go
package main

import (
    "fmt"
    "github.com/zerfoo/ztensor/compute"
    "github.com/zerfoo/ztensor/numeric"
    "github.com/zerfoo/ztensor/tensor"
)

func main() {
    eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
    a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
    b, _ := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
    c, _ := eng.MatMul(a, b)
    fmt.Println(c.Shape()) // [2, 2]
    fmt.Println(c.Data())  // [22 28 49 64]
}
```

## License

Apache 2.0

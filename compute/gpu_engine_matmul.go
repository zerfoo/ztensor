package compute

// gpu_engine_matmul.go contains shared helper functions extracted from the 14
// quantized matmul methods in gpu_engine.go. Each helper captures a recurring
// pattern (upload, shape validation, GEMV output, dequant+GEMM) so that the
// individual matmul methods can be thin wrappers.

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/gpuapi"
	"github.com/zerfoo/ztensor/tensor"
)

// uploadRawBytes uploads raw quantized bytes to the GPU. If the storage already
// has a GPU pointer (via GPUPtr()), it returns that pointer with a no-op free.
// Otherwise it allocates device memory, copies the bytes, and returns a free
// function the caller must defer.
//
// gpuPtr should return (ptr, _, _) from the storage's GPUPtr() method.
// rawBytes is the byte payload to upload (e.g. qs.RawBytes()).
func (e *GPUEngine[T]) uploadRawBytes(gpuPtr unsafe.Pointer, rawBytes []byte) (devPtr unsafe.Pointer, free func(), err error) {
	if gpuPtr != nil {
		return gpuPtr, func() {}, nil
	}
	devPtr, err = e.pool.Alloc(e.deviceID, len(rawBytes))
	if err != nil {
		return nil, nil, err
	}
	free = func() { e.pool.Free(e.deviceID, devPtr, len(rawBytes)) }
	if err := e.runtime.Memcpy(devPtr, unsafe.Pointer(&rawBytes[0]), len(rawBytes), gpuapi.MemcpyHostToDevice); err != nil {
		free()
		return nil, nil, err
	}
	return devPtr, free, nil
}

// aShapeCheck2D validates shapes for A-side quantized matmul methods.
// Returns m, k, n and whether the caller should fall back to CPU.
// Falls back when either operand has <2 dims, either has >2 dims,
// or k is not a multiple of kAlignment.
func aShapeCheck2D(aShape, bShape []int, kAlignment int) (m, k, n int, fallback bool) {
	if len(aShape) < 2 || len(bShape) < 2 {
		return 0, 0, 0, true
	}
	if len(aShape) > 2 || len(bShape) > 2 {
		return 0, 0, 0, true
	}
	m = aShape[0]
	k = aShape[1]
	n = bShape[1]
	if kAlignment > 0 && k%kAlignment != 0 {
		return 0, 0, 0, true
	}
	return m, k, n, false
}

// bweightShapeMKN validates shapes for BWeight (virtual-transposed weight)
// matmul methods. It flattens A's batch dimensions and builds the output shape.
// Returns m, k, n, outShape, and whether the caller should fall back to CPU.
// Falls back when either operand has <2 dims, B has >2 dims, or k is not a
// multiple of kAlignment.
func bweightShapeMKN(aShape, bShape []int, kAlignment int) (m, k, n int, outShape []int, fallback bool) {
	if len(aShape) < 2 || len(bShape) < 2 {
		return 0, 0, 0, nil, true
	}
	if len(bShape) > 2 {
		return 0, 0, 0, nil, true
	}
	k = aShape[len(aShape)-1]
	m = 1
	for i := 0; i < len(aShape)-1; i++ {
		m *= aShape[i]
	}
	n = bShape[1]
	if kAlignment > 0 && k%kAlignment != 0 {
		return 0, 0, 0, nil, true
	}
	outShape = make([]int, len(aShape))
	copy(outShape, aShape[:len(aShape)-1])
	outShape[len(outShape)-1] = n
	return m, k, n, outShape, false
}

// quantGemvResult allocates a GPU output buffer of outElems float32s, calls
// gemvFn to fill it, and wraps the result as a tensor with the given shape.
// On any error it cleans up and returns (nil, err) so the caller can fall back.
func (e *GPUEngine[T]) quantGemvResult(outShape []int, outElems int, gemvFn func(devY unsafe.Pointer) error, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	cSize := outElems * f32Size
	devY, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return nil, err
	}
	if err := gemvFn(devY); err != nil {
		e.pool.Free(e.deviceID, devY, cSize)
		return nil, err
	}
	return makeGPUResult[T](e, outShape, devY, outElems, dst...)
}

// dequantSgemm dequantizes weight data to F32 on the GPU (or CPU as fallback),
// uploads the other operand, allocates the output, and calls cuBLAS Sgemm.
// C[m,n] = dequantA[m,k] * B[k,n].
//
// devW is the quantized device pointer. dequantFn is the GPU dequantization
// function; if nil, cpuDequant is used instead. cpuDequant produces F32 data
// on the host (may be nil if dequantFn always succeeds).
func (e *GPUEngine[T]) dequantSgemm(
	devW unsafe.Pointer, m, k, n int,
	dequantFn func(src, dst unsafe.Pointer, rows, cols int) error,
	cpuDequant func([]float32),
	b *tensor.TensorNumeric[T],
	outShape []int, name string,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	dequantSize := m * k * f32Size
	devAF32, err := e.pool.Alloc(e.deviceID, dequantSize)
	if err != nil {
		return nil, err
	}
	defer e.pool.Free(e.deviceID, devAF32, dequantSize)

	dequanted := false
	if dequantFn != nil {
		if err := dequantFn(devW, devAF32, m, k); err == nil {
			dequanted = true
		}
	}
	if !dequanted {
		if cpuDequant == nil {
			return nil, fmt.Errorf("%s: no dequant path available", name)
		}
		dequant := make([]float32, m*k)
		cpuDequant(dequant)
		if err := e.runtime.Memcpy(devAF32, unsafe.Pointer(&dequant[0]), dequantSize, gpuapi.MemcpyHostToDevice); err != nil {
			return nil, err
		}
	}

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return nil, err
	}
	defer cleanupB()

	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return nil, err
	}

	if err := e.blas.Sgemm(m, n, k, 1.0, devAF32, devB, 0.0, devC); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return nil, fmt.Errorf("%s: Sgemm: %w", name, err)
	}

	return makeGPUResult[T](e, outShape, devC, m*n, dst...)
}

// sgemmNTOrFallback dequantizes weight data to F32, then computes
// C[m,n] = A[m,k] * dequant(B)[n,k]^T using SgemmNT if available,
// or an explicit Transpose2D + Sgemm otherwise.
//
// devW is the quantized device pointer laid out as [n, k].
// dequantFn is the GPU dequantization function (may be nil).
// cpuDequant produces F32 data on the host (may be nil if dequantFn always succeeds).
func (e *GPUEngine[T]) sgemmNTOrFallback(
	devW unsafe.Pointer, m, k, n int,
	dequantFn func(src, dst unsafe.Pointer, rows, cols int) error,
	cpuDequant func([]float32),
	a *tensor.TensorNumeric[T],
	outShape []int, name string,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	dequantSize := n * k * f32Size
	devBF32, err := e.pool.Alloc(e.deviceID, dequantSize)
	if err != nil {
		return nil, err
	}
	defer e.pool.Free(e.deviceID, devBF32, dequantSize)

	dequanted := false
	if dequantFn != nil {
		if err := dequantFn(devW, devBF32, n, k); err == nil {
			dequanted = true
		}
	}
	if !dequanted {
		if cpuDequant == nil {
			return nil, fmt.Errorf("%s: no dequant path available", name)
		}
		dequant := make([]float32, n*k)
		cpuDequant(dequant)
		if err := e.runtime.Memcpy(devBF32, unsafe.Pointer(&dequant[0]), dequantSize, gpuapi.MemcpyHostToDevice); err != nil {
			return nil, err
		}
	}

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return nil, err
	}
	defer cleanupA()

	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return nil, err
	}

	// Prefer SgemmNT (avoids explicit transpose).
	if ntBLAS, ok := e.blas.(gpuapi.BLASTransposeB); ok {
		if err := ntBLAS.SgemmNT(m, n, k, 1.0, devA, devBF32, 0.0, devC); err != nil {
			e.pool.Free(e.deviceID, devC, cSize)
			return nil, fmt.Errorf("%s: SgemmNT: %w", name, err)
		}
		return makeGPUResult[T](e, outShape, devC, m*n, dst...)
	}

	// Fallback: transpose dequantized B then use Sgemm.
	devBT, err := e.pool.Alloc(e.deviceID, dequantSize)
	if err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return nil, err
	}
	defer e.pool.Free(e.deviceID, devBT, dequantSize)

	if err := e.kernels.Transpose2D(devBF32, devBT, n, k, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return nil, err
	}

	if err := e.blas.Sgemm(m, n, k, 1.0, devA, devBT, 0.0, devC); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return nil, fmt.Errorf("%s: Sgemm: %w", name, err)
	}

	return makeGPUResult[T](e, outShape, devC, m*n, dst...)
}

package compute

// gpu_bf16.go contains the native bfloat16 GPU elementwise path. Unlike the
// FP16 path (which keeps T=float32 and stores Float16Storage), the bf16 path is
// native: T == float16.BFloat16, GPUStorage[BFloat16] holds 2-byte elements,
// and the kernels operate directly on __nv_bfloat16 device memory. This mirrors
// the existing native bf16 GEMM (BFloat16Gemm) so a full bf16 graph -- matmul
// plus elementwise plus AdamW -- runs on-device without falling back to CPU.
//
// bf16 shares f32's 8-bit exponent, so these kernels do not reopen the ADR-072
// forward-conditioning cliff; only the 7-bit mantissa differs. Reductions and
// transcendentals accumulate in FP32 inside the kernels (see elementwise_bf16.cu).

import (
	"context"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/internal/gpuapi"
	"github.com/zerfoo/ztensor/tensor"
)

const bf16Size = 2 // sizeof(__nv_bfloat16)

// isBFloat16 reports whether the generic type T is float16.BFloat16.
func isBFloat16[T tensor.Numeric]() bool {
	var zero T
	_, ok := any(zero).(float16.BFloat16)

	return ok
}

// gpuBinaryOpBF16 runs a native bf16 binary kernel (c = op(a, b)) on
// same-shape, same-length operands. Buffers are 2 bytes/element; getDevicePtr
// and makeGPUResult are element-size-generic (they use unsafe.Sizeof(T)), so
// they handle bf16 correctly. Broadcasting is not handled here -- the caller
// falls back to CPU for mismatched shapes.
func gpuBinaryOpBF16[T tensor.Numeric](
	e *GPUEngine[T],
	a, b *tensor.TensorNumeric[T],
	kernelFn func(devA, devB, devC unsafe.Pointer, n int, stream gpuapi.Stream) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	e.setDevice()

	n := a.GetStorage().Len()

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return nil, err
	}
	defer cleanupA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return nil, err
	}
	defer cleanupB()

	byteSize := n * bf16Size

	devC, reused := tryReuseDstPtr[T](n, dst)
	if !reused {
		devC, err = e.pool.Alloc(e.deviceID, byteSize)
		if err != nil {
			return nil, err
		}
	}

	if err := kernelFn(devA, devB, devC, n, e.stream); err != nil {
		if !reused {
			e.pool.Free(e.deviceID, devC, byteSize)
		}

		return nil, err
	}

	if reused {
		return finishReusedDst[T](dst[0], a.Shape()), nil
	}
	return makeGPUResult[T](e, a.Shape(), devC, n, dst...)
}

// gpuSoftmaxBF16 runs a native bf16 softmax along the given axis using the
// fused scaled-softmax kernel (scale = 1.0) with FP32 max/sum accumulation.
func gpuSoftmaxBF16[T tensor.Numeric](
	e *GPUEngine[T],
	ctx context.Context,
	a *tensor.TensorNumeric[T],
	axis int,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	shape := a.Shape()
	rank := len(shape)
	if rank == 0 {
		return e.cpu.Softmax(ctx, a, axis, dst...)
	}
	if axis < 0 {
		axis = rank + axis
	}
	if axis < 0 || axis >= rank {
		return e.cpu.Softmax(ctx, a, axis, dst...)
	}

	e.setDevice()

	n := a.GetStorage().Len()
	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= shape[i]
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}
	axisSize := shape[axis]

	devIn, cleanupIn, err := getDevicePtr(e, a)
	if err != nil {
		return e.cpu.Softmax(ctx, a, axis, dst...)
	}
	defer cleanupIn()

	byteSize := n * bf16Size
	devOut, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return e.cpu.Softmax(ctx, a, axis, dst...)
	}

	if err := e.kernels.ScaledSoftmaxBF16(devIn, devOut, outer, inner, axisSize, 1.0, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, byteSize)
		return nil, err
	}

	return makeGPUResult[T](e, shape, devOut, n, dst...)
}

// gpuUnaryOpBF16 runs a native bf16 unary kernel (c = op(a)).
func gpuUnaryOpBF16[T tensor.Numeric](
	e *GPUEngine[T],
	a *tensor.TensorNumeric[T],
	kernelFn func(devA, devC unsafe.Pointer, n int, stream gpuapi.Stream) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	e.setDevice()

	n := a.GetStorage().Len()

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return nil, err
	}
	defer cleanupA()

	byteSize := n * bf16Size

	devC, reused := tryReuseDstPtr[T](n, dst)
	if !reused {
		devC, err = e.pool.Alloc(e.deviceID, byteSize)
		if err != nil {
			return nil, err
		}
	}

	if err := kernelFn(devA, devC, n, e.stream); err != nil {
		if !reused {
			e.pool.Free(e.deviceID, devC, byteSize)
		}

		return nil, err
	}

	if reused {
		return finishReusedDst[T](dst[0], a.Shape()), nil
	}
	return makeGPUResult[T](e, a.Shape(), devC, n, dst...)
}

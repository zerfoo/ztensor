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
	"fmt"
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

// execBroadcast2D runs a 2D broadcast binary kernel and builds the result.
//
// For f32 it calls the kernel directly (prior behavior). For bf16 -- which has
// no native broadcast kernel -- it converts both operands to f32 on-device,
// runs the existing f32 broadcast kernel, then converts the result back to bf16,
// all on the engine stream. This keeps bf16 broadcast ops on the GPU and
// CUDA-graph-capturable, instead of the host CPU fallback whose D2H/H2D copies
// break stream capture (e.g. QKL2Norm's Mul(x, inv)). Computing in f32 also
// matches the bf16 GEMM/reduction convention (f32 accumulation, bf16 storage).
// devA/devB are T-typed device pointers; nA/nB are the element counts of a/b.
func execBroadcast2D[T tensor.Numeric](
	e *GPUEngine[T], outShape []int,
	devA unsafe.Pointer, nA int, devB unsafe.Pointer, nB int, outElems int,
	saRow, saCol, sbRow, sbCol, mDim, dDim int,
	kernelFn func(devA, devB, devC unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream gpuapi.Stream) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	if !isBFloat16[T]() {
		devC, err := e.pool.Alloc(e.deviceID, outElems*f32Size)
		if err != nil {
			return nil, err
		}
		if err := kernelFn(devA, devB, devC, saRow, saCol, sbRow, sbCol, mDim, dDim, e.stream); err != nil {
			e.pool.Free(e.deviceID, devC, outElems*f32Size)
			return nil, err
		}
		return makeGPUResult[T](e, outShape, devC, outElems, dst...)
	}

	// bf16: a,b -> f32 scratch, f32 broadcast, result -> bf16. Scratch is freed
	// after the kernels are enqueued on the engine stream (arena frees are
	// stream-ordered, matching the getDevicePtr FP16->F32 scratch pattern).
	aF, err := e.pool.Alloc(e.deviceID, nA*f32Size)
	if err != nil {
		return nil, err
	}
	defer e.pool.Free(e.deviceID, aF, nA*f32Size)
	if err := e.kernels.BF16ToF32(devA, aF, nA, e.stream); err != nil {
		return nil, err
	}
	bF, err := e.pool.Alloc(e.deviceID, nB*f32Size)
	if err != nil {
		return nil, err
	}
	defer e.pool.Free(e.deviceID, bF, nB*f32Size)
	if err := e.kernels.BF16ToF32(devB, bF, nB, e.stream); err != nil {
		return nil, err
	}
	cF, err := e.pool.Alloc(e.deviceID, outElems*f32Size)
	if err != nil {
		return nil, err
	}
	defer e.pool.Free(e.deviceID, cF, outElems*f32Size)
	if err := kernelFn(aF, bF, cF, saRow, saCol, sbRow, sbCol, mDim, dDim, e.stream); err != nil {
		return nil, err
	}
	cB, err := e.pool.Alloc(e.deviceID, outElems*bf16Size)
	if err != nil {
		return nil, err
	}
	if err := e.kernels.F32ToBF16(cF, cB, outElems, e.stream); err != nil {
		e.pool.Free(e.deviceID, cB, outElems*bf16Size)
		return nil, err
	}
	return makeGPUResult[T](e, outShape, cB, outElems, dst...)
}

// bf16ScalarToF32 converts a bf16 scalar (carried as T) to float32. Used by the
// bf16 scalar-op path, where toFloat32 (an any.(float32) assertion) would panic.
func bf16ScalarToF32[T tensor.Numeric](v T) float32 {
	return any(v).(float16.BFloat16).ToFloat32()
}

// gpuScalarOpBF16 runs a scalar kernel (c = op(a, scalar)) for bf16 by converting
// a to f32 on-device, running the f32 scalar kernel, and converting the result
// back to bf16 -- keeping the op on the GPU and capture-safe (the CPU fallback's
// host copies break CUDA-graph capture, e.g. QKL2Norm's AddScalar(eps)).
func gpuScalarOpBF16[T tensor.Numeric](
	e *GPUEngine[T], a *tensor.TensorNumeric[T], scalar float32,
	kernelFn func(devA unsafe.Pointer, scalar float32, devC unsafe.Pointer, n int, stream gpuapi.Stream) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	e.setDevice()
	n := a.GetStorage().Len()
	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return nil, err
	}
	defer cleanupA()
	aF, err := e.pool.Alloc(e.deviceID, n*f32Size)
	if err != nil {
		return nil, err
	}
	defer e.pool.Free(e.deviceID, aF, n*f32Size)
	if err := e.kernels.BF16ToF32(devA, aF, n, e.stream); err != nil {
		return nil, err
	}
	cF, err := e.pool.Alloc(e.deviceID, n*f32Size)
	if err != nil {
		return nil, err
	}
	defer e.pool.Free(e.deviceID, cF, n*f32Size)
	if err := kernelFn(aF, scalar, cF, n, e.stream); err != nil {
		return nil, err
	}
	cB, err := e.pool.Alloc(e.deviceID, n*bf16Size)
	if err != nil {
		return nil, err
	}
	if err := e.kernels.F32ToBF16(cF, cB, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, cB, n*bf16Size)
		return nil, err
	}
	return makeGPUResult[T](e, a.Shape(), cB, n, dst...)
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

// gpuFusedAddRMSNormBF16 runs the native bf16 fused add+RMSNorm kernel
// (sum = input+residual, normed = rmsnorm(sum)*weight) with FP32 reductions.
// It is the bf16 analogue of the f32 GPUFusedAddRMSNorm body. bf16 buffers are
// 2 bytes/element; getDevicePtr/makeGPUResult are element-size-generic.
func gpuFusedAddRMSNormBF16[T tensor.Numeric](
	e *GPUEngine[T],
	input, residual, weight *tensor.TensorNumeric[T],
	eps float32,
) (normed *tensor.TensorNumeric[T], residualOut *tensor.TensorNumeric[T], scales *tensor.TensorNumeric[T], err error) {
	inShape := input.Shape()
	if len(inShape) < 2 {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm(bf16): input must be at least 2D, got %v", inShape)
	}
	D := inShape[len(inShape)-1]
	rows := 1
	for i := 0; i < len(inShape)-1; i++ {
		rows *= inShape[i]
	}

	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm(bf16) input: %w", err)
	}
	defer inCleanup()

	resPtr, resCleanup, err := getDevicePtr(e, residual)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm(bf16) residual: %w", err)
	}
	defer resCleanup()

	wPtr, wCleanup, err := getDevicePtr(e, weight)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm(bf16) weight: %w", err)
	}
	defer wCleanup()

	outElems := rows * D
	outBytes := outElems * bf16Size
	e.setDevice()
	devNormed, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm(bf16) alloc normed: %w", err)
	}
	devSum, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		e.pool.Free(e.deviceID, devNormed, outBytes)
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm(bf16) alloc sum: %w", err)
	}

	if err := e.kernels.FusedAddRMSNormBF16(inPtr, resPtr, wPtr, devNormed, devSum, eps, rows, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, devNormed, outBytes)
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, err
	}

	normed, err = makeGPUResult[T](e, inShape, devNormed, outElems)
	if err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, err
	}
	residualOut, err = makeGPUResult[T](e, inShape, devSum, outElems)
	if err != nil {
		return nil, nil, nil, err
	}
	return normed, residualOut, nil, nil
}

// gpuFusedNormAddBF16 runs the native bf16 fused RMSNorm+add kernel
// (output = rmsnorm(input)*weight + residual) with FP32 reductions. The bf16
// analogue of the f32 GPUFusedNormAdd body.
func gpuFusedNormAddBF16[T tensor.Numeric](
	e *GPUEngine[T],
	input, weight, residual *tensor.TensorNumeric[T],
	eps float32,
) (*tensor.TensorNumeric[T], error) {
	inShape := input.Shape()
	if len(inShape) < 2 {
		return nil, fmt.Errorf("GPUFusedNormAdd(bf16): input must be at least 2D, got %v", inShape)
	}
	D := inShape[len(inShape)-1]
	rows := 1
	for i := 0; i < len(inShape)-1; i++ {
		rows *= inShape[i]
	}

	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd(bf16) input: %w", err)
	}
	defer inCleanup()

	wPtr, wCleanup, err := getDevicePtr(e, weight)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd(bf16) weight: %w", err)
	}
	defer wCleanup()

	resPtr, resCleanup, err := getDevicePtr(e, residual)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd(bf16) residual: %w", err)
	}
	defer resCleanup()

	outElems := rows * D
	outBytes := outElems * bf16Size
	e.setDevice()
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd(bf16) alloc: %w", err)
	}

	if err := e.kernels.FusedNormAddBF16(inPtr, wPtr, resPtr, devOut, eps, rows, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}
	return makeGPUResult[T](e, inShape, devOut, outElems)
}

// gpuFusedQKNormRoPEBF16 runs the native bf16 fused per-head RMSNorm+RoPE kernel
// with FP32 reductions and RoPE arithmetic. The bf16 analogue of the f32
// GPUFusedQKNormRoPE body. cos/sin are bf16 (the engine's generic tensor type).
func gpuFusedQKNormRoPEBF16[T tensor.Numeric](
	e *GPUEngine[T],
	input, weightQ, weightK, cosAngles, sinAngles *tensor.TensorNumeric[T],
	eps float32,
	totalHeads, headDim, numQHeads, halfRotary int,
) (*tensor.TensorNumeric[T], error) {
	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE(bf16) input: %w", err)
	}
	defer inCleanup()

	wqPtr, wqCleanup, err := getDevicePtr(e, weightQ)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE(bf16) weightQ: %w", err)
	}
	defer wqCleanup()

	wkPtr, wkCleanup, err := getDevicePtr(e, weightK)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE(bf16) weightK: %w", err)
	}
	defer wkCleanup()

	cosPtr, cosCleanup, err := getDevicePtr(e, cosAngles)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE(bf16) cos: %w", err)
	}
	defer cosCleanup()

	sinPtr, sinCleanup, err := getDevicePtr(e, sinAngles)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE(bf16) sin: %w", err)
	}
	defer sinCleanup()

	outElems := totalHeads * headDim
	outBytes := outElems * bf16Size
	e.setDevice()
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE(bf16) alloc: %w", err)
	}

	if err := e.kernels.FusedQKNormRoPEBF16(inPtr, wqPtr, wkPtr, cosPtr, sinPtr, devOut, eps, totalHeads, headDim, numQHeads, halfRotary, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}
	return makeGPUResult[T](e, []int{totalHeads, headDim}, devOut, outElems)
}

// gpuSumAxisBF16 runs a native bf16 axis reduction (sum along `axis`) using the
// FP32-accumulating SumAxisBF16 kernel. It is the bf16 analogue of the f32
// gpuSum body and mirrors its axis-normalization, keepDims/squeeze output-shape
// logic, and CPU fallbacks exactly. The kernel accumulates each axis stripe in
// FP32 and rounds the result to bf16; for axis-sized reductions this is more
// accurate than pairwise bf16 addition but still only carries bf16's 7-bit
// mantissa, so callers must tolerate a few bf16 steps vs an f64 reference.
//
// invDivisor scales the per-stripe FP32 sum before the bf16 round: pass 1.0 for
// a plain sum (gpuSum/gpuReduceSum) or 1/axisSize for a mean (gpuReduceMean).
// On every CPU fallback the divide is reapplied (cpu.ReduceMean) so the contract
// holds identically for both sum and mean.
func gpuSumAxisBF16[T tensor.Numeric](
	e *GPUEngine[T],
	ctx context.Context,
	a *tensor.TensorNumeric[T],
	axis int,
	keepDims bool,
	invDivisor float32,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, fmt.Errorf("Sum: input tensor must not be nil")
	}

	e.setDevice()

	// cpuFallback computes the same reduction on the CPU engine: a plain sum when
	// invDivisor == 1, otherwise the mean (so the divide-folding contract holds
	// identically on every fallback path).
	cpuFallback := func() (*tensor.TensorNumeric[T], error) {
		if invDivisor == 1.0 {
			return e.cpu.Sum(ctx, a, axis, keepDims, dst...)
		}
		return e.cpu.ReduceMean(ctx, a, axis, keepDims, dst...)
	}

	// Negative axis falls back to CPU, matching the f32 gpuSum contract.
	if axis < 0 {
		return cpuFallback()
	}

	shape := a.Shape()
	rank := len(shape)

	if axis >= rank {
		return nil, fmt.Errorf("Sum: axis %d out of bounds for %d dimensions", axis, rank)
	}

	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= shape[i]
	}

	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}

	axisSize := shape[axis]
	numStripes := outer * inner

	var newShape []int
	if keepDims {
		newShape = make([]int, rank)
		for i, d := range shape {
			if i == axis {
				newShape[i] = 1
			} else {
				newShape[i] = d
			}
		}
	} else {
		for i, d := range shape {
			if i != axis {
				newShape = append(newShape, d)
			}
		}
		if len(newShape) == 0 {
			newShape = []int{1}
		}
	}

	devIn, cleanupIn, err := getDevicePtr(e, a)
	if err != nil {
		return cpuFallback()
	}
	defer cleanupIn()

	outByteSize := numStripes * bf16Size

	// Reuse dst's existing GPU memory when possible (mirrors f32 gpuSum #84).
	devOut, reused := tryReuseDstPtr[T](numStripes, dst)
	if !reused {
		devOut, err = e.pool.Alloc(e.deviceID, outByteSize)
		if err != nil {
			return cpuFallback()
		}
	}

	if err := e.kernels.SumAxisBF16(devIn, devOut, outer, inner, axisSize, invDivisor, e.stream); err != nil {
		if !reused {
			e.pool.Free(e.deviceID, devOut, outByteSize)
		}

		return nil, err
	}

	if reused {
		return finishReusedDst[T](dst[0], newShape), nil
	}
	return makeGPUResult[T](e, newShape, devOut, numStripes, dst...)
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

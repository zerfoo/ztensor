package compute

// gpu_engine_reduction.go contains reduction operations (Softmax, Sum,
// ReduceSum, ReduceMax, ReduceMean), argmax, and fused scaled-softmax
// kernels for GPUEngine.

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/gpuapi"
	"github.com/zerfoo/ztensor/tensor"
)

// Sum computes the sum of elements along an axis.
func (e *GPUEngine[T]) Sum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuSum(ctx, a, axis, keepDims, dst...)
}

// Softmax applies the softmax function along an axis.
func (e *GPUEngine[T]) Softmax(ctx context.Context, a *tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuSoftmax(ctx, a, axis, dst...)
}

// ReduceSum computes the sum of elements along an axis.
func (e *GPUEngine[T]) ReduceSum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuReduceSum(ctx, a, axis, keepDims, dst...)
}

// ReduceMax computes the maximum of elements along an axis (CPU fallback).
func (e *GPUEngine[T]) ReduceMax(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.ReduceMax(ctx, a, axis, keepDims, dst...)
}

// ReduceMean computes the mean of elements along an axis.
func (e *GPUEngine[T]) ReduceMean(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuReduceMean(ctx, a, axis, keepDims, dst...)
}

// GPUArgmax finds the index of the maximum element in a GPU-resident float32 tensor.
// Returns the index as an int without copying the full tensor to the host.
// Only copies back a single int32 (4 bytes) instead of the entire tensor.
func (e *GPUEngine[T]) GPUArgmax(t *tensor.TensorNumeric[float32]) (int, error) {
	gs, ok := t.GetStorage().(*tensor.GPUStorage[float32])
	if !ok {
		return 0, fmt.Errorf("GPUArgmax: tensor not GPU-resident")
	}

	e.setDevice()

	n := gs.Len()
	devInput := gs.Ptr()

	// Allocate scratch: 2 * ceil(n/256) * 4 bytes (blockVals + blockIdxs).
	numBlocks := (n + 255) / 256
	scratchSize := 2 * numBlocks * 4
	devScratch, err := e.pool.Alloc(e.deviceID, scratchSize)
	if err != nil {
		return 0, fmt.Errorf("GPUArgmax: scratch alloc: %w", err)
	}
	defer e.pool.Free(e.deviceID, devScratch, scratchSize)

	// Allocate device result (single int32).
	devResult, err := e.pool.Alloc(e.deviceID, 4)
	if err != nil {
		return 0, fmt.Errorf("GPUArgmax: result alloc: %w", err)
	}
	defer e.pool.Free(e.deviceID, devResult, 4)

	if err := e.kernels.Argmax(devInput, devResult, devScratch, n, e.stream); err != nil {
		return 0, fmt.Errorf("GPUArgmax: %w", err)
	}

	// Copy single int32 result back to host.
	var result int32
	if err := e.runtime.Memcpy(unsafe.Pointer(&result), devResult, 4, gpuapi.MemcpyDeviceToHost); err != nil {
		return 0, fmt.Errorf("GPUArgmax: D2H copy: %w", err)
	}

	return int(result), nil
}

// GPUScaledSoftmax computes softmax(input * scale) in a single GPU kernel launch.
// This replaces MulScalar + Softmax (2 kernel launches) with 1, saving 26 launches
// per token for 26 transformer layers.
func (e *GPUEngine[T]) GPUScaledSoftmax(input *tensor.TensorNumeric[T], scale float32, axis int) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return nil, fmt.Errorf("GPUScaledSoftmax: only float32 supported")
	}

	e.setDevice()

	if input == nil {
		return nil, fmt.Errorf("GPUScaledSoftmax: input tensor must not be nil")
	}

	shape := input.Shape()
	rank := len(shape)

	if rank == 0 {
		return nil, fmt.Errorf("GPUScaledSoftmax: scalar tensors not supported")
	}

	if axis < 0 {
		axis = rank + axis
	}

	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("GPUScaledSoftmax: axis %d out of bounds for %d dimensions", axis, rank)
	}

	n := input.GetStorage().Len()

	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= shape[i]
	}

	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}

	axisSize := shape[axis]

	// FP16 paths — skip entirely for F32 compute.
	if e.dtype != DTypeF32 {
		// Native FP16 path: input already has Float16Storage on GPU — no conversion needed.
		if fs, ok := any(input.GetStorage()).(*tensor.Float16Storage); ok {
			return fp16ScaledSoftmaxNative(e, fs, input.Shape(), scale, outer, inner, axisSize)
		}
		// FP16 path: convert to FP16, run FP16 scaled softmax, convert back.
		return fp16ScaledSoftmax(e, input, scale, outer, inner, axisSize)
	}

	devIn, cleanupIn, err := getDevicePtr(e, input)
	if err != nil {
		return nil, fmt.Errorf("GPUScaledSoftmax input: %w", err)
	}
	defer cleanupIn()

	byteSize := n * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return nil, fmt.Errorf("GPUScaledSoftmax alloc: %w", err)
	}

	if err := e.kernels.ScaledSoftmaxF32(devIn, devOut, outer, inner, axisSize, scale, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, byteSize)
		return nil, err
	}

	return makeGPUResult[T](e, shape, devOut, n)
}

// GPUFusedSoftmaxVMul computes softmax(scores * scale) @ V in a single GPU
// kernel launch. Decode-optimized (seqQ=1): avoids materializing the attention
// weights tensor, saving one kernel launch and the associated memory traffic.
// scores: [BH, 1, seqKV], V: [BH, seqKV, D]. Returns output: [BH, 1, D].
func (e *GPUEngine[T]) GPUFusedSoftmaxVMul(scores, V *tensor.TensorNumeric[T], scale float32) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return nil, fmt.Errorf("GPUFusedSoftmaxVMul: only float32 supported")
	}

	if scores == nil || V == nil {
		return nil, fmt.Errorf("GPUFusedSoftmaxVMul: input tensors must not be nil")
	}

	e.setDevice()

	sShape := scores.Shape()
	vShape := V.Shape()

	// scores must be [BH, 1, seqKV] or [BH, seqKV]
	var BH, seqKV int
	switch len(sShape) {
	case 3:
		if sShape[1] != 1 {
			return nil, fmt.Errorf("GPUFusedSoftmaxVMul: scores seqQ must be 1 for decode, got %d", sShape[1])
		}
		BH, seqKV = sShape[0], sShape[2]
	case 2:
		BH, seqKV = sShape[0], sShape[1]
	default:
		return nil, fmt.Errorf("GPUFusedSoftmaxVMul: scores must be 2D or 3D, got %dD", len(sShape))
	}

	// V must be [BH, seqKV, D]
	if len(vShape) != 3 || vShape[0] != BH || vShape[1] != seqKV {
		return nil, fmt.Errorf("GPUFusedSoftmaxVMul: V shape mismatch: want [%d, %d, D], got %v", BH, seqKV, vShape)
	}
	D := vShape[2]

	scoresPtr, scoresCleanup, err := getDevicePtr(e, scores)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedSoftmaxVMul scores: %w", err)
	}
	defer scoresCleanup()

	vPtr, vCleanup, err := getDevicePtr(e, V)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedSoftmaxVMul V: %w", err)
	}
	defer vCleanup()

	outElems := BH * D
	outBytes := outElems * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedSoftmaxVMul alloc: %w", err)
	}

	if err := e.kernels.FusedSoftmaxVMulF32(scoresPtr, vPtr, devOut, scale, BH, seqKV, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}

	outShape := []int{BH, 1, D}
	return makeGPUResult[T](e, outShape, devOut, outElems)
}

package compute

// gpu_engine_memory.go contains data movement, layout transformation,
// and memory operations for GPUEngine: Copy, Zero, Gather, Scatter,
// Transpose, Split, Concat, Repeat, Reshape, and format conversion.

import (
	"context"
	"fmt"
	"os"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/gpuapi"
	"github.com/zerfoo/ztensor/tensor"
)

// Transpose transposes a tensor along the given axes.
func (e *GPUEngine[T]) Transpose(ctx context.Context, a *tensor.TensorNumeric[T], axes []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}

	// Only use GPU path for GPU-resident tensors (Phase 6 behavior).
	// CPU-backed tensors fall back to CPU transpose to avoid unexpected
	// H2D copies that may interfere with CUDA graph capture/replay.
	_, isGPU := a.GetStorage().(*tensor.GPUStorage[T])
	isFP16 := false
	if e.dtype != DTypeF32 {
		_, isFP16 = any(a.GetStorage()).(*tensor.Float16Storage)
	}
	if !isGPU && !isFP16 {
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}

	e.setDevice()

	shape := a.Shape()
	rank := len(shape)

	if debugGPU {
		fmt.Fprintf(os.Stderr, "TRANSPOSE: shape=%v rank=%d axes=%v storage=%T\n", shape, rank, axes, a.GetStorage())
	}

	// Default: reverse axes (same as CPU Transpose with nil axes).
	if len(axes) == 0 {
		axes = make([]int, rank)
		for i := range rank {
			axes[i] = rank - 1 - i
		}
	}

	if len(axes) != rank {
		if debugGPU {
			fmt.Fprintf(os.Stderr, "TRANSPOSE CPU FALLBACK: reason=axes_rank_mismatch shape=%v\n", shape)
		}
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}

	// GPU transpose kernel supports up to 4D; fall back to CPU for higher ranks.
	if rank > 4 {
		if debugGPU {
			fmt.Fprintf(os.Stderr, "TRANSPOSE CPU FALLBACK: reason=rank_gt_4 shape=%v\n", shape)
		}
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}

	// Compute output shape.
	outShape := make([]int, rank)
	for i, ax := range axes {
		outShape[i] = shape[ax]
	}

	// Fast path: if the transpose only swaps unit-sized dimensions, it is
	// equivalent to a reshape (no data movement). This is common during
	// single-token generation where seqLen=1. Check by comparing the
	// non-unit dimensions in input vs output order.
	if isTransposeReshape(shape, outShape) {
		if debugGPU {
			fmt.Fprintf(os.Stderr, "TRANSPOSE: reshape fast path shape=%v outShape=%v storage=%T\n", shape, outShape, a.GetStorage())
		}
		if e.dtype != DTypeF32 {
			if fs, ok := any(a.GetStorage()).(*tensor.Float16Storage); ok {
				storageT := any(fs).(tensor.Storage[T])
				t, tErr := tensor.NewWithStorage[T](outShape, storageT)
				if tErr != nil {
					return nil, tErr
				}
				return t, nil
			}
		}
		gs := a.GetStorage().(*tensor.GPUStorage[T])
		viewGS := gs.View(gs.Len())
		t, tErr := tensor.NewWithStorage[T](outShape, viewGS)
		if tErr != nil {
			return nil, tErr
		}
		if len(dst) > 0 && dst[0] != nil {
			dst[0].SetStorage(viewGS)
			dst[0].SetShape(outShape)
			return dst[0], nil
		}
		return t, nil
	}

	// Compute total elements.
	total := 1
	for _, d := range shape {
		total *= d
	}

	// Compute input strides.
	inStrides := make([]int, rank)
	stride := 1
	for i := rank - 1; i >= 0; i-- {
		inStrides[i] = stride
		stride *= shape[i]
	}

	if debugGPU {
		fmt.Fprintf(os.Stderr, "TRANSPOSE getDevicePtr: storage=%T\n", a.GetStorage())
	}
	devIn, cleanupIn, err := getDevicePtr(e, a)
	if err != nil {
		if debugGPU {
			fmt.Fprintf(os.Stderr, "TRANSPOSE CPU FALLBACK: reason=getDevicePtr_failed shape=%v\n", shape)
		}
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}
	defer cleanupIn()
	if debugGPU {
		fmt.Fprintf(os.Stderr, "TRANSPOSE getDevicePtr OK: ptr=%p\n", devIn)
	}

	byteSize := total * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}

	// Fast path: 2D transpose.
	if rank == 2 && axes[0] == 1 && axes[1] == 0 {
		if debugGPU {
			e.logger.Debug("TRANSPOSE: using 2D fast path",
				"rows", fmt.Sprintf("%d", shape[0]),
				"cols", fmt.Sprintf("%d", shape[1]))
		}
		if err := e.kernels.Transpose2D(devIn, devOut, shape[0], shape[1], e.stream); err != nil {
			e.pool.Free(e.deviceID, devOut, byteSize)
			return nil, err
		}
		return makeGPUResult[T](e, outShape, devOut, total, dst...)
	}

	// General N-D transpose via stride-based kernel.
	// Precompute output strides on the host so the kernel avoids O(ndim^2) per thread.
	if debugGPU {
		e.logger.Debug("TRANSPOSE: using general N-D path",
			"rank", fmt.Sprintf("%d", rank),
			"axes", fmt.Sprintf("%v", axes))
	}
	outStrides := make([]int, rank)
	outStride := 1
	for i := rank - 1; i >= 0; i-- {
		outStrides[i] = outStride
		outStride *= outShape[i]
	}

	inStrides32 := make([]int32, rank)
	outStrides32 := make([]int32, rank)
	perm32 := make([]int32, rank)
	for i := range rank {
		inStrides32[i] = int32(inStrides[i])
		outStrides32[i] = int32(outStrides[i])
		perm32[i] = int32(axes[i])
	}

	if err := e.kernels.TransposeND(devIn, devOut, inStrides32, outStrides32, perm32, rank, total, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, byteSize)
		return nil, err
	}

	return makeGPUResult[T](e, outShape, devOut, total, dst...)
}

// Zero sets all elements to zero.
func (e *GPUEngine[T]) Zero(ctx context.Context, a *tensor.TensorNumeric[T]) error {
	// GPU path: use cudaMemsetAsync on the engine's stream.
	if gs, ok := a.GetStorage().(*tensor.GPUStorage[T]); ok {
		return e.runtime.MemsetAsync(gs.Ptr(), 0, gs.ByteSize(), e.stream)
	}
	// CPU fallback for non-GPU tensors.
	return e.cpu.Zero(ctx, a)
}

// Zeros fills the tensor with zeros.
func (e *GPUEngine[T]) Zeros(ctx context.Context, a *tensor.TensorNumeric[T], shape []int) error {
	return e.cpu.Zeros(ctx, a, shape)
}

// Copy copies data from source to destination tensor.
func (e *GPUEngine[T]) Copy(ctx context.Context, dst, src *tensor.TensorNumeric[T]) error {
	dstGS, dstIsGPU := dst.GetStorage().(*tensor.GPUStorage[T])
	srcGS, srcIsGPU := src.GetStorage().(*tensor.GPUStorage[T])
	if dstIsGPU && srcIsGPU {
		// D2D copy on engine stream.
		return e.runtime.MemcpyAsync(dstGS.Ptr(), srcGS.Ptr(), dstGS.ByteSize(), gpuapi.MemcpyDeviceToDevice, e.stream)
	}
	// Fall back to CPU for mixed or CPU-only tensors.
	return e.cpu.Copy(ctx, dst, src)
}

// Gather performs an embedding-style gather.
func (e *GPUEngine[T]) Gather(ctx context.Context, params *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[T]) error {
	if !isFloat32[T]() {
		return e.cpu.Gather(ctx, params, indices, output)
	}

	// Q8 GPU gather: dequantize only the requested rows on GPU.
	if qs, ok := any(params.GetStorage()).(*tensor.Q8Storage); ok {
		if ptr, _, _ := qs.GPUPtr(); ptr != nil {
			return e.gatherQ8(params, indices, output, qs, ptr)
		}
	}

	// Check whether params are GPU-resident (F32 or FP16 storage).
	_, isGPU := params.GetStorage().(*tensor.GPUStorage[T])
	var fp16Stor *tensor.Float16Storage
	isFP16 := false
	if e.dtype != DTypeF32 {
		fp16Stor, isFP16 = any(params.GetStorage()).(*tensor.Float16Storage)
	}
	if !isGPU && !isFP16 {
		return e.cpu.Gather(ctx, params, indices, output)
	}

	e.setDevice()

	pShape := params.Shape()
	if len(pShape) != 2 {
		return e.cpu.Gather(ctx, params, indices, output)
	}
	V := pShape[0]
	D := pShape[1]

	// Flatten indices to get N.
	idxData := indices.Data()
	N := len(idxData)
	if N == 0 {
		return nil
	}

	// Get device pointer for params. For Float16Storage, convert FP16->F32
	// into a temporary buffer so the F32 Gather kernel can operate on it.
	var devParams unsafe.Pointer
	var cleanupParams func()
	if isFP16 {
		fp16Ptr, _, _ := fp16Stor.GPUPtr()
		if fp16Ptr == nil {
			return e.cpu.Gather(ctx, params, indices, output)
		}
		nElems := V * D
		f32Bytes := nElems * f32Size
		f32Ptr, err := e.pool.Alloc(e.deviceID, f32Bytes)
		if err != nil {
			return e.cpu.Gather(ctx, params, indices, output)
		}
		if err := e.kernels.FP16ToF32(fp16Ptr, f32Ptr, nElems, e.stream); err != nil {
			e.pool.Free(e.deviceID, f32Ptr, f32Bytes)
			return e.cpu.Gather(ctx, params, indices, output)
		}
		devParams = f32Ptr
		cleanupParams = func() { e.pool.Free(e.deviceID, f32Ptr, f32Bytes) }
	} else {
		var err error
		devParams, cleanupParams, err = getDevicePtr(e, params)
		if err != nil {
			return e.cpu.Gather(ctx, params, indices, output)
		}
	}
	defer cleanupParams()

	// Upload indices to GPU as int64 (Go int on 64-bit platforms).
	// The gather kernel accepts int64 indices directly, avoiding the
	// CPU-side int64→int32 conversion that would trigger a D2H copy
	// for GPU-resident indices and block CUDA graph capture.
	intSize := int(unsafe.Sizeof(int(0)))
	idxByteSize := N * intSize
	devIdx, err := e.pool.Alloc(e.deviceID, idxByteSize)
	if err != nil {
		return e.cpu.Gather(ctx, params, indices, output)
	}
	defer e.pool.Free(e.deviceID, devIdx, idxByteSize)

	if err := e.runtime.Memcpy(devIdx, unsafe.Pointer(&idxData[0]), idxByteSize, gpuapi.MemcpyHostToDevice); err != nil {
		return e.cpu.Gather(ctx, params, indices, output)
	}

	// Allocate output on GPU.
	outByteSize := N * D * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, outByteSize)
	if err != nil {
		return e.cpu.Gather(ctx, params, indices, output)
	}

	if err := e.kernels.Gather(devParams, devIdx, devOut, N, D, V, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outByteSize)
		return fmt.Errorf("GPU Gather: %w", err)
	}

	// When dtype is FP16, convert the F32 gather output to FP16 on GPU.
	// This is the single F32->FP16 conversion point for the entire forward pass;
	// all downstream ops receive Float16Storage and operate in FP16 natively.
	if e.dtype == DTypeFP16 {
		outElems := N * D
		fp16Bytes := outElems * fp16Size
		fp16Ptr, err := e.pool.Alloc(e.deviceID, fp16Bytes)
		if err != nil {
			e.pool.Free(e.deviceID, devOut, outByteSize)
			return fmt.Errorf("Gather FP16 alloc: %w", err)
		}
		if err := e.kernels.F32ToFP16(devOut, fp16Ptr, outElems, e.stream); err != nil {
			e.pool.Free(e.deviceID, fp16Ptr, fp16Bytes)
			e.pool.Free(e.deviceID, devOut, outByteSize)
			return fmt.Errorf("Gather F32->FP16: %w", err)
		}
		e.pool.Free(e.deviceID, devOut, outByteSize)
		fs := any(tensor.NewFloat16StorageGPU(fp16Ptr, outElems, e.deviceID)).(tensor.Storage[T])
		output.SetStorage(fs)
		return nil
	}

	// Set output storage to GPU (pool-backed so Free returns to pool, not cudaFree).
	gs, err := tensor.NewGPUStorageFromPool[T](devOut, N*D, e.pool, e.deviceID)
	if err != nil {
		e.pool.Free(e.deviceID, devOut, outByteSize)
		return err
	}
	output.SetStorage(gs)

	return nil
}

// gatherQ8 performs Q8_0 embedding gather on GPU using the Q8 gather kernel.
// Dequantizes only the requested rows, keeping the full Q8 table compressed.
func (e *GPUEngine[T]) gatherQ8(
	params *tensor.TensorNumeric[T],
	indices *tensor.TensorNumeric[int],
	output *tensor.TensorNumeric[T],
	qs *tensor.Q8Storage,
	devQ8 unsafe.Pointer,
) error {
	e.setDevice()

	pShape := params.Shape()
	V := pShape[0]
	D := pShape[1]

	idxData := indices.Data()
	N := len(idxData)
	if N == 0 {
		return nil
	}

	// Upload indices as int32 to GPU.
	idx32 := make([]int32, N)
	for i, id := range idxData {
		idx32[i] = int32(id)
	}
	idxBytes := N * 4
	devIdx, err := e.pool.Alloc(e.deviceID, idxBytes)
	if err != nil {
		return e.cpu.Gather(context.Background(), params, indices, output)
	}
	defer e.pool.Free(e.deviceID, devIdx, idxBytes)

	if err := e.runtime.Memcpy(devIdx, unsafe.Pointer(&idx32[0]), idxBytes, gpuapi.MemcpyHostToDevice); err != nil {
		return e.cpu.Gather(context.Background(), params, indices, output)
	}

	// Allocate output [N, D] on GPU.
	outElems := N * D
	outBytes := outElems * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return e.cpu.Gather(context.Background(), params, indices, output)
	}

	// Launch Q8 gather kernel.
	if err := e.kernels.GatherQ8F32(devQ8, devIdx, devOut, N, D, V, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return e.cpu.Gather(context.Background(), params, indices, output)
	}

	// Write result into output tensor as GPUStorage (pool-backed).
	gs, err := tensor.NewGPUStorageFromPool[float32](devOut, outElems, e.pool, e.deviceID)
	if err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return fmt.Errorf("gatherQ8: create GPU storage: %w", err)
	}
	output.SetStorage(any(gs).(tensor.Storage[T]))
	return nil
}

// ScatterAdd performs a row-wise scatter-add for embeddings.
func (e *GPUEngine[T]) ScatterAdd(ctx context.Context, dEmbeddingTable *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], dOut *tensor.TensorNumeric[T]) error {
	return e.cpu.ScatterAdd(ctx, dEmbeddingTable, indices, dOut)
}

// RandomUniform fills the tensor with uniform random values.
func (e *GPUEngine[T]) RandomUniform(ctx context.Context, t *tensor.TensorNumeric[T], minVal, maxVal T) error {
	return e.cpu.RandomUniform(ctx, t, minVal, maxVal)
}

// Fill fills the tensor with a scalar value.
func (e *GPUEngine[T]) Fill(ctx context.Context, t *tensor.TensorNumeric[T], value T) error {
	return e.gpuFill(ctx, t, value)
}

// Split splits a tensor into multiple tensors along an axis.
func (e *GPUEngine[T]) Split(ctx context.Context, a *tensor.TensorNumeric[T], numSplits int, axis int) ([]*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Split(ctx, a, numSplits, axis)
	}
	if gs, ok := a.GetStorage().(*tensor.GPUStorage[T]); ok {
		return e.gpuSplit(gs.Ptr(), a.Shape(), numSplits, axis)
	}
	if e.dtype != DTypeF32 {
		if fs, ok := any(a.GetStorage()).(*tensor.Float16Storage); ok {
			ptr, _, _ := fs.GPUPtr()
			if ptr != nil {
				return e.gpuSplitFP16(ptr, a.Shape(), numSplits, axis)
			}
		}
	}
	return e.cpu.Split(ctx, a, numSplits, axis)
}

// Concat concatenates tensors along an axis.
func (e *GPUEngine[T]) Concat(ctx context.Context, tensors []*tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || len(tensors) == 0 {
		return e.cpu.Concat(ctx, tensors, axis, dst...)
	}
	// Check all inputs are GPU-resident (GPUStorage or Float16Storage).
	ptrs := make([]unsafe.Pointer, len(tensors))
	allFP16 := true
	for i, t := range tensors {
		if gs, ok := t.GetStorage().(*tensor.GPUStorage[T]); ok {
			ptrs[i] = gs.Ptr()
			allFP16 = false
		} else if e.dtype != DTypeF32 {
			if fs, ok := any(t.GetStorage()).(*tensor.Float16Storage); ok {
				p, _, _ := fs.GPUPtr()
				if p == nil {
					return e.cpu.Concat(ctx, tensors, axis, dst...)
				}
				ptrs[i] = p
			} else {
				return e.cpu.Concat(ctx, tensors, axis, dst...)
			}
		} else {
			return e.cpu.Concat(ctx, tensors, axis, dst...)
		}
	}
	if allFP16 && e.dtype != DTypeF32 {
		return e.gpuConcatFP16(ptrs, tensors, axis, dst...)
	}
	return e.gpuConcat(ptrs, tensors, axis, dst...)
}

// Repeat repeats the tensor along an axis.
func (e *GPUEngine[T]) Repeat(ctx context.Context, a *tensor.TensorNumeric[T], axis int, repetitions int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || a == nil {
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}

	shape := a.Shape()
	if axis < 0 || axis >= len(shape) {
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}
	if repetitions <= 0 {
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}

	// Get device pointer (handles GPUStorage[T] and Float16Storage).
	isFP16 := false
	if e.dtype != DTypeF32 {
		_, isFP16 = any(a.GetStorage()).(*tensor.Float16Storage)
	}
	gs, isGPU := a.GetStorage().(*tensor.GPUStorage[T])
	if !isGPU && !isFP16 {
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}

	e.setDevice()

	newShape := make([]int, len(shape))
	copy(newShape, shape)
	newShape[axis] *= repetitions

	outElems := 1
	for _, d := range newShape {
		outElems *= d
	}
	outBytes := outElems * f32Size

	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}

	var devA unsafe.Pointer
	var cleanupA func()
	if isGPU {
		devA = gs.Ptr()
		cleanupA = func() {}
	} else {
		// Float16Storage: convert FP16→F32 for the F32 repeat kernel.
		f32Engine, ok := any(e).(*GPUEngine[float32])
		if !ok {
			e.pool.Free(e.deviceID, devOut, outBytes)
			return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
		}
		devA, cleanupA, err = getDevicePtr(f32Engine, any(a).(*tensor.TensorNumeric[float32]))
		if err != nil {
			e.pool.Free(e.deviceID, devOut, outBytes)
			return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
		}
	}
	defer cleanupA()

	// Compute dimensions for the repeat kernel.
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}
	axisDim := shape[axis]
	innerSize := 1
	for i := axis + 1; i < len(shape); i++ {
		innerSize *= shape[i]
	}

	if err := e.kernels.Repeat(devA, devOut, outerSize, axisDim, innerSize, repetitions, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}

	// For FP16 inputs, convert the F32 output back to Float16Storage.
	if isFP16 {
		fp16Bytes := outElems * fp16Size
		fp16Out, allocErr := e.pool.Alloc(e.deviceID, fp16Bytes)
		if allocErr != nil {
			e.pool.Free(e.deviceID, devOut, outBytes)
			return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
		}
		if convErr := e.kernels.F32ToFP16(devOut, fp16Out, outElems, e.stream); convErr != nil {
			e.pool.Free(e.deviceID, devOut, outBytes)
			e.pool.Free(e.deviceID, fp16Out, fp16Bytes)
			return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
		}
		e.pool.Free(e.deviceID, devOut, outBytes)
		fs := tensor.NewFloat16StorageGPU(fp16Out, outElems, e.deviceID)
		storageT := any(fs).(tensor.Storage[T])
		return tensor.NewWithStorage[T](newShape, storageT)
	}

	return makeGPUResult[T](e, newShape, devOut, outElems, dst...)
}

// RepeatInterleave expands a 4D tensor from [B, numKV, S, D] to [B, numQ, S, D]
// by repeating each head along axis 1 (the head dimension) `reps` times.
// This is a fused kernel for GQA key/value head expansion, replacing the
// Reshape -> Repeat -> Reshape chain with a single kernel launch.
// axis must be 1 and the input must be 4D [B, numKV, S, D].
func (e *GPUEngine[T]) RepeatInterleave(ctx context.Context, a *tensor.TensorNumeric[T], axis int, reps int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := a.Shape()
	if !isFloat32[T]() || a == nil || axis != 1 || len(shape) != 4 || reps <= 0 {
		// Fall back to generic Repeat path for unsupported configurations.
		return e.Repeat(ctx, a, axis, reps, dst...)
	}

	B, numKV, S, D := shape[0], shape[1], shape[2], shape[3]
	numQ := numKV * reps

	gs, isGPU := a.GetStorage().(*tensor.GPUStorage[T])
	if !isGPU {
		return e.Repeat(ctx, a, axis, reps, dst...)
	}

	e.setDevice()

	outElems := B * numQ * S * D
	outBytes := outElems * f32Size

	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return e.Repeat(ctx, a, axis, reps, dst...)
	}

	if err := e.kernels.RepeatInterleaveF32(gs.Ptr(), devOut, B, numKV, S, D, reps, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return e.Repeat(ctx, a, axis, reps, dst...)
	}

	outShape := []int{B, numQ, S, D}
	return makeGPUResult[T](e, outShape, devOut, outElems, dst...)
}

// OneHot creates a one-hot encoding.
func (e *GPUEngine[T]) OneHot(ctx context.Context, input *tensor.TensorNumeric[int], depth int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.OneHot(ctx, input, depth, dst...)
}

// Reshape changes the shape without changing data.
func (e *GPUEngine[T]) Reshape(ctx context.Context, a *tensor.TensorNumeric[T], shape []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Resolve -1 dimension and verify size.
	currentSize := a.Size()
	inferredShape := make([]int, len(shape))
	copy(inferredShape, shape)
	inferIdx := -1
	knownSize := 1
	for i, d := range inferredShape {
		if d == -1 {
			inferIdx = i
		} else {
			knownSize *= d
		}
	}
	if inferIdx >= 0 {
		inferredShape[inferIdx] = currentSize / knownSize
	}
	newSize := 1
	for _, d := range inferredShape {
		newSize *= d
	}

	// Float16Storage: zero-copy reshape (same GPU pointer, new shape).
	if e.dtype != DTypeF32 {
		if fs, ok := any(a.GetStorage()).(*tensor.Float16Storage); ok && newSize == currentSize {
			return tensor.NewWithStorage[T](inferredShape, any(fs).(tensor.Storage[T]))
		}
	}

	// GPUStorage[T]: zero-copy reshape.
	if gs, ok := a.GetStorage().(*tensor.GPUStorage[T]); ok && isFloat32[T]() && newSize == currentSize {
		return tensor.NewWithStorage[T](inferredShape, gs.View(gs.Len()))
	}

	return e.cpu.Reshape(ctx, a, shape, dst...)
}

// ConvertFP16ToF32 converts a tensor with Float16Storage to a regular float32
// GPU tensor using the FP16->F32 kernel. Returns the input unchanged if it
// does not have Float16Storage.
func (e *GPUEngine[T]) ConvertFP16ToF32(t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	fs, ok := any(t.GetStorage()).(*tensor.Float16Storage)
	if !ok {
		return t, nil
	}

	fp16Ptr, _, _ := fs.GPUPtr()
	if fp16Ptr == nil {
		// CPU-side Float16Storage: decode via Slice (no GPU conversion possible).
		data := fs.Slice()
		out, err := tensor.New(t.Shape(), data)
		if err != nil {
			return nil, fmt.Errorf("ConvertFP16ToF32: create f32 tensor: %w", err)
		}
		return out, nil
	}

	e.setDevice()

	nElems := fs.Len()
	f32Bytes := nElems * f32Size
	f32Ptr, err := e.pool.Alloc(e.deviceID, f32Bytes)
	if err != nil {
		return nil, fmt.Errorf("ConvertFP16ToF32: alloc: %w", err)
	}

	if err := e.kernels.FP16ToF32(fp16Ptr, f32Ptr, nElems, e.stream); err != nil {
		e.pool.Free(e.deviceID, f32Ptr, f32Bytes)
		return nil, fmt.Errorf("ConvertFP16ToF32: kernel: %w", err)
	}

	gsOut, err := tensor.NewGPUStorageFromPool[float32](f32Ptr, nElems, e.pool, e.deviceID)
	if err != nil {
		e.pool.Free(e.deviceID, f32Ptr, f32Bytes)
		return nil, fmt.Errorf("ConvertFP16ToF32: gpu storage: %w", err)
	}
	out, err := tensor.NewWithStorage[float32](t.Shape(), gsOut)
	if err != nil {
		return nil, fmt.Errorf("ConvertFP16ToF32: wrap tensor: %w", err)
	}
	return out, nil
}

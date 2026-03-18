package compute

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda/kernels"
	"github.com/zerfoo/ztensor/tensor"
)

// PagedGQA computes scaled dot-product attention with block-table indirection
// for paged KV caches. When the paged attention kernel is not available, it
// returns an error.
//
// Q:            [batch*numQHeads, headDim] query tensor (GPU-resident).
// blockPtrsK:   device array of float* pointers to K blocks.
// blockPtrsV:   device array of float* pointers to V blocks.
// blockIndices: device array [batch * maxNumBlocks] logical→physical mapping.
// Returns output [batch*numQHeads, headDim].
func (e *GPUEngine[T]) PagedGQA(
	Q *tensor.TensorNumeric[float32],
	blockPtrsK, blockPtrsV unsafe.Pointer,
	blockIndices unsafe.Pointer,
	seqLen, blockSize, headDim int,
	numQHeads, numKVHeads int,
	batch int,
) (*tensor.TensorNumeric[float32], error) {
	if !kernels.IsPagedAttentionSupported() {
		return nil, fmt.Errorf("PagedGQA: paged attention kernel not available")
	}

	// Get Q device pointer. We work with float32 tensors directly.
	qEng := any(e).(*GPUEngine[float32])
	qPtr, qCleanup, err := getDevicePtr(qEng, Q)
	if err != nil {
		return nil, fmt.Errorf("PagedGQA: Q device ptr: %w", err)
	}
	defer qCleanup()

	outElems := batch * numQHeads * headDim
	outBytes := outElems * f32Size
	e.setDevice()
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("PagedGQA: alloc output: %w", err)
	}

	var streamPtr unsafe.Pointer
	if e.stream != nil {
		streamPtr = e.stream.Ptr()
	}

	if err := kernels.PagedAttentionForward(
		qPtr, devOut,
		blockPtrsK, blockPtrsV,
		blockIndices,
		seqLen, blockSize, headDim,
		numQHeads, numKVHeads,
		batch,
		streamPtr,
	); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, fmt.Errorf("PagedGQA: kernel: %w", err)
	}

	return makeGPUResult(qEng, []int{batch * numQHeads, headDim}, devOut, outElems)
}

// IsPagedGQASupported returns true when the paged attention CUDA kernel is
// loaded and available.
func (e *GPUEngine[T]) IsPagedGQASupported() bool {
	return kernels.IsPagedAttentionSupported()
}

// Static type assertion: GPUEngine[float32] satisfies PagedGQAer.
var _ PagedGQAer = (*GPUEngine[float32])(nil)

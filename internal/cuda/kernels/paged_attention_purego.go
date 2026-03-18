//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// PagedAttentionForward computes scaled dot-product attention with block-table
// indirection for paged KV caches.
//
// Q:            [batch*numQHeads, headDim] -- single query per head.
// O:            [batch*numQHeads, headDim] -- output, same shape as Q.
// blockPtrsK:   device array of float* pointers to K blocks.
//
//	Each block holds [blockSize, numKVHeads, headDim] floats.
//
// blockPtrsV:   device array of float* pointers to V blocks (same layout).
// blockIndices: device array [batch * maxNumBlocks] mapping logical block
//
//	index to physical block index in blockPtrs arrays.
//
// seqLen:       actual number of valid K/V token positions.
// blockSize:    number of token positions per block.
// headDim:      dimension per head.
// numQHeads:    number of query heads per batch element.
// numKVHeads:   number of KV heads per batch element.
// batch:        number of sequences in the batch.
func PagedAttentionForward(
	Q, O unsafe.Pointer,
	blockPtrsK, blockPtrsV unsafe.Pointer,
	blockIndices unsafe.Pointer,
	seqLen, blockSize, headDim int,
	numQHeads, numKVHeads int,
	batch int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("paged_attention_forward_f32 kernel: kernels not available")
	}
	if k.launchPagedAttentionF32 == 0 {
		return fmt.Errorf("paged_attention_forward_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchPagedAttentionF32,
		uintptr(Q), uintptr(O),
		uintptr(blockPtrsK), uintptr(blockPtrsV),
		uintptr(blockIndices),
		uintptr(seqLen), uintptr(blockSize), uintptr(headDim),
		uintptr(numQHeads), uintptr(numKVHeads),
		uintptr(batch),
		uintptr(stream))
	return checkKernel(ret, "paged_attention_forward_f32")
}

// IsPagedAttentionSupported returns true if the paged attention kernel symbol
// was loaded from libkernels.so.
func IsPagedAttentionSupported() bool {
	k := klib()
	return k != nil && k.launchPagedAttentionF32 != 0
}

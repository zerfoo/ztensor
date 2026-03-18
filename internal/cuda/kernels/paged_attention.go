//go:build cuda && cutlass

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "paged_attention.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// PagedAttentionForward computes scaled dot-product attention with block-table
// indirection for paged KV caches.
func PagedAttentionForward(
	Q, O unsafe.Pointer,
	blockPtrsK, blockPtrsV unsafe.Pointer,
	blockIndices unsafe.Pointer,
	seqLen, blockSize, headDim int,
	numQHeads, numKVHeads int,
	batch int,
	stream unsafe.Pointer,
) error {
	err := C.paged_attention_forward_f32(
		(*C.float)(Q), (*C.float)(O),
		(**C.float)(blockPtrsK), (**C.float)(blockPtrsV),
		(*C.int)(blockIndices),
		C.int(seqLen), C.int(blockSize), C.int(headDim),
		C.int(numQHeads), C.int(numKVHeads),
		C.int(batch),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("paged_attention_forward_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

// IsPagedAttentionSupported returns true when compiled with CUDA+CUTLASS.
func IsPagedAttentionSupported() bool {
	return true
}

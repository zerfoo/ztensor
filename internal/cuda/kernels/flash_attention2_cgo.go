//go:build cuda && cutlass

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "flash_attention2.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// FlashAttention2Forward computes scaled dot-product attention using the
// FlashAttention-2 tiled algorithm. All tensors are [batch, heads, seq_len,
// head_dim] in row-major order. When causal is true, an upper-triangular
// mask is applied. Memory usage is O(N), not O(N^2).
func FlashAttention2Forward(
	Q, K, V, O unsafe.Pointer,
	batch, heads, seqLen, headDim int,
	causal bool,
	stream unsafe.Pointer,
) error {
	c := C.int(0)
	if causal {
		c = 1
	}
	err := C.flash_attention2_forward_f32(
		(*C.float)(Q), (*C.float)(K), (*C.float)(V), (*C.float)(O),
		C.int(batch), C.int(heads), C.int(seqLen), C.int(headDim),
		c, C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("flash_attention2_forward_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

// FlashAttention2Decode computes single-query attention for autoregressive
// decode using FlashAttention-2 with multi-warp KV parallelism.
// Supports GQA: numQueryHeads may differ from numKVHeads (must be a multiple).
func FlashAttention2Decode(
	Q, K, V, O unsafe.Pointer,
	numBH, maxKVLen, headDim, kvLen int,
	kvLenPtr unsafe.Pointer,
	numQueryHeads, numKVHeads int,
	stream unsafe.Pointer,
) error {
	err := C.flash_attention2_decode_f32(
		(*C.float)(Q), (*C.float)(K), (*C.float)(V), (*C.float)(O),
		C.int(numBH), C.int(maxKVLen), C.int(headDim),
		C.int(kvLen), (*C.int)(kvLenPtr),
		C.int(numQueryHeads), C.int(numKVHeads),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("flash_attention2_decode_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

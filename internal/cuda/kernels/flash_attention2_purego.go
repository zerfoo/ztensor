//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
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
	k := klib()
	if k == nil {
		return fmt.Errorf("flash_attention2_forward_f32 kernel: kernels not available")
	}
	if k.launchFlashAttention2F32 == 0 {
		return fmt.Errorf("flash_attention2_forward_f32 kernel: symbol not loaded")
	}
	c := uintptr(0)
	if causal {
		c = 1
	}
	ret := cuda.Ccall(k.launchFlashAttention2F32,
		uintptr(Q), uintptr(K), uintptr(V), uintptr(O),
		uintptr(batch), uintptr(heads), uintptr(seqLen), uintptr(headDim),
		c, uintptr(stream))
	return checkKernel(ret, "flash_attention2_forward_f32")
}

// FlashAttention2Decode computes single-query attention for autoregressive
// decode using FlashAttention-2 with multi-warp KV parallelism.
// Supports GQA: numQueryHeads may differ from numKVHeads (must be a multiple).
//
// Q: [batch*numQueryHeads, 1, headDim]   -- single query per head.
// K: [batch*numKVHeads, maxKVLen, headDim]  -- pre-allocated KV cache buffer.
// V: [batch*numKVHeads, maxKVLen, headDim]
// O: [batch*numQueryHeads, 1, headDim]   -- output.
//
// kvLen is the actual KV sequence length (used when kvLenPtr is nil).
// kvLenPtr is a GPU-resident int32 pointer; when non-nil the kernel reads
// the KV length from GPU memory at runtime, making it compatible with
// CUDA graph replay (the value is not frozen at capture time).
func FlashAttention2Decode(
	Q, K, V, O unsafe.Pointer,
	numBH, maxKVLen, headDim, kvLen int,
	kvLenPtr unsafe.Pointer,
	numQueryHeads, numKVHeads int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("flash_attention2_decode_f32 kernel: kernels not available")
	}
	if k.launchFlashAttention2DecodeF32 == 0 {
		return fmt.Errorf("flash_attention2_decode_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchFlashAttention2DecodeF32,
		uintptr(Q), uintptr(K), uintptr(V), uintptr(O),
		uintptr(numBH), uintptr(maxKVLen), uintptr(headDim),
		uintptr(kvLen), uintptr(kvLenPtr),
		uintptr(numQueryHeads), uintptr(numKVHeads),
		uintptr(stream))
	return checkKernel(ret, "flash_attention2_decode_f32")
}

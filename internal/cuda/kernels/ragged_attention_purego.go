//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// RaggedAttentionForward computes scaled dot-product attention for
// variable-length sequences packed into a single batch (ragged batching).
// A block-diagonal attention mask prevents cross-sequence attention.
//
// Q:           [totalTokens * numQHeads, headDim] -- packed queries.
// K:           [totalTokens * numKVHeads, headDim] -- packed keys.
// V:           [totalTokens * numKVHeads, headDim] -- packed values.
// O:           [totalTokens * numQHeads, headDim] -- output.
// seqLens:     [batch] int32 -- actual sequence length for each sequence.
// cumSeqLens:  [batch] int32 -- cumulative offsets (prefix sums, first = 0).
// batch:       number of sequences.
// numQHeads:   number of query heads.
// numKVHeads:  number of KV heads.
// headDim:     dimension per head.
func RaggedAttentionForward(
	Q, K, V, O unsafe.Pointer,
	seqLens, cumSeqLens unsafe.Pointer,
	batch, numQHeads, numKVHeads, headDim int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("ragged_attention_forward_f32 kernel: kernels not available")
	}
	if k.launchRaggedAttentionF32 == 0 {
		return fmt.Errorf("ragged_attention_forward_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchRaggedAttentionF32,
		uintptr(Q), uintptr(K), uintptr(V), uintptr(O),
		uintptr(seqLens), uintptr(cumSeqLens),
		uintptr(batch), uintptr(numQHeads), uintptr(numKVHeads), uintptr(headDim),
		uintptr(stream))
	return checkKernel(ret, "ragged_attention_forward_f32")
}

// IsRaggedAttentionSupported returns true if the ragged attention kernel symbol
// was loaded from libkernels.so.
func IsRaggedAttentionSupported() bool {
	k := klib()
	return k != nil && k.launchRaggedAttentionF32 != 0
}

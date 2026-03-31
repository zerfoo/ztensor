package compute

import (
	"github.com/zerfoo/ztensor/tensor"
)

// FusedSoftmaxVMulProvider is implemented by engines that support fused
// softmax + V multiply in a single GPU kernel launch for decode attention.
// It computes output = softmax(scores * scale) @ V without materializing
// the attention weights tensor, saving one kernel launch and the memory
// traffic of writing/reading the weights.
//
// This is decode-optimized (seqQ=1). For prefill (seqQ>1), callers should
// fall back to separate ScaledSoftmax + MatMul.
//
// This API is not covered by the v1 stability guarantee.
type FusedSoftmaxVMulProvider[T tensor.Numeric] interface {
	// GPUFusedSoftmaxVMul computes softmax(scores * scale) @ V.
	// scores: [BH, 1, seqKV] (decode: seqQ=1, treated as [BH, seqKV]).
	// V: [BH, seqKV, D].
	// Returns output: [BH, 1, D].
	GPUFusedSoftmaxVMul(scores, V *tensor.TensorNumeric[T], scale float32) (*tensor.TensorNumeric[T], error)
}

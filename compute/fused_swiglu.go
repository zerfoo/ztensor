package compute

import (
	"github.com/zerfoo/ztensor/tensor"
)

// FusedSwiGLUProvider is implemented by engines that support fused GPU SwiGLU.
// It computes output[i] = w1[i] * sigmoid(w1[i]) * w3[i] in a single kernel,
// eliminating the Concat + Split + sigmoid + Mul + Mul chain.
//
// This API is not covered by the v1 stability guarantee.
type FusedSwiGLUProvider[T tensor.Numeric] interface {
	GPUFusedSwiGLU(w1, w3 *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
}

package compute

import (
	"github.com/zerfoo/ztensor/tensor"
)

// FusedScaledSoftmaxProvider is implemented by engines that support fused GPU scaled softmax.
// It computes output = softmax(input * scale) in a single kernel launch,
// eliminating the MulScalar + Softmax chain (saves 1 kernel launch per call).
//
// This API is not covered by the v1 stability guarantee.
type FusedScaledSoftmaxProvider[T tensor.Numeric] interface {
	GPUScaledSoftmax(input *tensor.TensorNumeric[T], scale float32, axis int) (*tensor.TensorNumeric[T], error)
}

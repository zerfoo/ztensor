package compute

import (
	"github.com/zerfoo/ztensor/tensor"
)

// FusedQKNormRoPEProvider is implemented by engines that support fused
// per-head QK RMSNorm + RoPE in a single GPU kernel launch.
// This replaces 4 kernel launches (Q_norm + K_norm + Q_RoPE + K_RoPE)
// with 1 per GQA layer during decode.
//
// This API is not covered by the v1 stability guarantee.
type FusedQKNormRoPEProvider[T tensor.Numeric] interface {
	// GPUFusedQKNormRoPE applies per-head RMSNorm + RoPE to combined Q+K data.
	// input: [totalHeads, headDim] (Q heads then K heads, contiguous).
	// weightQ/weightK: [headDim] RMSNorm weights.
	// cosAngles/sinAngles: [halfRotary] precomputed angles for current position.
	// Returns output: [totalHeads, headDim].
	GPUFusedQKNormRoPE(
		input *tensor.TensorNumeric[T],
		weightQ, weightK *tensor.TensorNumeric[T],
		cosAngles, sinAngles *tensor.TensorNumeric[T],
		eps float32,
		totalHeads, headDim, numQHeads, halfRotary int,
	) (*tensor.TensorNumeric[T], error)
}

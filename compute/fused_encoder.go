package compute

import "unsafe"

// FusedEncoderProvider is implemented by engines that support fused PatchTST
// encoder layer forward and backward passes. The fused kernel replaces ~78
// discrete engine operations per layer with a single orchestrated call,
// using cuBLAS for GEMMs and custom CUDA sub-kernels for LayerNorm, GELU,
// softmax, head transpose, and residual operations.
//
// Callers must pre-allocate all buffer arrays and pass device pointers.
// Buffer index constants (FEW_*, FEB_*, FEG_*, etc.) are defined in
// internal/cuda/kernels/fused_encoder_fwd_purego.go and fused_encoder_bwd_purego.go.
//
// This API is not covered by the v1 stability guarantee.
type FusedEncoderProvider interface {
	// FusedEncoderAvailable returns true if the fused encoder kernel is loaded.
	FusedEncoderAvailable() bool

	// AllocDeviceFloat32 allocates numElements float32s on the GPU and returns
	// the device pointer. Memory is pool-managed and freed when the engine closes.
	AllocDeviceFloat32(numElements int) (unsafe.Pointer, error)

	// CopyToDevice copies len(src) float32 values from host to a device pointer.
	CopyToDevice(dst unsafe.Pointer, src []float32) error

	// FusedEncoderForward executes one encoder layer forward pass.
	// weights: [16]unsafe.Pointer to layer weights.
	// bufs: [16]unsafe.Pointer to pre-allocated forward cache buffers.
	// input/output: [totalRows, dModel] device pointers.
	FusedEncoderForward(
		weights *[16]unsafe.Pointer,
		bufs *[16]unsafe.Pointer,
		input, output unsafe.Pointer,
		totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches int,
	) error

	// FusedEncoderBackward computes all gradients for one encoder layer.
	// weights: [16]unsafe.Pointer to layer weights.
	// weightT: [6]unsafe.Pointer to pre-transposed weights.
	// fwdBufs: [16]unsafe.Pointer to forward cache (from FusedEncoderForward).
	// bwdBufs: [15]unsafe.Pointer to backward scratch buffers.
	// grads: [16]unsafe.Pointer to gradient accumulators (accumulated, not zeroed).
	// dOutput: upstream gradient; dInput: output gradient; input: original layer input.
	FusedEncoderBackward(
		weights *[16]unsafe.Pointer,
		weightT *[6]unsafe.Pointer,
		fwdBufs *[16]unsafe.Pointer,
		bwdBufs *[15]unsafe.Pointer,
		grads *[16]unsafe.Pointer,
		dOutput, dInput, input unsafe.Pointer,
		totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches int,
	) error
}

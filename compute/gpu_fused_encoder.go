package compute

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cublas"
)

// blasHandlePtr extracts the raw cuBLAS handle pointer from the BLAS interface.
// Returns nil if the BLAS is not backed by cuBLAS.
func blasHandlePtr(b interface{}) unsafe.Pointer {
	type handleProvider interface {
		Handle() *cublas.Handle
	}
	if hp, ok := b.(handleProvider); ok {
		h := hp.Handle()
		if h != nil {
			return h.Ptr()
		}
	}
	return nil
}

// FusedEncoderAvailable returns true if the fused encoder kernel is loaded
// and the engine has a cuBLAS handle to pass to it.
func (e *GPUEngine[T]) FusedEncoderAvailable() bool {
	return e.kernels.FusedEncoderFwdAvailable() && blasHandlePtr(e.blas) != nil
}

// FusedEncoderForward executes one fused encoder layer forward pass.
func (e *GPUEngine[T]) FusedEncoderForward(
	weights *[16]unsafe.Pointer,
	bufs *[16]unsafe.Pointer,
	input, output unsafe.Pointer,
	totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches int,
) error {
	h := blasHandlePtr(e.blas)
	if h == nil {
		return fmt.Errorf("FusedEncoderForward: cuBLAS handle not available")
	}
	e.setDevice()
	return e.kernels.FusedEncoderFwdF32(h, weights, bufs, input, output,
		totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches, e.stream)
}

// FusedEncoderBackward computes all gradients for one fused encoder layer.
func (e *GPUEngine[T]) FusedEncoderBackward(
	weights *[16]unsafe.Pointer,
	weightT *[6]unsafe.Pointer,
	fwdBufs *[16]unsafe.Pointer,
	bwdBufs *[15]unsafe.Pointer,
	grads *[16]unsafe.Pointer,
	dOutput, dInput, input unsafe.Pointer,
	totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches int,
) error {
	h := blasHandlePtr(e.blas)
	if h == nil {
		return fmt.Errorf("FusedEncoderBackward: cuBLAS handle not available")
	}
	e.setDevice()
	// The KernelRunner interface uses *[16] for weightT, but we have *[6].
	// Convert via unsafe pointer.
	var wt16 [16]unsafe.Pointer
	copy(wt16[:6], weightT[:])
	return e.kernels.FusedEncoderBwdF32(h, weights, &wt16, fwdBufs, bwdBufs, grads,
		dOutput, dInput, input,
		totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches, e.stream)
}

// Compile-time check that GPUEngine implements FusedEncoderProvider.
var _ FusedEncoderProvider = (*GPUEngine[float32])(nil)

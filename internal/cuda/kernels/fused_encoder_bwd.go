//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lcublas -lstdc++
#include "fused_encoder_bwd.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// FusedEncoderBwdF32 computes all gradients for one encoder layer backward pass (CGo path).
func FusedEncoderBwdF32(
	cublasHandle unsafe.Pointer,
	weights *[FEW_COUNT]unsafe.Pointer,
	weightT *[FEWT_COUNT]unsafe.Pointer,
	fwdBufs *[FEB_COUNT]unsafe.Pointer,
	bwdBufs *[FEBB_COUNT]unsafe.Pointer,
	grads *[FEG_COUNT]unsafe.Pointer,
	dOutput, dInput, input unsafe.Pointer,
	totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches int,
	stream unsafe.Pointer,
) error {
	err := C.fused_encoder_bwd_f32(
		cublasHandle,
		(*unsafe.Pointer)(unsafe.Pointer(weights)),
		(*unsafe.Pointer)(unsafe.Pointer(weightT)),
		(*unsafe.Pointer)(unsafe.Pointer(fwdBufs)),
		(*unsafe.Pointer)(unsafe.Pointer(bwdBufs)),
		(*unsafe.Pointer)(unsafe.Pointer(grads)),
		(*C.float)(dOutput),
		(*C.float)(dInput),
		(*C.float)(input),
		C.int(totalRows), C.int(dModel), C.int(nHeads), C.int(headDim),
		C.int(ffnDim), C.int(bsC), C.int(numPatches),
		C.cudaStream_t(stream),
	)
	if err != 0 {
		return fmt.Errorf("fused_encoder_bwd_f32 failed with cuda error %d", int(err))
	}
	return nil
}

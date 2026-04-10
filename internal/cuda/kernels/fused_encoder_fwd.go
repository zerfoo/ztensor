//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lcublas -lstdc++
#include "fused_encoder_fwd.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// FusedEncoderFwdF32 executes one encoder layer forward pass in a single call (CGo path).
func FusedEncoderFwdF32(
	cublasHandle unsafe.Pointer,
	weights *[FEW_COUNT]unsafe.Pointer,
	bufs *[FEB_COUNT]unsafe.Pointer,
	input, output unsafe.Pointer,
	totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches int,
	stream unsafe.Pointer,
) error {
	err := C.fused_encoder_fwd_f32(
		cublasHandle,
		(*unsafe.Pointer)(unsafe.Pointer(weights)),
		(*unsafe.Pointer)(unsafe.Pointer(bufs)),
		(*C.float)(input),
		(*C.float)(output),
		C.int(totalRows), C.int(dModel), C.int(nHeads), C.int(headDim),
		C.int(ffnDim), C.int(bsC), C.int(numPatches),
		C.cudaStream_t(stream),
	)
	if err != 0 {
		return fmt.Errorf("fused_encoder_fwd_f32 failed with cuda error %d", int(err))
	}
	return nil
}

// FusedEncoderFwdScratchBytes returns the total bytes needed for all FEB_COUNT buffers.
func FusedEncoderFwdScratchBytes(
	totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches int,
) int64 {
	return int64(C.fused_encoder_fwd_scratch_bytes(
		C.int(totalRows), C.int(dModel), C.int(nHeads), C.int(headDim),
		C.int(ffnDim), C.int(bsC), C.int(numPatches)))
}

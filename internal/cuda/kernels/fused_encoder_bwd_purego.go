//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// Backward buffer index constants matching C enum FusedEncoderGrad.
const (
	FEG_DQW     = 0
	FEG_DQB     = 1
	FEG_DKW     = 2
	FEG_DKB     = 3
	FEG_DVW     = 4
	FEG_DVB     = 5
	FEG_DOW     = 6
	FEG_DOB     = 7
	FEG_DFFN1W  = 8
	FEG_DFFN1B  = 9
	FEG_DFFN2W  = 10
	FEG_DFFN2B  = 11
	FEG_DNORM1W = 12
	FEG_DNORM1B = 13
	FEG_DNORM2W = 14
	FEG_DNORM2B = 15
	FEG_COUNT   = 16
)

// Weight transpose indices matching C enum FusedEncoderWeightT.
const (
	FEWT_QWT    = 0
	FEWT_KWT    = 1
	FEWT_VWT    = 2
	FEWT_OWT    = 3
	FEWT_FFN1WT = 4
	FEWT_FFN2WT = 5
	FEWT_COUNT  = 6
)

// Backward scratch buffer indices matching C enum FusedEncoderBwdBuf.
const (
	FEBB_DFFN1_OUT  = 0
	FEBB_DFFN1_PRE  = 1
	FEBB_DNORMED2   = 2
	FEBB_DX_RES1    = 3
	FEBB_DATTN_OUT  = 4
	FEBB_DATTN_OUT_H = 5
	FEBB_DQH        = 6
	FEBB_DKH        = 7
	FEBB_DVH        = 8
	FEBB_DSCORES    = 9
	FEBB_DQ         = 10
	FEBB_DK         = 11
	FEBB_DV         = 12
	FEBB_DNORMED1   = 13
	FEBB_TEMP       = 14
	FEBB_COUNT      = 15
)

// FusedEncoderBwdF32 computes all gradients for one encoder layer backward pass.
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
	k := klib()
	if k == nil || k.launchFusedEncoderBwdF32 == 0 {
		return fmt.Errorf("fused_encoder_bwd_f32 kernel: not available")
	}
	ret := cuda.Ccall(k.launchFusedEncoderBwdF32,
		uintptr(cublasHandle),
		uintptr(unsafe.Pointer(weights)),
		uintptr(unsafe.Pointer(weightT)),
		uintptr(unsafe.Pointer(fwdBufs)),
		uintptr(unsafe.Pointer(bwdBufs)),
		uintptr(unsafe.Pointer(grads)),
		uintptr(dOutput),
		uintptr(dInput),
		uintptr(input),
		uintptr(totalRows), uintptr(dModel), uintptr(nHeads), uintptr(headDim),
		uintptr(ffnDim), uintptr(bsC), uintptr(numPatches),
		uintptr(stream))
	return checkKernel(ret, "fused_encoder_bwd_f32")
}

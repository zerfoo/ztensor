//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// Buffer index constants matching the C enum FusedEncoderWeight.
const (
	FEW_QW     = 0
	FEW_QB     = 1
	FEW_KW     = 2
	FEW_KB     = 3
	FEW_VW     = 4
	FEW_VB     = 5
	FEW_OW     = 6
	FEW_OB     = 7
	FEW_FFN1W  = 8
	FEW_FFN1B  = 9
	FEW_FFN2W  = 10
	FEW_FFN2B  = 11
	FEW_NORM1W = 12
	FEW_NORM1B = 13
	FEW_NORM2W = 14
	FEW_NORM2B = 15
	FEW_COUNT  = 16
)

// Buffer index constants matching the C enum FusedEncoderFwdBuf.
const (
	FEB_NORMED1    = 0
	FEB_LN1_INVSTD = 1
	FEB_Q          = 2
	FEB_K          = 3
	FEB_V          = 4
	FEB_QH         = 5
	FEB_KH         = 6
	FEB_VH         = 7
	FEB_ATTN_SCORES = 8
	FEB_ATTN_OUT_H = 9
	FEB_ATTN_OUT   = 10
	FEB_X_RES1     = 11
	FEB_NORMED2    = 12
	FEB_LN2_INVSTD = 13
	FEB_FFN1_PRE   = 14
	FEB_FFN1_OUT   = 15
	FEB_COUNT      = 16
)

// FusedEncoderFwdF32 executes one encoder layer forward pass in a single call.
//
// Parameters:
//   - cublasHandle: raw cuBLAS handle pointer (from cublas.Handle.Ptr())
//   - weights: [FEW_COUNT]unsafe.Pointer to layer weight device memory
//   - bufs: [FEB_COUNT]unsafe.Pointer to pre-allocated cache buffers
//   - input, output: [totalRows, dModel] device pointers
//   - totalRows..numPatches: dimension parameters
//   - stream: CUDA stream pointer
func FusedEncoderFwdF32(
	cublasHandle unsafe.Pointer,
	weights *[FEW_COUNT]unsafe.Pointer,
	bufs *[FEB_COUNT]unsafe.Pointer,
	input, output unsafe.Pointer,
	totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil || k.launchFusedEncoderFwdF32 == 0 {
		return fmt.Errorf("fused_encoder_fwd_f32 kernel: not available")
	}
	ret := cuda.Ccall(k.launchFusedEncoderFwdF32,
		uintptr(cublasHandle),
		uintptr(unsafe.Pointer(weights)),
		uintptr(unsafe.Pointer(bufs)),
		uintptr(input),
		uintptr(output),
		uintptr(totalRows), uintptr(dModel), uintptr(nHeads), uintptr(headDim),
		uintptr(ffnDim), uintptr(bsC), uintptr(numPatches),
		uintptr(stream))
	return checkKernel(ret, "fused_encoder_fwd_f32")
}

// FusedEncoderFwdScratchBytes returns the total bytes needed for all FEB_COUNT buffers.
func FusedEncoderFwdScratchBytes(
	totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches int,
) int64 {
	k := klib()
	if k == nil || k.launchFusedEncoderFwdScratch == 0 {
		return -1
	}
	ret := cuda.Ccall(k.launchFusedEncoderFwdScratch,
		uintptr(totalRows), uintptr(dModel), uintptr(nHeads), uintptr(headDim),
		uintptr(ffnDim), uintptr(bsC), uintptr(numPatches))
	return int64(ret)
}

// FusedEncoderAvailable returns true if the fused encoder kernels are loaded.
func FusedEncoderAvailable() bool {
	k := klib()
	return k != nil && k.launchFusedEncoderFwdF32 != 0
}

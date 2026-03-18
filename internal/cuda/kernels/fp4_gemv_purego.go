//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// IsFP4GemvSupported returns true if the current GPU supports NVFP4 GEMV
// (requires sm_100+ Blackwell architecture).
func IsFP4GemvSupported() bool {
	if !cuda.Available() {
		return false
	}
	major, _, err := cuda.DeviceComputeCapability(0)
	if err != nil {
		return false
	}
	return major >= 10
}

// FP4GemvF16 performs NVFP4 fused dequant-GEMV with FP16 activations:
//
//	y[m] = sum_k( dequant(W_fp4[m,k]) * x_fp16[k] )
//
// W_fp4: device pointer to packed NVFP4 data [M, K] (8 bytes per block of 16).
// scales: device pointer to [M * ceil(K/16)] float16 block scales.
// x: device pointer to [K] float16 input vector.
// y: device pointer to [M] float32 output vector.
func FP4GemvF16(
	wFP4, scales, x, y unsafe.Pointer,
	M, K int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fp4_gemv kernel: kernels not available")
	}
	if k.launchFP4GemvF16 == 0 {
		return fmt.Errorf("fp4_gemv kernel: symbol not resolved (sm_100+ required)")
	}
	ret := cuda.Ccall(k.launchFP4GemvF16,
		uintptr(wFP4), uintptr(scales),
		uintptr(x), uintptr(y),
		uintptr(M), uintptr(K), uintptr(stream))
	return checkKernel(ret, "fp4_gemv_f16")
}

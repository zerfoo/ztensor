//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// GemvQ5KF32 performs Q5_K fused dequant-GEMV: y = dequant(W_q5k) * x.
// W_q5k is raw Q5_K super-blocks, x is [K] FP32, y is [M] FP32.
func GemvQ5KF32(
	W_q5k, x, y unsafe.Pointer, //nolint:gocritic // match CGo API
	M, K int, //nolint:gocritic // match CGo API
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gemv_q5k_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGemvQ5KF32,
		uintptr(W_q5k), uintptr(x), uintptr(y),
		uintptr(M), uintptr(K), uintptr(stream))
	return checkKernel(ret, "gemv_q5k_f32")
}

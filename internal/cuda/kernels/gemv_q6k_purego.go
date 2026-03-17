//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// GemvQ6KF32 performs Q6_K fused dequant-GEMV: y = dequant(W_q6k) * x.
// W_q6k is raw Q6_K super-blocks, x is [K] FP32, y is [M] FP32.
func GemvQ6KF32(
	W_q6k, x, y unsafe.Pointer, //nolint:gocritic // match CGo API
	M, K int, //nolint:gocritic // match CGo API
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gemv_q6k_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGemvQ6KF32,
		uintptr(W_q6k), uintptr(x), uintptr(y),
		uintptr(M), uintptr(K), uintptr(stream))
	return checkKernel(ret, "gemv_q6k_f32")
}

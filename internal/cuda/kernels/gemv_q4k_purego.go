//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// GemvQ4KF32 performs Q4_K fused dequant-GEMV: y = dequant(W_q4k) * x.
// W_q4k is raw Q4_K super-blocks, x is [K] FP32, y is [M] FP32.
func GemvQ4KF32(
	W_q4k, x, y unsafe.Pointer, //nolint:gocritic // match CGo API
	M, K int, //nolint:gocritic // match CGo API
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gemv_q4k_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGemvQ4KF32,
		uintptr(W_q4k), uintptr(x), uintptr(y),
		uintptr(M), uintptr(K), uintptr(stream))
	return checkKernel(ret, "gemv_q4k_f32")
}

// GemvQ4KDp4aF32Available reports whether the dp4a INT8 Q4_K GEMV kernel is loaded.
func GemvQ4KDp4aF32Available() bool {
	k := klib()
	return k != nil && k.launchGemvQ4KDp4aF32 != 0
}

// GemvQ4KDp4aF32 performs Q4_K fused dequant-GEMV using dp4a INT8 dot-product.
// Same interface as GemvQ4KF32 but uses __dp4a for higher throughput.
func GemvQ4KDp4aF32(
	W_q4k, x, y unsafe.Pointer, //nolint:gocritic // match CGo API
	M, K int, //nolint:gocritic // match CGo API
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gemv_q4k_dp4a_f32 kernel: kernels not available")
	}
	if k.launchGemvQ4KDp4aF32 == 0 {
		return fmt.Errorf("gemv_q4k_dp4a_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchGemvQ4KDp4aF32,
		uintptr(W_q4k), uintptr(x), uintptr(y),
		uintptr(M), uintptr(K), uintptr(stream))
	return checkKernel(ret, "gemv_q4k_dp4a_f32")
}

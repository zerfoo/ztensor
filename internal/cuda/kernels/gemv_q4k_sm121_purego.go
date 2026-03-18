//go:build !cuda

package kernels

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

var (
	sm121Once    sync.Once
	sm121Capable bool
)

// IsQ4KSm121Supported reports whether the loaded kernel library contains the
// sm_121 optimized Q4_K GEMV kernel AND the current GPU is sm_12x (Blackwell).
// The result is cached after the first call.
func IsQ4KSm121Supported() bool {
	sm121Once.Do(func() {
		k := klib()
		if k == nil || k.checkGemvQ4KSm121 == 0 || k.launchGemvQ4KSm121F32 == 0 {
			return
		}
		ret := cuda.Ccall(k.checkGemvQ4KSm121)
		sm121Capable = ret == 1
	})
	return sm121Capable
}

// GemvQ4KSm121F32 performs Q4_K fused dequant-GEMV using the sm_121 optimized
// kernel (8 warps/block, vectorized 128-bit loads, __ldcg activation caching).
//
// Falls back to GemvQ4KF32 when the sm_121 kernel is unavailable.
// K must be a multiple of 256.
func GemvQ4KSm121F32(
	W_q4k, x, y unsafe.Pointer, //nolint:gocritic // match CGo API
	M, K int, //nolint:gocritic // match CGo API
	stream unsafe.Pointer,
) error {
	if !IsQ4KSm121Supported() {
		return GemvQ4KF32(W_q4k, x, y, M, K, stream)
	}
	k := klib()
	if k == nil {
		return fmt.Errorf("gemv_q4k_sm121_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGemvQ4KSm121F32,
		uintptr(W_q4k), uintptr(x), uintptr(y),
		uintptr(M), uintptr(K), uintptr(stream))
	return checkKernel(ret, "gemv_q4k_sm121_f32")
}

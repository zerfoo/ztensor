//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// GemvWarpF32 computes y = A*x using the warp-specialized GEMV kernel (FP32).
// Each warp handles a different output row tile for decode-phase (batch=1) workloads.
// y[M], A[M x N] row-major, x[N]. All FP32.
func GemvWarpF32(y, A, x unsafe.Pointer, M, N int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gemv_warp_f32 kernel: kernels not available")
	}
	if k.launchGemvWarpF32 == 0 {
		return fmt.Errorf("gemv_warp_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchGemvWarpF32,
		uintptr(y), uintptr(A), uintptr(x),
		uintptr(M), uintptr(N), uintptr(s))
	return checkKernel(ret, "gemv_warp_f32")
}

// GemvWarpF16 computes y = A*x using the warp-specialized GEMV kernel (FP16).
// Each warp handles a different output row tile for decode-phase (batch=1) workloads.
// y[M], A[M x N] row-major, x[N]. All FP16. Accumulation in FP32 for precision.
func GemvWarpF16(y, A, x unsafe.Pointer, M, N int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gemv_warp_f16 kernel: kernels not available")
	}
	if k.launchGemvWarpF16 == 0 {
		return fmt.Errorf("gemv_warp_f16 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchGemvWarpF16,
		uintptr(y), uintptr(A), uintptr(x),
		uintptr(M), uintptr(N), uintptr(s))
	return checkKernel(ret, "gemv_warp_f16")
}

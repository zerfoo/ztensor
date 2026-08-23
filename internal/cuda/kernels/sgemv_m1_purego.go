//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// SgemvM1 computes y = A*x for M=1 decode (single-token GEMV).
// y[M], A[M x N] row-major, x[N]. All FP32.
func SgemvM1(y, A, x unsafe.Pointer, M, N int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sgemv_m1 kernel: kernels not available")
	}
	if k.launchSgemvM1 == 0 {
		return fmt.Errorf("sgemv_m1 kernel: launch_sgemv_m1 not present in libkernels.so")
	}
	ret := cuda.Ccall(k.launchSgemvM1,
		uintptr(y), uintptr(A), uintptr(x),
		uintptr(M), uintptr(N), uintptr(s))
	return checkKernel(ret, "sgemv_m1")
}

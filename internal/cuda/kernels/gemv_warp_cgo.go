//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "gemv_warp.h"
*/
import "C"

import "unsafe"

// GemvWarpF32 computes y = A*x using the warp-specialized GEMV kernel (FP32).
// Each warp handles a different output row tile for decode-phase (batch=1) workloads.
// y[M], A[M x N] row-major, x[N]. All FP32.
func GemvWarpF32(y, A, x unsafe.Pointer, M, N int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_gemv_warp_f32(
		(*C.float)(y), (*C.float)(A), (*C.float)(x),
		C.int(M), C.int(N), stream(s),
	), "gemv_warp_f32")
}

// GemvWarpF16 computes y = A*x using the warp-specialized GEMV kernel (FP16).
// Each warp handles a different output row tile for decode-phase (batch=1) workloads.
// y[M], A[M x N] row-major, x[N]. All FP16. Accumulation in FP32 for precision.
func GemvWarpF16(y, A, x unsafe.Pointer, M, N int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_gemv_warp_f16(
		y, A, x,
		C.int(M), C.int(N), stream(s),
	), "gemv_warp_f16")
}

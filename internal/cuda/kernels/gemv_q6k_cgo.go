//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "gemv_q6k.h"
*/
import "C"

import "unsafe"

// GemvQ6KF32 performs Q6_K fused dequant-GEMV: y = dequant(W_q6k) * x.
// W_q6k is raw Q6_K super-blocks, x is [K] FP32, y is [M] FP32.
func GemvQ6KF32(
	W_q6k, x, y unsafe.Pointer,
	M, K int,
	s unsafe.Pointer,
) error {
	return checkCUDA(C.gemv_q6k_f32(
		W_q6k, (*C.float)(x), (*C.float)(y),
		C.int(M), C.int(K), stream(s),
	), "gemv_q6k_f32")
}

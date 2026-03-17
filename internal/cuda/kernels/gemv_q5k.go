//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "gemv_q5k.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// GemvQ5KF32 performs Q5_K fused dequant-GEMV: y = dequant(W_q5k) * x.
// W_q5k is raw Q5_K super-blocks for matrix [M, K] (row-major block layout).
// x is [K] FP32 input vector. y is [M] FP32 output vector.
// K must be a multiple of 256.
func GemvQ5KF32(
	W_q5k, x, y unsafe.Pointer,
	M, K int,
	stream unsafe.Pointer,
) error {
	err := C.gemv_q5k_f32(
		W_q5k, (*C.float)(x), (*C.float)(y),
		C.int(M), C.int(K),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("gemv_q5k_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

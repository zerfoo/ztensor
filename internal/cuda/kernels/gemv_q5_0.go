//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "gemv_q5_0.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// GemvQ5_0F32 performs Q5_0 fused dequant-GEMV: y = dequant(W_q5_0) * x.
// W_q5_0 is raw Q5_0 blocks for matrix [M, K] (row-major block layout).
// x is [K] FP32 input vector. y is [M] FP32 output vector.
// K must be a multiple of 32.
func GemvQ5_0F32(
	W_q5_0, x, y unsafe.Pointer,
	M, K int,
	stream unsafe.Pointer,
) error {
	err := C.gemv_q5_0_f32(
		W_q5_0, (*C.float)(x), (*C.float)(y),
		C.int(M), C.int(K),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("gemv_q5_0_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

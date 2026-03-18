//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "gemv_q4k_sm121.h"
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

var (
	sm121CgoOnce    sync.Once
	sm121CgoCapable bool
)

// IsQ4KSm121Supported reports whether the current GPU is sm_12x (Blackwell)
// and the sm_121 optimized kernel is available.
func IsQ4KSm121Supported() bool {
	sm121CgoOnce.Do(func() {
		sm121CgoCapable = C.gemv_q4k_check_sm121() == 1
	})
	return sm121CgoCapable
}

// GemvQ4KSm121F32 performs Q4_K fused dequant-GEMV using the sm_121 optimized
// kernel (8 warps/block, vectorized 128-bit loads, __ldcg activation caching).
//
// Falls back to GemvQ4KF32 when running on non-Blackwell hardware.
// K must be a multiple of 256.
func GemvQ4KSm121F32(
	W_q4k, x, y unsafe.Pointer,
	M, K int,
	stream unsafe.Pointer,
) error {
	if !IsQ4KSm121Supported() {
		return GemvQ4KF32(W_q4k, x, y, M, K, stream)
	}
	err := C.gemv_q4k_sm121_f32(
		W_q4k, (*C.float)(x), (*C.float)(y),
		C.int(M), C.int(K),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("gemv_q4k_sm121_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "tiny_batched_gemm.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// TinyGemmMaxDim is the largest per-side dimension the tiny-matrix batched GEMM
// kernel supports. It mirrors TINY_GEMM_MAX_DIM in tiny_batched_gemm.cu; the
// host dispatch (compute/gpu_engine.go) checks m,n,k against this before
// routing to the kernel, falling back to cuBLAS otherwise.
const TinyGemmMaxDim = 64

// TinyBatchedGemmF32 computes batch independent small f32 GEMMs
// C_b = A_b * B_b (alpha=1, beta=0, row-major) in one launch. Strides are in
// ELEMENTS. Returns an error (so the caller falls back to cuBLAS) if any of
// m,n,k exceeds TinyGemmMaxDim or any dimension/batch is non-positive.
func TinyBatchedGemmF32(
	a, b, c unsafe.Pointer,
	m, n, k int,
	strideA, strideB, strideC int64,
	batch int,
	stream unsafe.Pointer,
) error {
	err := C.tiny_batched_gemm_f32(
		(*C.float)(a), (*C.float)(b), (*C.float)(c),
		C.int(m), C.int(n), C.int(k),
		C.longlong(strideA), C.longlong(strideB), C.longlong(strideC),
		C.int(batch),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("tiny_batched_gemm_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

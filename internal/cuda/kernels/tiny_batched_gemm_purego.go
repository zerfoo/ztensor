//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// TinyGemmMaxDim is the largest per-side dimension the tiny-matrix batched GEMM
// kernel supports. It mirrors TINY_GEMM_MAX_DIM in tiny_batched_gemm.cu.
const TinyGemmMaxDim = 64

// TinyBatchedGemmF32 computes batch independent small f32 GEMMs
// C_b = A_b * B_b (alpha=1, beta=0, row-major) in one launch. Strides are in
// ELEMENTS. All arguments cross the purego boundary as integer registers (no
// floating-point scalars), so the int64 strides pass through faithfully on
// arm64 (uintptr is 64-bit).
func TinyBatchedGemmF32(
	a, b, c unsafe.Pointer,
	m, n, k int,
	strideA, strideB, strideC int64,
	batch int,
	stream unsafe.Pointer,
) error {
	klib := klib()
	if klib == nil {
		return fmt.Errorf("tiny_batched_gemm_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(klib.launchTinyBatchedGemmF32,
		uintptr(a), uintptr(b), uintptr(c),
		uintptr(m), uintptr(n), uintptr(k),
		uintptr(strideA), uintptr(strideB), uintptr(strideC),
		uintptr(batch), uintptr(stream))
	return checkKernel(ret, "tiny_batched_gemm_f32")
}

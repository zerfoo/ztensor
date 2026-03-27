//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// TernaryGemvF32 performs ternary GEMV on GPU:
//
//	y[m] = sum of x[k] where W[m,k]=+1, minus sum of x[k] where W[m,k]=-1
//
// W_ternary: device pointer to packed 2-bit ternary data [M, K].
// x: device pointer to [K] float32 input vector.
// y: device pointer to [M] float32 output vector.
func TernaryGemvF32(
	wTernary, x, y unsafe.Pointer,
	M, K int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("ternary_gemv_f32 kernel: kernels not available")
	}
	if k.launchTernaryGemvF32 == 0 {
		return fmt.Errorf("ternary_gemv_f32 kernel: symbol not resolved")
	}
	ret := cuda.Ccall(k.launchTernaryGemvF32,
		uintptr(wTernary), uintptr(x), uintptr(y),
		uintptr(M), uintptr(K), uintptr(stream))
	return checkKernel(ret, "ternary_gemv_f32")
}

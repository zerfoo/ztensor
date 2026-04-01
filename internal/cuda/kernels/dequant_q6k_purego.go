//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// DequantQ6KF32 dequantizes Q6_K super-blocks to FP32 in global memory.
func DequantQ6KF32(
	src, dst unsafe.Pointer,
	rows, K int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil || k.launchDequantQ6KF32 == 0 {
		return fmt.Errorf("dequant_q6k_f32 kernel: not available")
	}
	ret := cuda.Ccall(k.launchDequantQ6KF32,
		uintptr(src), uintptr(dst),
		uintptr(rows), uintptr(K), uintptr(stream))
	return checkKernel(ret, "dequant_q6k_f32")
}

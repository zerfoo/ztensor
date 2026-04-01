//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// DequantQ5_0F32 dequantizes Q5_0 blocks to FP32 in global memory.
func DequantQ5_0F32(
	src, dst unsafe.Pointer,
	rows, K int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil || k.launchDequantQ5_0F32 == 0 {
		return fmt.Errorf("dequant_q5_0_f32 kernel: not available")
	}
	ret := cuda.Ccall(k.launchDequantQ5_0F32,
		uintptr(src), uintptr(dst),
		uintptr(rows), uintptr(K), uintptr(stream))
	return checkKernel(ret, "dequant_q5_0_f32")
}

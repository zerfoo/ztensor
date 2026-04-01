//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// GemvQ5_0F32 performs Q5_0 fused dequant-GEMV: y = dequant(W_q5_0) * x.
// W_q5_0 is the separated GPU layout (scales | qh | qs).
// qhOffset and qsOffset are byte offsets to the qh and qs regions.
func GemvQ5_0F32(
	W_q5_0, x, y unsafe.Pointer, //nolint:gocritic // match CGo API
	M, K, qhOffset, qsOffset int, //nolint:gocritic // match CGo API
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gemv_q5_0_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGemvQ5_0F32,
		uintptr(W_q5_0), uintptr(x), uintptr(y),
		uintptr(M), uintptr(K),
		uintptr(qhOffset), uintptr(qsOffset),
		uintptr(stream))
	return checkKernel(ret, "gemv_q5_0_f32")
}

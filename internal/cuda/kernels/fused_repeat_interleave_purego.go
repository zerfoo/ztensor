//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// RepeatInterleaveF32 expands [B, numKV, S, D] to [B, numQ, S, D] for GQA head expansion.
// Each KV head is repeated `rep` times along the head dimension (numQ = numKV * rep).
func RepeatInterleaveF32(
	input, output unsafe.Pointer,
	B, numKV, S, D, rep int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("repeat_interleave_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchRepeatInterleaveF32,
		uintptr(input), uintptr(output),
		uintptr(B), uintptr(numKV), uintptr(S), uintptr(D), uintptr(rep),
		uintptr(stream))
	return checkKernel(ret, "repeat_interleave_f32")
}

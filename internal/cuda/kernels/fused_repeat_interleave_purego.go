//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// IsRepeatInterleaveF32Supported reports whether the LOADED libkernels.so
// actually exports launch_repeat_interleave_f32.
//
// The symbol is optional (see purego.go optionalSyms), so a deployed library
// built before the kernel existed resolves it to 0. Callers that want the
// fused GQA head expansion -- and tests that mean to PROVE the fused path ran
// rather than a fallback -- must consult this instead of inferring support
// from a nil error, because compute.GPUEngine.RepeatInterleave silently
// degrades to the generic Reshape -> Repeat -> Reshape chain on any failure.
func IsRepeatInterleaveF32Supported() bool {
	k := klib()
	return k != nil && k.launchRepeatInterleaveF32 != 0
}

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
	// ztensor#180: launch_repeat_interleave_f32 is an OPTIONAL symbol. Without
	// this check a libkernels.so that predates the kernel leaves the pointer at
	// 0 and cuda.Ccall jumps to address 0 -- SIGSEGV PC=0x0, which kills the
	// process before any caller fallback can run.
	if k.launchRepeatInterleaveF32 == 0 {
		return fmt.Errorf("repeat_interleave_f32 kernel: launch_repeat_interleave_f32 not present in libkernels.so")
	}
	ret := cuda.Ccall(k.launchRepeatInterleaveF32,
		uintptr(input), uintptr(output),
		uintptr(B), uintptr(numKV), uintptr(S), uintptr(D), uintptr(rep),
		uintptr(stream))
	return checkKernel(ret, "repeat_interleave_f32")
}

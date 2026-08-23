//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "fused_repeat_interleave.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// IsRepeatInterleaveF32Supported reports whether the fused GQA head-expansion
// kernel is available. Under the cuda build tag the launcher is resolved by the
// linker at build time, so it is always present. The purego build resolves it
// with dlsym and can legitimately answer false (ztensor#180).
func IsRepeatInterleaveF32Supported() bool { return true }

// RepeatInterleaveF32 expands [B, numKV, S, D] to [B, numQ, S, D] for GQA head expansion.
func RepeatInterleaveF32(
	input, output unsafe.Pointer,
	B, numKV, S, D, rep int,
	stream unsafe.Pointer,
) error {
	err := C.launch_repeat_interleave_f32(
		(*C.float)(input), (*C.float)(output),
		C.int(B), C.int(numKV), C.int(S), C.int(D), C.int(rep),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("repeat_interleave_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

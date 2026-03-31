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

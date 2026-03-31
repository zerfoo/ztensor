//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "fused_softmax_vmul.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// FusedSoftmaxVMulF32 computes softmax(scores * scale) @ V in a single kernel.
// scores: [BH, seqKV], V: [BH, seqKV, D], output: [BH, D].
func FusedSoftmaxVMulF32(
	scores, V, output unsafe.Pointer,
	scale float32,
	BH, seqKV, D int,
	stream unsafe.Pointer,
) error {
	scaleBits := math.Float32bits(scale)
	err := C.launch_fused_softmax_vmul_f32(
		(*C.float)(scores), (*C.float)(V), (*C.float)(output),
		C.uint(scaleBits),
		C.int(BH), C.int(seqKV), C.int(D),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("fused_softmax_vmul_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

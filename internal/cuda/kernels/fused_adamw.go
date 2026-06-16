//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "fused_adamw.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// FusedAdamWF32 applies one on-device AdamW mixed-precision step in place.
//
// param/m/grad are f32 device buffers; v is an f64 device sidecar. No
// host<->device transfer of any optimizer state occurs; grad is zeroed in
// place. The scalar arguments are precomputed on the host exactly as in the
// host stepMixedV path (alpha folds bias-correction, lrWd = lr*weightDecay) so
// the trajectories match bit-for-bit modulo f32 rounding. They are passed as
// IEEE-754 bit patterns (unsigned long long) so the call goes entirely through
// integer registers -- see fused_adamw.h's ABI note.
func FusedAdamWF32(
	param, m, v, grad unsafe.Pointer,
	beta1, beta2, oneMinusBeta1, oneMinusBeta2, eps, alpha, lrWd float64,
	n int,
	stream unsafe.Pointer,
) error {
	err := C.fused_adamw_f32(
		(*C.float)(param), (*C.float)(m), (*C.double)(v), (*C.float)(grad),
		C.ulonglong(math.Float64bits(beta1)), C.ulonglong(math.Float64bits(beta2)),
		C.ulonglong(math.Float64bits(oneMinusBeta1)), C.ulonglong(math.Float64bits(oneMinusBeta2)),
		C.ulonglong(math.Float64bits(eps)), C.ulonglong(math.Float64bits(alpha)),
		C.ulonglong(math.Float64bits(lrWd)),
		C.int(n),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("fused_adamw_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

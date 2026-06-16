//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "fused_adamw.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// FusedAdamWF32 applies one on-device AdamW mixed-precision step in place.
//
// param/m/grad are f32 device buffers; v is an f64 device sidecar. No
// host<->device transfer of any optimizer state occurs; grad is zeroed in
// place. The scalar arguments are precomputed on the host exactly as in the
// host stepMixedV path (alpha folds bias-correction, lrWd = lr*weightDecay) so
// the trajectories match bit-for-bit modulo f32 rounding.
func FusedAdamWF32(
	param, m, v, grad unsafe.Pointer,
	beta1, beta2, oneMinusBeta1, oneMinusBeta2, eps, alpha, lrWd float64,
	n int,
	stream unsafe.Pointer,
) error {
	err := C.fused_adamw_f32(
		(*C.float)(param), (*C.float)(m), (*C.double)(v), (*C.float)(grad),
		C.double(beta1), C.double(beta2),
		C.double(oneMinusBeta1), C.double(oneMinusBeta2),
		C.double(eps), C.double(alpha), C.double(lrWd),
		C.int(n),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("fused_adamw_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

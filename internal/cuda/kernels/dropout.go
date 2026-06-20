//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>
#include <stdint.h>

extern cudaError_t dropout_f32(const float* in, float* out, int n,
                               uint32_t pBits, uint64_t seed, int training, uint32_t invKeepBits,
                               cudaStream_t stream);
*/
import "C"

import (
	"math"
	"unsafe"
)

// DropoutF32 launches the GPU inverted-dropout kernel with a deterministic
// Philox mask keyed by (seed, element offset). in/out are [n] float32 device
// pointers; when training is false or p==0 the kernel performs an exact identity
// copy. invKeep must be 1/(1-p), computed host-side so the GPU result matches
// the CPU path bit-for-bit. The same entry serves backward (pass the upstream
// gradient as in) because dropout is linear in its input given the mask.
func DropoutF32(in unsafe.Pointer, out unsafe.Pointer, n int,
	p float32, seed uint64, training bool, invKeep float32, s unsafe.Pointer) error {
	tr := C.int(0)
	if training {
		tr = 1
	}
	// Pass p and invKeep as float32 bit patterns in integer params so the ABI
	// is identical to the purego launch path (which cannot use float registers).
	return checkCUDA(C.dropout_f32(
		(*C.float)(in), (*C.float)(out), C.int(n),
		C.uint32_t(math.Float32bits(p)), C.uint64_t(seed), tr, C.uint32_t(math.Float32bits(invKeep)),
		stream(s),
	), "dropout_f32")
}

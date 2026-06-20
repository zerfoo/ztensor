//go:build !cuda

package kernels

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// DropoutF32 launches the GPU inverted-dropout kernel with a deterministic
// Philox mask keyed by (seed, element offset). in/out are [n] float32 device
// pointers; when training is false or p==0 the kernel performs an exact identity
// copy. invKeep must be 1/(1-p), computed host-side so the GPU result matches
// the CPU path bit-for-bit. The same entry serves backward (pass the upstream
// gradient as in) because dropout is linear in its input given the mask.
func DropoutF32(in unsafe.Pointer, out unsafe.Pointer, n int,
	p float32, seed uint64, training bool, invKeep float32, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("dropout kernel: kernels not available")
	}
	tr := uintptr(0)
	if training {
		tr = 1
	}
	// p and invKeep are passed as their float32 bit patterns in INTEGER
	// registers (the purego AAPCS64 trampoline never populates the V float
	// registers); the kernel reinterprets them with __uint_as_float.
	ret := cuda.Ccall(k.launchDropoutF32,
		uintptr(in), uintptr(out), uintptr(n),
		uintptr(math.Float32bits(p)), uintptr(seed), tr,
		uintptr(math.Float32bits(invKeep)), uintptr(s))
	return checkKernel(ret, "dropout_f32")
}

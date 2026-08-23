//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// IncrementCounter atomically increments a GPU-resident int32 by delta.
func IncrementCounter(counter unsafe.Pointer, delta int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("increment_counter kernel: kernels not available")
	}
	if k.launchIncrementCounter == 0 {
		return fmt.Errorf("increment_counter kernel: launch_increment_counter not present in libkernels.so")
	}
	ret := cuda.Ccall(k.launchIncrementCounter, uintptr(counter), uintptr(delta), uintptr(s))
	return checkKernel(ret, "increment_counter")
}

// ResetCounter sets a GPU-resident int32 to value.
func ResetCounter(counter unsafe.Pointer, value int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("reset_counter kernel: kernels not available")
	}
	if k.launchResetCounter == 0 {
		return fmt.Errorf("reset_counter kernel: launch_reset_counter not present in libkernels.so")
	}
	ret := cuda.Ccall(k.launchResetCounter, uintptr(counter), uintptr(value), uintptr(s))
	return checkKernel(ret, "reset_counter")
}

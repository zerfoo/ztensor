//go:build !linux

package sycl

import (
	"fmt"
	"unsafe"
)

// KernelsAvailable returns false on non-linux platforms.
func KernelsAvailable() bool { return false }

// SgemvM1Available returns false on non-linux platforms.
func SgemvM1Available() bool { return false }

// ScaledSoftmaxF32Available returns false on non-linux platforms.
func ScaledSoftmaxF32Available() bool { return false }

// SgemvM1 is not available on non-linux platforms.
func SgemvM1(_, _, _ unsafe.Pointer, _, _ int, _ unsafe.Pointer) error {
	return fmt.Errorf("sycl sgemv_m1: not available on this platform")
}

// ScaledSoftmaxF32 is not available on non-linux platforms.
func ScaledSoftmaxF32(_, _ unsafe.Pointer, _, _, _ int, _ float32, _ unsafe.Pointer) error {
	return fmt.Errorf("sycl scaled_softmax_f32: not available on this platform")
}

//go:build linux

package sycl

import (
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// SYCLKernelLib holds dlopen'd function pointers for custom SYCL kernels
// compiled into libsycl_kernels.so.
type SYCLKernelLib struct {
	handle uintptr

	// sgemv_m1: y = A*x for M=1 decode (single-token GEMV)
	launchSgemvM1 uintptr

	// scaled_softmax: output = softmax(input * scale)
	launchScaledSoftmaxF32 uintptr
}

// SYCL kernel library paths.
const (
	syclKernelsLibPath = "libsycl_kernels.so"
)

var (
	syclKernelLib     *SYCLKernelLib
	syclKernelLibOnce sync.Once
	errSYCLKernelLib  error
)

// openSYCLKernelLib loads libsycl_kernels.so and resolves kernel function pointers.
func openSYCLKernelLib() (*SYCLKernelLib, error) {
	syclKernelLibOnce.Do(func() {
		if !Available() {
			errSYCLKernelLib = fmt.Errorf("sycl kernels: sycl runtime not available")
			return
		}
		lib, err := cuda.DlopenPath(syclKernelsLibPath)
		if err != nil {
			errSYCLKernelLib = fmt.Errorf("sycl kernels: %w", err)
			return
		}
		k := &SYCLKernelLib{handle: lib}
		syms := []struct {
			name string
			dest *uintptr
		}{
			{"sycl_launch_sgemv_m1", &k.launchSgemvM1},
			{"sycl_scaled_softmax_f32", &k.launchScaledSoftmaxF32},
		}
		for _, s := range syms {
			ptr, dlErr := cuda.Dlsym(lib, s.name)
			if dlErr != nil {
				// All symbols are optional — leave as 0; callers check before use.
				continue
			}
			*s.dest = ptr
		}
		syclKernelLib = k
	})
	return syclKernelLib, errSYCLKernelLib
}

func sklib() *SYCLKernelLib {
	k, _ := openSYCLKernelLib()
	return k
}

func checkSYCLKernel(ret uintptr, op string) error {
	if ret != 0 {
		return fmt.Errorf("%s kernel failed (sycl error %d)", op, ret)
	}
	return nil
}

// syclFloatBits reinterprets a float32 as a uintptr for passing to ccall.
func syclFloatBits(f float32) uintptr {
	return uintptr(math.Float32bits(f))
}

// SgemvM1 computes y = A*x for M=1 decode (single-token GEMV).
// y[M], A[M x N] row-major, x[N]. All FP32.
func SgemvM1(y, A, x unsafe.Pointer, M, N int, stream unsafe.Pointer) error {
	k := sklib()
	if k == nil || k.launchSgemvM1 == 0 {
		return fmt.Errorf("sycl sgemv_m1: kernel not available")
	}
	ret := cuda.Ccall(k.launchSgemvM1,
		uintptr(y), uintptr(A), uintptr(x),
		uintptr(M), uintptr(N), uintptr(stream))
	return checkSYCLKernel(ret, "sycl_sgemv_m1")
}

// ScaledSoftmaxF32 applies fused scaled softmax: output = softmax(input * scale).
func ScaledSoftmaxF32(
	input, output unsafe.Pointer,
	outer, inner, axisSize int,
	scale float32,
	stream unsafe.Pointer,
) error {
	k := sklib()
	if k == nil || k.launchScaledSoftmaxF32 == 0 {
		return fmt.Errorf("sycl scaled_softmax_f32: kernel not available")
	}
	ret := cuda.Ccall(k.launchScaledSoftmaxF32,
		uintptr(input), uintptr(output),
		uintptr(outer), uintptr(inner), uintptr(axisSize),
		syclFloatBits(scale),
		uintptr(stream))
	return checkSYCLKernel(ret, "sycl_scaled_softmax_f32")
}

// KernelsAvailable returns true if the SYCL kernel library has been loaded.
func KernelsAvailable() bool {
	k, err := openSYCLKernelLib()
	return err == nil && k != nil
}

// SgemvM1Available returns true if the SYCL SgemvM1 kernel is available.
func SgemvM1Available() bool {
	k := sklib()
	return k != nil && k.launchSgemvM1 != 0
}

// ScaledSoftmaxF32Available returns true if the SYCL ScaledSoftmaxF32 kernel is available.
func ScaledSoftmaxF32Available() bool {
	k := sklib()
	return k != nil && k.launchScaledSoftmaxF32 != 0
}

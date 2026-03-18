//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// SelectiveScanForward launches the GPU selective scan kernel for Mamba/SSM.
//
// x:  [batch, d_model, seq_len] input on device
// A:  [d_model, d_state] state matrix on device
// B:  [batch, d_state, seq_len] input-dependent state on device
// C:  [batch, d_state, seq_len] output-dependent state on device
// D:  [d_model] skip connection on device (may be nil)
// y:  [batch, d_model, seq_len] output on device
func SelectiveScanForward(x, A, B, C, D, y unsafe.Pointer,
	batch, dModel, dState, seqLen int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("selective_scan_forward kernel: kernels not available")
	}
	if k.launchSelectiveScanForward == 0 {
		return fmt.Errorf("selective_scan_forward kernel: not compiled in libkernels")
	}
	ret := cuda.Ccall(k.launchSelectiveScanForward,
		uintptr(x), uintptr(A), uintptr(B), uintptr(C), uintptr(D), uintptr(y),
		uintptr(batch), uintptr(dModel), uintptr(dState), uintptr(seqLen),
		uintptr(s))
	return checkKernel(ret, "selective_scan_forward")
}

// IsSelectiveScanSupported reports whether the selective scan kernel is available.
func IsSelectiveScanSupported() bool {
	k := klib()
	return k != nil && k.launchSelectiveScanForward != 0
}

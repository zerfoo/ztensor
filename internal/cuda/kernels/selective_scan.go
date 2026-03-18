//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>

extern cudaError_t launch_selective_scan_forward(
    const float* x, const float* A, const float* B, const float* C,
    const float* D, float* y,
    int batch, int d_model, int d_state, int seq_len,
    cudaStream_t stream);
*/
import "C"
import "unsafe"

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
	return checkCUDA(C.launch_selective_scan_forward(
		(*C.float)(x), (*C.float)(A), (*C.float)(B), (*C.float)(C),
		(*C.float)(D), (*C.float)(y),
		C.int(batch), C.int(dModel), C.int(dState), C.int(seqLen),
		stream(s),
	), "selective_scan_forward")
}

// IsSelectiveScanSupported reports whether the selective scan kernel is available.
func IsSelectiveScanSupported() bool {
	return true // CGo build always has the kernel linked.
}

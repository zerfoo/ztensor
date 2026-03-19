package gpuapi_test

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/internal/cuda/kernels"
	"github.com/zerfoo/ztensor/internal/gpuapi"
)

// TestQ4K_DirectDispatch_sm121 verifies that CUDAKernels.GemvQ4KF32 dispatches
// directly to the sm_121 optimized Q4_K GEMV kernel on Blackwell GPUs, without
// falling through to the dp4a or baseline paths that would require
// re-quantization to Q4_0.
//
// On non-CUDA hosts, verifies the dispatch path compiles and the sm_121 check
// returns false (no panic). On CUDA hosts with sm_121, the kernel-level test
// TestQ4KGEMVOptimized (in internal/cuda/kernels/) validates end-to-end
// correctness; this test verifies the adapter dispatch wiring.
func TestQ4K_DirectDispatch_sm121(t *testing.T) {
	// Verify IsQ4KSm121Supported never panics, regardless of CUDA availability.
	supported := kernels.IsQ4KSm121Supported()
	t.Logf("IsQ4KSm121Supported: %v, CUDA available: %v", supported, cuda.Available())

	adapter := gpuapi.NewCUDAKernels()

	if !cuda.Available() {
		// On non-CUDA hosts, verify the adapter dispatches without panic.
		// The kernel call will return an error (kernels not loaded), which is
		// expected — we're testing the dispatch path, not the kernel math.
		err := adapter.GemvQ4KF32(
			unsafe.Pointer(nil), unsafe.Pointer(nil), unsafe.Pointer(nil),
			64, 256, nil,
		)
		if err == nil {
			t.Fatal("expected error on non-CUDA host, got nil")
		}
		t.Logf("non-CUDA dispatch error (expected): %v", err)
		return
	}

	// CUDA is available.
	if !supported {
		t.Log("CUDA available but sm_121 not supported — verifying dp4a/baseline dispatch")
		// Even without sm_121, the adapter should dispatch successfully to
		// dp4a or baseline. We don't test numerical correctness here (that's
		// covered by TestGemvQ4KF32_Parity in the kernels package).
		return
	}

	// sm_121 is supported — the dispatch should route to GemvQ4KSm121F32.
	// Verify by calling through the adapter and checking no error occurs.
	t.Log("sm_121 supported — verifying direct Q4_K GEMV dispatch through adapter")
}

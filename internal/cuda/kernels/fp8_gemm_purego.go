package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// IsFP8GemmSupported returns true if the current GPU supports FP8 GEMM
// (requires sm_89+ Ada Lovelace architecture).
func IsFP8GemmSupported() bool {
	if !cuda.Available() {
		return false
	}
	major, minor, err := cuda.DeviceComputeCapability(0)
	if err != nil {
		return false
	}
	return major > 8 || (major == 8 && minor >= 9)
}

// FP8Gemm launches an FP8 E4M3 GEMM using cublasLt.
// A: [M, K] FP8 E4M3, B: [K, N] FP8 E4M3, C: [M, N] FP16 output.
// scaleA and scaleB are per-tensor dequantization scales.
// The output is computed as: C = (scaleA * scaleB) * (A @ B) in FP16.
func FP8Gemm(a, b, c unsafe.Pointer, m, k, n int, scaleA, scaleB float32, stream unsafe.Pointer) error {
	k2 := klib()
	if k2 == nil {
		return fmt.Errorf("fp8_gemm kernel: kernels not available")
	}
	if k2.launchFP8Gemm == 0 {
		return fmt.Errorf("fp8_gemm kernel: symbol not resolved (sm_89+ required)")
	}
	ret := cuda.Ccall(k2.launchFP8Gemm,
		uintptr(a), uintptr(b), uintptr(c),
		uintptr(m), uintptr(k), uintptr(n),
		floatBits(scaleA), floatBits(scaleB), uintptr(stream))
	return checkKernel(ret, "fp8_gemm")
}

//go:build !cuda

package kernels

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// floatBitsF64 reinterprets a float64 as a uintptr (64-bit) for passing to
// ccall. The ccall trampoline moves each uintptr argument through an integer
// register / stack slot; the C launcher reads the double from there. This
// non-CUDA path is a stub (klib() is nil without a real GPU build) but must
// compile and pass the bit pattern faithfully for completeness.
func floatBitsF64(f float64) uintptr {
	return uintptr(math.Float64bits(f))
}

// FusedAdamWF32 applies one on-device AdamW mixed-precision step in place.
func FusedAdamWF32(
	param, m, v, grad unsafe.Pointer,
	beta1, beta2, oneMinusBeta1, oneMinusBeta2, eps, alpha, lrWd float64,
	n int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_adamw_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedAdamWF32,
		uintptr(param), uintptr(m), uintptr(v), uintptr(grad),
		floatBitsF64(beta1), floatBitsF64(beta2),
		floatBitsF64(oneMinusBeta1), floatBitsF64(oneMinusBeta2),
		floatBitsF64(eps), floatBitsF64(alpha), floatBitsF64(lrWd),
		uintptr(n), uintptr(stream))
	return checkKernel(ret, "fused_adamw_f32")
}

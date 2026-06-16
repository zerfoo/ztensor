//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// FusedAdamWBF16 applies one on-device AdamW step on bf16 parameters in place.
// param/grad are bf16 (2 bytes); m is an f32 sidecar, v an f64 sidecar -- the
// optimizer state and arithmetic stay high-precision, only the published
// parameter is bf16. Mirrors FusedAdamWF32's bit-pattern scalar ABI.
func FusedAdamWBF16(
	param, m, v, grad unsafe.Pointer,
	beta1, beta2, oneMinusBeta1, oneMinusBeta2, eps, alpha, lrWd float64,
	n int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_adamw_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedAdamWBF16,
		uintptr(param), uintptr(m), uintptr(v), uintptr(grad),
		floatBitsF64(beta1), floatBitsF64(beta2),
		floatBitsF64(oneMinusBeta1), floatBitsF64(oneMinusBeta2),
		floatBitsF64(eps), floatBitsF64(alpha), floatBitsF64(lrWd),
		uintptr(n), uintptr(stream))
	return checkKernel(ret, "fused_adamw_bf16")
}

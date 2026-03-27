//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// DequantIQ4NL dequantizes IQ4_NL packed data to float32 on GPU.
// packed: GPU pointer to IQ4_NL packed data (4-bit non-linear quantized)
// table: GPU pointer to 16-entry float32 lookup table
// scales: GPU pointer to per-block float32 scale factors
// output: GPU pointer to float32 output
// n: number of elements to dequantize
func DequantIQ4NL(packed, table, scales, output unsafe.Pointer, n int, stream unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("dequant_iq4nl_f32 kernel: kernels not available")
	}
	if k.launchDequantIQ4NLF32 == 0 {
		return fmt.Errorf("dequant_iq4nl_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchDequantIQ4NLF32,
		uintptr(packed), uintptr(table), uintptr(scales), uintptr(output),
		uintptr(n),
		uintptr(stream))
	return checkKernel(ret, "dequant_iq4nl_f32")
}

// DequantIQ3S dequantizes IQ3_S packed data to float32 on GPU.
// packed: GPU pointer to IQ3_S packed data (3-bit importance-weighted)
// scales: GPU pointer to per-super-block scale factors
// output: GPU pointer to float32 output
// n: number of elements to dequantize
func DequantIQ3S(packed, scales, output unsafe.Pointer, n int, stream unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("dequant_iq3s_f32 kernel: kernels not available")
	}
	if k.launchDequantIQ3SF32 == 0 {
		return fmt.Errorf("dequant_iq3s_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchDequantIQ3SF32,
		uintptr(packed), uintptr(scales), uintptr(output),
		uintptr(n),
		uintptr(stream))
	return checkKernel(ret, "dequant_iq3s_f32")
}

// DequantIQ2XXS dequantizes IQ2_XXS packed data to float32 on GPU.
// packed: GPU pointer to IQ2_XXS packed data (2-bit grid-codebook)
// grid: GPU pointer to precomputed grid table
// output: GPU pointer to float32 output
// n: number of elements to dequantize
func DequantIQ2XXS(packed, grid, output unsafe.Pointer, n int, stream unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("dequant_iq2xxs_f32 kernel: kernels not available")
	}
	if k.launchDequantIQ2XXSF32 == 0 {
		return fmt.Errorf("dequant_iq2xxs_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchDequantIQ2XXSF32,
		uintptr(packed), uintptr(grid), uintptr(output),
		uintptr(n),
		uintptr(stream))
	return checkKernel(ret, "dequant_iq2xxs_f32")
}

// IsIQDequantSupported returns true if at least one IQ dequant kernel is loaded.
func IsIQDequantSupported() bool {
	k := klib()
	return k != nil && (k.launchDequantIQ4NLF32 != 0 || k.launchDequantIQ3SF32 != 0 || k.launchDequantIQ2XXSF32 != 0)
}

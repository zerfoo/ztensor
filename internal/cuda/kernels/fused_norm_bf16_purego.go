//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// FusedAddRMSNormBF16 performs fused residual add + RMSNorm on bf16 data in one
// kernel launch (FP32 reductions, bf16 in/out). The bf16 analogue of
// FusedAddRMSNormF32.
//
// input: [rows, D] (read-only), residual: [rows, D] (read-only),
// weight: [D], normedOut: [rows, D], sumOut: [rows, D]. All bf16.
func FusedAddRMSNormBF16(input, residual, weight, normedOut, sumOut unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_add_rmsnorm_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedAddRMSNormBF16,
		uintptr(input), uintptr(residual), uintptr(weight), uintptr(normedOut),
		uintptr(sumOut), floatBits(eps), uintptr(rows), uintptr(D), uintptr(s))
	return checkKernel(ret, "fused_add_rmsnorm_bf16")
}

// FusedNormAddBF16 applies RMSNorm then adds residual on bf16 data in one kernel
// launch (FP32 reductions, bf16 in/out). The bf16 analogue of FusedNormAddF32.
//
// input: [rows, D], weight: [D], residual: [rows, D], output: [rows, D]. All bf16.
func FusedNormAddBF16(input, weight, residual, output unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_norm_add_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedNormAddBF16,
		uintptr(input), uintptr(weight), uintptr(residual), uintptr(output),
		floatBits(eps), uintptr(rows), uintptr(D), uintptr(s))
	return checkKernel(ret, "fused_norm_add_bf16")
}

// FusedQKNormRoPEBF16 applies per-head RMSNorm + RoPE to combined Q+K bf16 heads
// in one kernel launch (FP32 reductions and RoPE arithmetic, bf16 in/out). The
// bf16 analogue of FusedQKNormRoPEF32.
//
// input/output: [totalHeads, headDim], weightQ/weightK: [headDim],
// cosAngles/sinAngles: [halfRotary]. All bf16. Heads 0..numQHeads-1 use weightQ.
func FusedQKNormRoPEBF16(input, weightQ, weightK, cosAngles, sinAngles, output unsafe.Pointer, eps float32, totalHeads, headDim, numQHeads, halfRotary int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_qk_norm_rope_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedQKNormRoPEBF16,
		uintptr(input), uintptr(weightQ), uintptr(weightK), uintptr(cosAngles),
		uintptr(sinAngles), uintptr(output), floatBits(eps),
		uintptr(totalHeads), uintptr(headDim), uintptr(numQHeads), uintptr(halfRotary),
		uintptr(s))
	return checkKernel(ret, "fused_qk_norm_rope_bf16")
}

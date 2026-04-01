//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// Gather launches the embedding table gather kernel with int64 indices.
// table: [V, D], indices: [N] int64, output: [N, D].
func Gather(table unsafe.Pointer, indices unsafe.Pointer,
	output unsafe.Pointer, N, D, V int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	k := klib()
	if k == nil {
		return fmt.Errorf("gather kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGather,
		uintptr(table), uintptr(indices), uintptr(output),
		uintptr(N), uintptr(D), uintptr(V), uintptr(s))
	return checkKernel(ret, "gather")
}

// GatherI32 launches the embedding table gather kernel with int32 indices.
// table: [V, D], indices: [N] int32, output: [N, D].
func GatherI32(table unsafe.Pointer, indices unsafe.Pointer,
	output unsafe.Pointer, N, D, V int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	k := klib()
	if k == nil {
		return fmt.Errorf("gather_i32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGatherI32,
		uintptr(table), uintptr(indices), uintptr(output),
		uintptr(N), uintptr(D), uintptr(V), uintptr(s))
	return checkKernel(ret, "gather_i32")
}

// GatherQ8F32 launches the Q8_0 embedding gather kernel.
// q8Table: raw Q8_0 bytes [V * (D/32) * 34], indices: [N] int32, output: [N, D] float32.
// Dequantizes only the requested rows on GPU (no full table decode).
func GatherQ8F32(q8Table unsafe.Pointer, indices unsafe.Pointer,
	output unsafe.Pointer, N, D, V int, s unsafe.Pointer) error {
	k := klib()
	if k == nil || k.launchGatherQ8F32 == 0 {
		return fmt.Errorf("gather_q8_f32 kernel: not available")
	}
	ret := cuda.Ccall(k.launchGatherQ8F32,
		uintptr(q8Table), uintptr(indices), uintptr(output),
		uintptr(N), uintptr(D), uintptr(V), uintptr(s))
	return checkKernel(ret, "gather_q8_f32")
}

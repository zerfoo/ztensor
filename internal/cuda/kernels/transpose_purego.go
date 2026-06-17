//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// Transpose2D launches the tiled 2D transpose kernel.
// Input: [rows, cols] -> Output: [cols, rows].
func Transpose2D(input, output unsafe.Pointer, rows, cols int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("transpose_2d kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchTranspose2D,
		uintptr(input), uintptr(output),
		uintptr(rows), uintptr(cols), uintptr(s))
	return checkKernel(ret, "transpose_2d")
}

// TransposeND launches the general N-D transpose kernel.
func TransposeND(input, output unsafe.Pointer,
	inStrides, outStrides, perm []int32,
	ndim, total int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("transpose_nd kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchTransposeND,
		uintptr(input), uintptr(output),
		uintptr(unsafe.Pointer(&inStrides[0])),
		uintptr(unsafe.Pointer(&outStrides[0])),
		uintptr(unsafe.Pointer(&perm[0])),
		uintptr(ndim), uintptr(total), uintptr(s))
	return checkKernel(ret, "transpose_nd")
}

// Transpose2DBF16 launches the tiled 2D transpose kernel for bf16 (16-bit)
// elements. Input: [rows, cols] -> Output: [cols, rows].
func Transpose2DBF16(input, output unsafe.Pointer, rows, cols int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("transpose_2d_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchTranspose2DBF16,
		uintptr(input), uintptr(output),
		uintptr(rows), uintptr(cols), uintptr(s))
	return checkKernel(ret, "transpose_2d_bf16")
}

// TransposeNDBF16 launches the general N-D transpose kernel for bf16 (16-bit)
// elements.
func TransposeNDBF16(input, output unsafe.Pointer,
	inStrides, outStrides, perm []int32,
	ndim, total int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("transpose_nd_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchTransposeNDBF16,
		uintptr(input), uintptr(output),
		uintptr(unsafe.Pointer(&inStrides[0])),
		uintptr(unsafe.Pointer(&outStrides[0])),
		uintptr(unsafe.Pointer(&perm[0])),
		uintptr(ndim), uintptr(total), uintptr(s))
	return checkKernel(ret, "transpose_nd_bf16")
}

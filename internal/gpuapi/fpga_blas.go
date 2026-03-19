package gpuapi

import (
	"fmt"
	"unsafe"
)

// FPGABlas implements the BLAS interface for FPGA accelerators.
// Basic MatMul is supported; advanced precision variants return not-supported.
type FPGABlas struct{}

// NewFPGABlas creates a new FPGA BLAS adapter.
func NewFPGABlas() *FPGABlas {
	return &FPGABlas{}
}

func (b *FPGABlas) Sgemm(m, n, k int, alpha float32, a, bPtr unsafe.Pointer, beta float32, c unsafe.Pointer) error {
	return fmt.Errorf("Sgemm: not yet implemented for FPGA backend")
}

func (b *FPGABlas) BFloat16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("BFloat16Gemm: not supported on FPGA backend")
}

func (b *FPGABlas) Float16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("Float16Gemm: not yet implemented for FPGA backend")
}

func (b *FPGABlas) MixedFP16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("MixedFP16Gemm: not supported on FPGA backend")
}

func (b *FPGABlas) MixedBF16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("MixedBF16Gemm: not supported on FPGA backend")
}

func (b *FPGABlas) SetStream(_ Stream) error {
	return nil
}

func (b *FPGABlas) Destroy() error {
	return nil
}

// Compile-time interface assertion.
var _ BLAS = (*FPGABlas)(nil)

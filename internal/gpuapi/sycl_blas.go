package gpuapi

import (
	"fmt"
	"unsafe"
)

// SYCLBlas implements the BLAS interface for SYCL devices.
// Basic Sgemm is delegated to oneMKL when available; advanced precision
// variants return not-supported until oneMKL bindings are added.
type SYCLBlas struct{}

// NewSYCLBlas creates a new SYCL BLAS adapter.
func NewSYCLBlas() *SYCLBlas {
	return &SYCLBlas{}
}

func (b *SYCLBlas) Sgemm(m, n, k int, alpha float32, a, bPtr unsafe.Pointer, beta float32, c unsafe.Pointer) error {
	return fmt.Errorf("Sgemm: not yet implemented for SYCL backend")
}

func (b *SYCLBlas) BFloat16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("BFloat16Gemm: not supported on SYCL backend")
}

func (b *SYCLBlas) Float16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("Float16Gemm: not yet implemented for SYCL backend")
}

func (b *SYCLBlas) MixedFP16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("MixedFP16Gemm: not supported on SYCL backend")
}

func (b *SYCLBlas) MixedBF16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("MixedBF16Gemm: not supported on SYCL backend")
}

func (b *SYCLBlas) SetStream(_ Stream) error {
	return nil
}

func (b *SYCLBlas) Destroy() error {
	return nil
}

// Compile-time interface assertion.
var _ BLAS = (*SYCLBlas)(nil)

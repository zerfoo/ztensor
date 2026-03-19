package gpuapi

import (
	"fmt"
	"unsafe"
)

// MetalBlas implements the BLAS interface using Metal Performance Shaders.
type MetalBlas struct{}

// NewMetalBlas creates a new MPS BLAS adapter.
func NewMetalBlas() *MetalBlas {
	return &MetalBlas{}
}

func (b *MetalBlas) Sgemm(m, n, k int, alpha float32, a, bPtr unsafe.Pointer, beta float32, c unsafe.Pointer) error {
	return fmt.Errorf("Sgemm: not yet implemented for Metal backend")
}

func (b *MetalBlas) BFloat16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("BFloat16Gemm: not supported on Metal backend")
}

func (b *MetalBlas) Float16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("Float16Gemm: not yet implemented for Metal backend")
}

func (b *MetalBlas) MixedFP16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("MixedFP16Gemm: not supported on Metal backend")
}

func (b *MetalBlas) MixedBF16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("MixedBF16Gemm: not supported on Metal backend")
}

func (b *MetalBlas) SetStream(_ Stream) error {
	return nil
}

func (b *MetalBlas) Destroy() error {
	return nil
}

// Compile-time interface assertion.
var _ BLAS = (*MetalBlas)(nil)

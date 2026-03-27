//go:build !cuda

package kernels

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

func TestIQDequantIQ4NLNoCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("test targets non-CUDA environment; CUDA is available")
	}

	err := DequantIQ4NL(
		unsafe.Pointer(uintptr(0)),
		unsafe.Pointer(uintptr(0)),
		unsafe.Pointer(uintptr(0)),
		unsafe.Pointer(uintptr(0)),
		256,
		unsafe.Pointer(uintptr(0)),
	)
	if err == nil {
		t.Fatal("expected error when CUDA is not available, got nil")
	}
}

func TestIQDequantIQ3SNoCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("test targets non-CUDA environment; CUDA is available")
	}

	err := DequantIQ3S(
		unsafe.Pointer(uintptr(0)),
		unsafe.Pointer(uintptr(0)),
		unsafe.Pointer(uintptr(0)),
		256,
		unsafe.Pointer(uintptr(0)),
	)
	if err == nil {
		t.Fatal("expected error when CUDA is not available, got nil")
	}
}

func TestIQDequantIQ2XXSNoCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("test targets non-CUDA environment; CUDA is available")
	}

	err := DequantIQ2XXS(
		unsafe.Pointer(uintptr(0)),
		unsafe.Pointer(uintptr(0)),
		unsafe.Pointer(uintptr(0)),
		256,
		unsafe.Pointer(uintptr(0)),
	)
	if err == nil {
		t.Fatal("expected error when CUDA is not available, got nil")
	}
}

func TestIQDequantSupportedNoCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("test targets non-CUDA environment; CUDA is available")
	}

	if IsIQDequantSupported() {
		t.Fatal("expected IsIQDequantSupported() == false without CUDA")
	}
}

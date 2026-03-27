//go:build !cuda

package kernels

import (
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
)

func TestKVDequantQ4NoCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("test targets non-CUDA environment; CUDA is available")
	}

	err := KVDequantQ4(nil, nil, nil, nil, 16, 128, 128, nil)
	if err == nil {
		t.Fatal("expected error when CUDA is not available, got nil")
	}
}

func TestKVDequantQ3NoCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("test targets non-CUDA environment; CUDA is available")
	}

	err := KVDequantQ3(nil, nil, nil, nil, 16, 128, 128, nil)
	if err == nil {
		t.Fatal("expected error when CUDA is not available, got nil")
	}
}

func TestIsKVDequantQ4SupportedNoCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("test targets non-CUDA environment; CUDA is available")
	}

	if IsKVDequantQ4Supported() {
		t.Fatal("expected IsKVDequantQ4Supported() == false without CUDA")
	}
}

func TestIsKVDequantQ3SupportedNoCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("test targets non-CUDA environment; CUDA is available")
	}

	if IsKVDequantQ3Supported() {
		t.Fatal("expected IsKVDequantQ3Supported() == false without CUDA")
	}
}

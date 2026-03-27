//go:build !cuda

package kernels

import (
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
)

func TestNSAAttentionForwardNoCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("test requires CUDA to be unavailable")
	}
	err := NSAAttentionForward(
		nil, nil, nil, nil,
		nil, nil, nil,
		1, 1, 64, 128,
		4, 4,
		16, 4, 8, 128,
		nil,
	)
	if err == nil {
		t.Fatal("expected error when CUDA is unavailable, got nil")
	}
}

func TestIsNSAAttentionSupportedNoCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("test requires CUDA to be unavailable")
	}
	if IsNSAAttentionSupported() {
		t.Fatal("expected IsNSAAttentionSupported() == false without CUDA")
	}
}

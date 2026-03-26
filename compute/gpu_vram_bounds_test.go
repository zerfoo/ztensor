package compute

import (
	"context"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/log"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestMatMul_VRAMBoundsCheck verifies that MatMul returns a clear error
// (not a segfault) when the output allocation would exceed available VRAM.
func TestMatMul_VRAMBoundsCheck(t *testing.T) {
	pool := newFakeMemPool()
	eng := &GPUEngine[float32]{
		cpu:           NewCPUEngine[float32](numeric.Float32Ops{}),
		runtime:       fakeRuntime{},
		pool:          pool,
		stream:        fakeStream{},
		logger:        log.Nop(),
		deviceID:      0,
		dtype:         DTypeF32,
		maxAllocBytes: 1024, // 1 KB limit for testing
	}
	ctx := context.Background()

	t.Run("small_matmul_succeeds", func(t *testing.T) {
		// 2x3 * 3x2 = 2x2 output = 16 bytes (4 floats * 4 bytes) — within limit.
		a, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("tensor.New A: %v", err)
		}
		b, err := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("tensor.New B: %v", err)
		}
		result, err := eng.MatMul(ctx, a, b)
		if err != nil {
			t.Fatalf("expected small MatMul to succeed, got: %v", err)
		}
		shape := result.Shape()
		if len(shape) != 2 || shape[0] != 2 || shape[1] != 2 {
			t.Errorf("expected shape [2,2], got %v", shape)
		}
	})

	t.Run("large_matmul_returns_error", func(t *testing.T) {
		// 128256x4096 * 4096x4096 output would be 128256*4096*4 = ~2 GB.
		// With 1 KB limit, this must fail with a clear error.
		a, err := tensor.New[float32]([]int{128256, 4096}, nil)
		if err != nil {
			t.Fatalf("tensor.New A: %v", err)
		}
		b, err := tensor.New[float32]([]int{4096, 4096}, nil)
		if err != nil {
			t.Fatalf("tensor.New B: %v", err)
		}
		_, err = eng.MatMul(ctx, a, b)
		if err == nil {
			t.Fatal("expected error for oversized MatMul, got nil")
		}
		if !strings.Contains(err.Error(), "exceeds available VRAM") {
			t.Errorf("error should mention VRAM limit, got: %v", err)
		}
		if !strings.Contains(err.Error(), "MatMul") {
			t.Errorf("error should mention MatMul, got: %v", err)
		}
	})

	t.Run("error_includes_allocation_size", func(t *testing.T) {
		// 32x32 * 32x32 = 32*32*4 = 4096 bytes — exceeds 1 KB limit.
		a, err := tensor.New[float32]([]int{32, 32}, nil)
		if err != nil {
			t.Fatalf("tensor.New A: %v", err)
		}
		b, err := tensor.New[float32]([]int{32, 32}, nil)
		if err != nil {
			t.Fatalf("tensor.New B: %v", err)
		}
		_, err = eng.MatMul(ctx, a, b)
		if err == nil {
			t.Fatal("expected error for MatMul exceeding 1KB limit")
		}
		// Output is 32*32*4 = 4096 bytes.
		if !strings.Contains(err.Error(), "4096 bytes") {
			t.Errorf("error should include allocation size '4096 bytes', got: %v", err)
		}
	})
}

// TestMatMul_VRAMBoundsCheck_DefaultLimit verifies that the default limit
// (4 GB) allows normal-sized MatMul operations.
func TestMatMul_VRAMBoundsCheck_DefaultLimit(t *testing.T) {
	pool := newFakeMemPool()
	eng := &GPUEngine[float32]{
		cpu:           NewCPUEngine[float32](numeric.Float32Ops{}),
		runtime:       fakeRuntime{},
		pool:          pool,
		stream:        fakeStream{},
		logger:        log.Nop(),
		deviceID:      0,
		dtype:         DTypeF32,
		maxAllocBytes: DefaultMaxAllocBytes,
	}
	ctx := context.Background()

	// 512x512 * 512x512 = 512*512*4 = 1 MB — well within 4 GB.
	a, err := tensor.New[float32]([]int{512, 512}, nil)
	if err != nil {
		t.Fatalf("tensor.New A: %v", err)
	}
	b, err := tensor.New[float32]([]int{512, 512}, nil)
	if err != nil {
		t.Fatalf("tensor.New B: %v", err)
	}
	result, err := eng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("expected MatMul with default limit to succeed, got: %v", err)
	}
	shape := result.Shape()
	if len(shape) != 2 || shape[0] != 512 || shape[1] != 512 {
		t.Errorf("expected shape [512,512], got %v", shape)
	}
}

// TestSetMaxAllocBytes verifies the setter and getter for maxAllocBytes.
func TestSetMaxAllocBytes(t *testing.T) {
	pool := newFakeMemPool()
	eng := &GPUEngine[float32]{
		cpu:           NewCPUEngine[float32](numeric.Float32Ops{}),
		runtime:       fakeRuntime{},
		pool:          pool,
		stream:        fakeStream{},
		logger:        log.Nop(),
		deviceID:      0,
		dtype:         DTypeF32,
		maxAllocBytes: DefaultMaxAllocBytes,
	}

	if eng.MaxAllocBytes() != DefaultMaxAllocBytes {
		t.Errorf("expected default %d, got %d", DefaultMaxAllocBytes, eng.MaxAllocBytes())
	}

	eng.SetMaxAllocBytes(8 * 1024 * 1024 * 1024) // 8 GB
	if eng.MaxAllocBytes() != 8*1024*1024*1024 {
		t.Errorf("expected 8 GB, got %d", eng.MaxAllocBytes())
	}

	// Setting to 0 or negative should reset to default.
	eng.SetMaxAllocBytes(0)
	if eng.MaxAllocBytes() != DefaultMaxAllocBytes {
		t.Errorf("expected default after setting 0, got %d", eng.MaxAllocBytes())
	}

	eng.SetMaxAllocBytes(-1)
	if eng.MaxAllocBytes() != DefaultMaxAllocBytes {
		t.Errorf("expected default after setting -1, got %d", eng.MaxAllocBytes())
	}
}

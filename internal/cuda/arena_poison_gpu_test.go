package cuda

import (
	"math"
	"testing"
	"unsafe"
)

// TestArenaPoison_GPU_DefaultHostStagedFill is the on-device counterpart of
// TestArenaPoison_CachedBufferAfterReset. It exercises the DEFAULT fill path
// (host-staged synchronous Memcpy) against a real CUDA arena: alloc on
// device, write clean values, Reset, copy back, and assert the NaN sentinel.
//
// Skips when CUDA is unavailable (CI is CPU-only); run on the GB10 via a
// Spark pod. The kernel-based fill registered by internal/gpuapi has its own
// GPU test there.
func TestArenaPoison_GPU_DefaultHostStagedFill(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}
	enableArenaPoisonForTest(t, true)
	// NOTE: deliberately no swapHostPoisonFillForTest -- the default
	// arenaPoisonFillHostStaged must poison real device memory.

	fallback := NewMemPool()
	arena, err := NewArenaPool(0, 1<<20, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	const numFloats = 1024
	byteSize := numFloats * 4
	ptr, err := arena.Alloc(0, byteSize)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}

	// Forward: write clean values to the device buffer; the fake node caches ptr.
	clean := make([]float32, numFloats)
	for i := range clean {
		clean[i] = 1.5
	}
	if err := Memcpy(ptr, unsafe.Pointer(&clean[0]), byteSize, MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}

	// Step boundary.
	arena.Reset()

	// Backward: the stale cached pointer must now read poison.
	got := make([]float32, numFloats)
	if err := Memcpy(unsafe.Pointer(&got[0]), ptr, byteSize, MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}
	for i, v := range got {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("device read [%d] after Reset = %v (bits 0x%08X), want NaN",
				i, v, math.Float32bits(v))
		}
	}
	if bits := math.Float32bits(got[0]); bits != ArenaPoisonWord {
		t.Fatalf("poison bits = 0x%08X, want 0x%08X", bits, ArenaPoisonWord)
	}
}

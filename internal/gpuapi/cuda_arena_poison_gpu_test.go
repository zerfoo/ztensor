package gpuapi

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// TestArenaPoisonKernelFill_GPU verifies the on-device fill path registered
// by NewCUDAArenaPool: the elementwise fill kernel must write the exact
// cuda.ArenaPoisonWord bit pattern (NaN payload preserved end-to-end through
// the float32 round-trip and the kernel store).
//
// Skips when CUDA is unavailable (CI is CPU-only); run on the GB10 via a
// Spark pod.
func TestArenaPoisonKernelFill_GPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	const numFloats = 1024
	byteSize := numFloats * 4
	ptr, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc: %v", err)
	}
	defer func() { _ = cuda.Free(ptr) }()

	// Pre-fill with a clean value so the poison overwrite is observable.
	clean := make([]float32, numFloats)
	for i := range clean {
		clean[i] = 1.5
	}
	if err := cuda.Memcpy(ptr, unsafe.Pointer(&clean[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}

	if err := arenaPoisonKernelFill(ptr, byteSize); err != nil {
		t.Skipf("fill kernel not available: %v", err)
	}

	got := make([]float32, numFloats)
	// Synchronous cudaMemcpy orders after the default-stream fill kernel.
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), ptr, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}
	for i, v := range got {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("got[%d] = %v (bits 0x%08X), want NaN", i, v, math.Float32bits(v))
		}
		if bits := math.Float32bits(v); bits != cuda.ArenaPoisonWord {
			t.Fatalf("got[%d] bits = 0x%08X, want 0x%08X (payload must survive bit-exact)",
				i, bits, cuda.ArenaPoisonWord)
		}
	}
}

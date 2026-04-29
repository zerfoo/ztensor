package compute

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestGPUEngine_UploadWeights_BulkPath verifies that a many-tensor upload
// (above the bulk threshold) collapses into a single allocation, that data
// is preserved across all uploaded tensors, and that the engine retains
// ownership of the underlying device buffer (zerfoo/ztensor#103).
func TestGPUEngine_UploadWeights_BulkPath(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	// N must exceed bulkUploadF32MinTensors (64) to exercise the bulk path.
	const N = 128
	const elemsPer = 17

	tensors := make([]*tensor.TensorNumeric[float32], N)
	for i := range N {
		data := make([]float32, elemsPer)
		for j := range elemsPer {
			data[j] = float32(i*1000 + j)
		}
		tt, _ := tensor.New[float32]([]int{elemsPer}, data)
		tensors[i] = tt
	}

	if got := len(gpuEng.bulkUploadBuffers); got != 0 {
		t.Fatalf("bulkUploadBuffers before upload = %d, want 0", got)
	}

	if err := gpuEng.UploadWeights(tensors); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	// One bulk buffer should now back all N tensors.
	if got := len(gpuEng.bulkUploadBuffers); got != 1 {
		t.Fatalf("bulkUploadBuffers after upload = %d, want 1 (bulk path collapses to one allocation)", got)
	}

	// Every tensor must now be GPU-resident.
	for i, tt := range tensors {
		if _, ok := tt.GetStorage().(*tensor.GPUStorage[float32]); !ok {
			t.Fatalf("tensor[%d] storage = %T, want *GPUStorage[float32]", i, tt.GetStorage())
		}
	}

	// Round-trip a sample to verify data preservation across the bulk copy.
	for _, i := range []int{0, 1, N / 2, N - 1} {
		got := tensors[i].Data()
		want := float32(i*1000 + (elemsPer - 1))
		if math.Abs(float64(got[elemsPer-1]-want)) > 1e-6 {
			t.Errorf("tensor[%d][%d] = %f, want %f", i, elemsPer-1, got[elemsPer-1], want)
		}
	}
}

// TestGPUEngine_UploadWeights_BelowBulkThreshold verifies that small inputs
// stay on the per-tensor path and the bulk allocation slice remains empty.
func TestGPUEngine_UploadWeights_BelowBulkThreshold(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	// Below bulkUploadF32MinTensors=64.
	const N = 8
	tensors := make([]*tensor.TensorNumeric[float32], N)
	for i := range N {
		data := make([]float32, 4)
		for j := range 4 {
			data[j] = float32(i + j)
		}
		tt, _ := tensor.New[float32]([]int{4}, data)
		tensors[i] = tt
	}

	if err := gpuEng.UploadWeights(tensors); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	if got := len(gpuEng.bulkUploadBuffers); got != 0 {
		t.Errorf("bulkUploadBuffers = %d, want 0 (below threshold should use per-tensor path)", got)
	}
	for i, tt := range tensors {
		if _, ok := tt.GetStorage().(*tensor.GPUStorage[float32]); !ok {
			t.Fatalf("tensor[%d] storage = %T, want *GPUStorage[float32]", i, tt.GetStorage())
		}
	}
}

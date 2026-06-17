package compute

import (
	"math"
	"testing"

	"github.com/zerfoo/float16"
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

// TestGPUEngine_UploadWeightsT_BF16 verifies the generic (T-typed) bulk upload:
// a GPUEngine[float16.BFloat16] can make its host-backed bf16 weight tensors
// device-resident via UploadWeightsT, so they no longer stay host-backed and
// pay a per-op H2D firehose. Asserts the tensors become *GPUStorage[BFloat16]
// (2-byte device elements, sized by unsafe.Sizeof, not f32Size) and that values
// round-trip across the bulk copy. CUDA-gated; skips without a GPU.
// GPU-UNVERIFIED until run on the GB10 verify pod.
func TestGPUEngine_UploadWeightsT_BF16(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	gpuEng, err := NewGPUEngine[float16.BFloat16](numeric.BFloat16Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine[BFloat16]: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	// N must exceed bulkUploadF32MinTensors (64) to exercise the bulk path.
	const N = 128
	const elemsPer = 17

	tensors := make([]*tensor.TensorNumeric[float16.BFloat16], N)
	wantVals := make([][]float32, N)
	for i := range N {
		data := make([]float16.BFloat16, elemsPer)
		wantVals[i] = make([]float32, elemsPer)
		for j := range elemsPer {
			// values exactly representable in bf16 so the round-trip is exact
			f := float16.BFloat16FromFloat32(float32(i) + float32(j)/64.0)
			data[j] = f
			wantVals[i][j] = f.ToFloat32()
		}
		tt, errNew := tensor.New[float16.BFloat16]([]int{elemsPer}, data)
		if errNew != nil {
			t.Fatalf("tensor.New[BFloat16]: %v", errNew)
		}
		tensors[i] = tt
	}

	if got := len(gpuEng.bulkUploadBuffers); got != 0 {
		t.Fatalf("bulkUploadBuffers before upload = %d, want 0", got)
	}

	if err := gpuEng.UploadWeightsT(tensors); err != nil {
		t.Fatalf("UploadWeightsT: %v", err)
	}

	// Every tensor must now be device-resident as *GPUStorage[BFloat16].
	for i, tt := range tensors {
		if _, ok := tt.GetStorage().(*tensor.GPUStorage[float16.BFloat16]); !ok {
			t.Fatalf("tensor[%d] storage = %T, want *GPUStorage[float16.BFloat16]", i, tt.GetStorage())
		}
	}

	// Round-trip a sample to verify the bulk copy preserved bf16 bytes.
	for _, i := range []int{0, 1, N / 2, N - 1} {
		got := tensors[i].Data()
		for j := range elemsPer {
			if got[j].ToFloat32() != wantVals[i][j] {
				t.Errorf("tensor[%d][%d] = %g, want %g", i, j, got[j].ToFloat32(), wantVals[i][j])
			}
		}
	}

	// A second UploadWeightsT must be a no-op: the tensors are already
	// *GPUStorage[BFloat16] and must be skipped (no new bulk buffers).
	before := len(gpuEng.bulkUploadBuffers)
	if err := gpuEng.UploadWeightsT(tensors); err != nil {
		t.Fatalf("UploadWeightsT (second call): %v", err)
	}
	if after := len(gpuEng.bulkUploadBuffers); after != before {
		t.Errorf("second UploadWeightsT allocated %d new bulk buffers, want 0 (already-resident skip)", after-before)
	}
}

// TestGPUEngine_UploadWeights_MultiChunk exercises the bounded-chunk upload
// path on real hardware (zerfoo/ztensor#106). It uploads a payload large enough
// to span several bulkUploadF32MaxChunkBytes (64 MiB) chunks, proving that (a) a
// real 64 MiB cudaMalloc + H2D copy does not wedge the GB10 driver, (b) the
// bulk buffer slice holds one pointer per chunk, and (c) tensor data round-trips
// correctly across chunk boundaries. Skips without CUDA.
func TestGPUEngine_UploadWeights_MultiChunk(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	// 256 tensors of 1 MiB each = 256 MiB total. With a 64 MiB byte cap this
	// tiles into 4 chunks (the tensor-count cap of 4096 is not reached), so the
	// upload issues 4 bounded device allocations + copies instead of one 256 MiB
	// allocation that would risk wedging the driver.
	const elemsPer = 256 * 1024 // 1 MiB per tensor
	const N = 256
	const wantChunks = 4

	tensors := make([]*tensor.TensorNumeric[float32], N)
	for i := range N {
		data := make([]float32, elemsPer)
		// Sentinel at both ends of each tensor to catch chunk-boundary offset bugs.
		data[0] = float32(i*1_000_000 + 1)
		data[elemsPer-1] = float32(i*1_000_000 + 2)
		tt, _ := tensor.New[float32]([]int{elemsPer}, data)
		tensors[i] = tt
	}

	if err := gpuEng.UploadWeights(tensors); err != nil {
		t.Fatalf("UploadWeights (multi-chunk): %v", err)
	}

	if got := len(gpuEng.bulkUploadBuffers); got != wantChunks {
		t.Fatalf("bulkUploadBuffers after multi-chunk upload = %d, want %d", got, wantChunks)
	}

	for i, tt := range tensors {
		if _, ok := tt.GetStorage().(*tensor.GPUStorage[float32]); !ok {
			t.Fatalf("tensor[%d] storage = %T, want *GPUStorage[float32]", i, tt.GetStorage())
		}
	}

	// Round-trip the first and last element of tensors at and around each chunk
	// boundary (every 64th tensor) to confirm views point at the right offsets.
	for _, i := range []int{0, 63, 64, 127, 128, 191, 192, N - 1} {
		got := tensors[i].Data()
		wantHead := float32(i*1_000_000 + 1)
		wantTail := float32(i*1_000_000 + 2)
		if math.Abs(float64(got[0]-wantHead)) > 1e-6 {
			t.Errorf("tensor[%d][0] = %f, want %f", i, got[0], wantHead)
		}
		if math.Abs(float64(got[elemsPer-1]-wantTail)) > 1e-6 {
			t.Errorf("tensor[%d][last] = %f, want %f", i, got[elemsPer-1], wantTail)
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

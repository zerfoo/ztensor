package compute

import (
	"errors"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/log"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestPreAllocateWorkspaces_FP8ScratchInitialized verifies that after
// UploadWeights, the FP8 scratchpad is non-nil (eagerly initialized).
func TestPreAllocateWorkspaces_FP8ScratchInitialized(t *testing.T) {
	eng := newPreallocEngine(t)
	if eng.fp8Scratch != nil {
		t.Fatal("precondition: fp8Scratch should be nil before UploadWeights")
	}

	if err := eng.UploadWeights(nil); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	if eng.fp8Scratch == nil {
		t.Fatal("fp8Scratch should be non-nil after UploadWeights")
	}
	if eng.fp8Scratch.scaleOne == nil {
		t.Fatal("fp8Scratch.scaleOne should be non-nil after pre-allocation")
	}
}

// TestPreAllocateWorkspaces_CalledByUploadWeights verifies that
// preAllocateWorkspaces fires at the end of UploadWeights even when
// called with an empty weight list (the pre-allocation is unconditional).
func TestPreAllocateWorkspaces_CalledByUploadWeights(t *testing.T) {
	eng := newPreallocEngine(t)

	if err := eng.UploadWeights([]*tensor.TensorNumeric[float32]{}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	if eng.fp8Scratch == nil {
		t.Fatal("fp8Scratch should be non-nil after UploadWeights")
	}
	if eng.fp8Scratch.scaleOne == nil {
		t.Fatal("fp8Scratch.scaleOne should be non-nil after pre-allocation")
	}
}

// TestPreAllocateWorkspaces_TableDriven exercises workspace pre-allocation
// with varying weight list sizes. Pre-allocation is unconditional, so
// fp8Scratch should be non-nil regardless of weight count.
func TestPreAllocateWorkspaces_TableDriven(t *testing.T) {
	tests := []struct {
		name       string
		numWeights int
	}{
		{name: "no weights", numWeights: 0},
		{name: "one nil entry", numWeights: 1},
		{name: "three nil entries", numWeights: 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newPreallocEngine(t)
			pool := eng.pool.(*fakeMemPool)

			// Pass nil tensor entries -- UploadWeights skips them.
			weights := make([]*tensor.TensorNumeric[float32], tt.numWeights)
			if err := eng.UploadWeights(weights); err != nil {
				t.Fatalf("UploadWeights: %v", err)
			}

			if eng.fp8Scratch == nil {
				t.Error("fp8Scratch should be non-nil after UploadWeights")
			}
			if eng.fp8Scratch.scaleOne == nil {
				t.Error("fp8Scratch.scaleOne should be non-nil")
			}
			// scaleOne alloc is the minimum: 1 pool.Alloc from getFP8Scratch.
			if pool.allocCount < 1 {
				t.Errorf("expected at least 1 alloc from pre-allocation, got %d", pool.allocCount)
			}
		})
	}
}

// TestCaptureAllocCount_ZeroAfterPrealloc verifies that captureAllocCount
// stays at zero when allocWeight is not called during capture. This is the
// expected state for a properly pre-allocated workload.
func TestCaptureAllocCount_ZeroAfterPrealloc(t *testing.T) {
	eng := newPreallocEngine(t)
	if err := eng.UploadWeights(nil); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	if got := eng.CaptureAllocCount(); got != 0 {
		t.Fatalf("CaptureAllocCount after UploadWeights: got %d, want 0", got)
	}
}

// TestCaptureAllocCount_IncrementsOnCaptureTimeAlloc verifies that
// allocWeight increments captureAllocCount when capture is active.
func TestCaptureAllocCount_IncrementsOnCaptureTimeAlloc(t *testing.T) {
	restore := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restore()

	eng := &GPUEngine[float32]{stream: fakePtrStream{}}

	// First attempt — should fail with capture sentinel and increment counter.
	_, err := eng.allocWeight(4096)
	if !errors.Is(err, ErrCaptureIncompatibleAllocation) {
		t.Fatalf("allocWeight: expected ErrCaptureIncompatibleAllocation, got %v", err)
	}

	if got := eng.CaptureAllocCount(); got != 1 {
		t.Fatalf("CaptureAllocCount after 1 attempt: got %d, want 1", got)
	}

	// Second attempt — count should increase.
	_, _ = eng.allocWeight(8192)
	if got := eng.CaptureAllocCount(); got != 2 {
		t.Fatalf("CaptureAllocCount after 2 attempts: got %d, want 2", got)
	}
}

// TestCaptureAllocCount_ResetByEndCapture verifies that EndCapture resets
// the captureAllocCount to zero after logging.
func TestCaptureAllocCount_ResetByEndCapture(t *testing.T) {
	// Arrange: inject a capture-active status for allocWeight, then swap to
	// a non-capture status for EndCapture.
	captureActive := true
	restore := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		if captureActive {
			return cuda.CaptureStatusActive, nil
		}
		return cuda.CaptureStatusNone, nil
	})
	defer restore()

	eng := &GPUEngine[float32]{
		stream: fakePtrStream{},
		logger: log.Nop(),
	}

	// Trigger two allocWeight attempts during capture.
	_, _ = eng.allocWeight(4096)
	_, _ = eng.allocWeight(8192)
	if got := eng.CaptureAllocCount(); got != 2 {
		t.Fatalf("CaptureAllocCount before EndCapture: got %d, want 2", got)
	}

	// EndCapture will fail (no real graph) but should still reset the counter.
	captureActive = false
	oldEnd := streamEndCaptureFn
	streamEndCaptureFn = func(_ *cuda.Stream) (*cuda.Graph, error) {
		return nil, errors.New("synthetic: no graph")
	}
	defer func() { streamEndCaptureFn = oldEnd }()

	_, _ = eng.EndCapture()

	if got := eng.CaptureAllocCount(); got != 0 {
		t.Fatalf("CaptureAllocCount after EndCapture: got %d, want 0", got)
	}
}

// TestPreAllocateWorkspaces_Idempotent verifies that calling
// preAllocateWorkspaces multiple times does not leak or double-allocate.
func TestPreAllocateWorkspaces_Idempotent(t *testing.T) {
	eng := newPreallocEngine(t)
	pool := eng.pool.(*fakeMemPool)

	eng.preAllocateWorkspaces()
	allocsAfterFirst := pool.allocCount

	eng.preAllocateWorkspaces()
	allocsAfterSecond := pool.allocCount

	if allocsAfterSecond != allocsAfterFirst {
		t.Fatalf("second preAllocateWorkspaces caused %d new allocs, want 0",
			allocsAfterSecond-allocsAfterFirst)
	}
}

// newPreallocEngine builds a GPUEngine suitable for testing workspace
// pre-allocation without real CUDA hardware.
func newPreallocEngine(t *testing.T) *GPUEngine[float32] {
	t.Helper()
	pool := newFakeMemPool()
	return &GPUEngine[float32]{
		cpu:           NewCPUEngine[float32](numeric.Float32Ops{}),
		runtime:       fakeRuntime{},
		pool:          pool,
		stream:        fakeStream{},
		logger:        log.Nop(),
		deviceID:      0,
		dtype:         DTypeF32,
		maxAllocBytes: DefaultMaxAllocBytes,
	}
}

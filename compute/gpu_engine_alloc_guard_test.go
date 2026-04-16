package compute

import (
	"errors"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// TestAllocWeight_PropagatesCaptureSentinel confirms the capture guard's
// sentinel flows out of allocWeight unchanged. A caller wrapping the error
// with fmt.Errorf("%w") must still match the sentinel via errors.Is so that
// fallback paths (CaptureSafe, later epics) can catch the exact failure mode.
func TestAllocWeight_PropagatesCaptureSentinel(t *testing.T) {
	restore := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restore()

	e := &GPUEngine[float32]{stream: fakePtrStream{}}
	ptr, err := e.allocWeight(4096)
	if err == nil {
		t.Fatal("allocWeight under active capture: expected error, got nil")
	}
	if !errors.Is(err, ErrCaptureIncompatibleAllocation) {
		t.Fatalf("allocWeight: expected ErrCaptureIncompatibleAllocation, got %v", err)
	}
	if ptr != nil {
		t.Fatalf("allocWeight: expected nil pointer on guard trip, got %p", ptr)
	}
}

// TestAllocWeight_PropagatesProbeError confirms that if the capture probe
// itself fails, allocWeight returns the wrapped probe error — not the
// sentinel, and not a nil error that would let a hang happen silently.
func TestAllocWeight_PropagatesProbeError(t *testing.T) {
	probeErr := errors.New("cudaStreamGetCaptureInfo failed: synthetic")
	restore := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusNone, probeErr
	})
	defer restore()

	e := &GPUEngine[float32]{stream: fakePtrStream{}}
	ptr, err := e.allocWeight(4096)
	if err == nil {
		t.Fatal("allocWeight with failing probe: expected error, got nil")
	}
	if !errors.Is(err, probeErr) {
		t.Fatalf("allocWeight: expected wrapped probe error, got %v", err)
	}
	if errors.Is(err, ErrCaptureIncompatibleAllocation) {
		t.Fatal("allocWeight: probe failure must not be reported as capture sentinel")
	}
	if ptr != nil {
		t.Fatalf("allocWeight: expected nil pointer on probe failure, got %p", ptr)
	}
}

// TestUploadBytes_PropagatesCaptureSentinel mirrors the allocWeight test on
// the upload path. uploadBytes is the second weight-load entry point touched
// during UploadWeights, so both must fail loud under active capture.
func TestUploadBytes_PropagatesCaptureSentinel(t *testing.T) {
	restore := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restore()

	e := &GPUEngine[float32]{stream: fakePtrStream{}}
	src := []byte{0x01, 0x02, 0x03, 0x04}
	err := e.uploadBytes(nil, src)
	if err == nil {
		t.Fatal("uploadBytes under active capture: expected error, got nil")
	}
	if !errors.Is(err, ErrCaptureIncompatibleAllocation) {
		t.Fatalf("uploadBytes: expected ErrCaptureIncompatibleAllocation, got %v", err)
	}
}

// TestUploadBytes_PropagatesProbeError confirms probe failures propagate out
// of uploadBytes the same way they do out of allocWeight.
func TestUploadBytes_PropagatesProbeError(t *testing.T) {
	probeErr := errors.New("cudaStreamGetCaptureInfo failed: synthetic")
	restore := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusNone, probeErr
	})
	defer restore()

	e := &GPUEngine[float32]{stream: fakePtrStream{}}
	src := []byte{0x01, 0x02}
	err := e.uploadBytes(nil, src)
	if err == nil {
		t.Fatal("uploadBytes with failing probe: expected error, got nil")
	}
	if !errors.Is(err, probeErr) {
		t.Fatalf("uploadBytes: expected wrapped probe error, got %v", err)
	}
	if errors.Is(err, ErrCaptureIncompatibleAllocation) {
		t.Fatal("uploadBytes: probe failure must not be reported as capture sentinel")
	}
}

// TestAllocWeight_PassesWhenNotCapturing_NilStream is a negative control: on
// an engine with a nil stream (CPU-only path), allocWeight must NOT be
// short-circuited by the guard. We cannot safely drive it into the real
// runtime Malloc here (no GPU), but we can confirm the guard returns nil and
// the failure, if any, comes from downstream (runtime == nil panic would
// indicate the guard path is wrong).
func TestEnsureNotCapturing_AllowsAllocationWhenStreamAbsent(t *testing.T) {
	e := &GPUEngine[float32]{}
	if err := e.ensureNotCapturing(); err != nil {
		t.Fatalf("ensureNotCapturing with nil stream: got %v, want nil", err)
	}
}

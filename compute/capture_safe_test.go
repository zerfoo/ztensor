package compute

import (
	"errors"
	"sync/atomic"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// TestCaptureSafe_CaptureSucceeds verifies that when WithCapture succeeds,
// CaptureSafe returns the GraphHandle and nil error.
func TestCaptureSafe_CaptureSucceeds(t *testing.T) {
	restore := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restore()

	e := &GPUEngine[float32]{}
	var calls atomic.Int32
	handle, err := CaptureSafe(e, func() error {
		calls.Add(1)
		return nil
	})
	if err != nil {
		t.Fatalf("CaptureSafe: unexpected error: %v", err)
	}
	if handle.ptr == nil {
		t.Fatal("CaptureSafe: expected non-nil GraphHandle on success")
	}
	if calls.Load() != 1 {
		t.Fatalf("CaptureSafe: fn called %d times, want 1", calls.Load())
	}
}

// TestCaptureSafe_IncompatibleAllocation_FallbackSucceeds verifies that when
// capture fails with ErrCaptureIncompatibleAllocation, fn is re-executed
// uncaptured and a zero GraphHandle is returned with nil error.
func TestCaptureSafe_IncompatibleAllocation_FallbackSucceeds(t *testing.T) {
	// Make BeginCapture fail with ErrCaptureIncompatibleAllocation so
	// WithCapture returns the incompatible allocation error without calling fn.
	// This means fn will only be called once — in the uncaptured fallback.
	restore := stubCapturePipeline(
		func(_ *cuda.Stream) error {
			return ErrCaptureIncompatibleAllocation
		},
		happyEnd,
		happyInstantiate,
		happyDestroy,
	)
	defer restore()

	e := &GPUEngine[float32]{}
	var calls atomic.Int32
	handle, err := CaptureSafe(e, func() error {
		calls.Add(1)
		return nil
	})
	if err != nil {
		t.Fatalf("CaptureSafe fallback: unexpected error: %v", err)
	}
	if handle.ptr != nil {
		t.Fatal("CaptureSafe fallback: expected zero GraphHandle")
	}
	if calls.Load() != 1 {
		t.Fatalf("CaptureSafe fallback: fn called %d times, want 1 (uncaptured retry only)", calls.Load())
	}
}

// TestCaptureSafe_IncompatibleAllocation_FnErrorDuringCapture verifies the
// case where fn runs during capture and returns the incompatible allocation
// error, then the uncaptured retry succeeds. fn is called twice total.
func TestCaptureSafe_IncompatibleAllocation_FnErrorDuringCapture(t *testing.T) {
	restore := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restore()

	e := &GPUEngine[float32]{}
	var calls atomic.Int32
	handle, err := CaptureSafe(e, func() error {
		n := calls.Add(1)
		if n == 1 {
			// First call (during capture): return incompatible allocation error.
			return ErrCaptureIncompatibleAllocation
		}
		// Second call (uncaptured retry): succeed.
		return nil
	})
	if err != nil {
		t.Fatalf("CaptureSafe fn-error fallback: unexpected error: %v", err)
	}
	if handle.ptr != nil {
		t.Fatal("CaptureSafe fn-error fallback: expected zero GraphHandle")
	}
	if calls.Load() != 2 {
		t.Fatalf("CaptureSafe fn-error fallback: fn called %d times, want 2", calls.Load())
	}
}

// TestCaptureSafe_IncompatibleAllocation_FallbackFails verifies that when
// capture fails with ErrCaptureIncompatibleAllocation and the uncaptured
// retry also fails, the retry error is returned.
func TestCaptureSafe_IncompatibleAllocation_FallbackFails(t *testing.T) {
	restore := stubCapturePipeline(
		func(_ *cuda.Stream) error {
			return ErrCaptureIncompatibleAllocation
		},
		happyEnd,
		happyInstantiate,
		happyDestroy,
	)
	defer restore()

	retryErr := errors.New("uncaptured retry failed")
	e := &GPUEngine[float32]{}
	var calls atomic.Int32
	handle, err := CaptureSafe(e, func() error {
		calls.Add(1)
		return retryErr
	})
	if !errors.Is(err, retryErr) {
		t.Fatalf("CaptureSafe fallback fail: expected retry error, got %v", err)
	}
	if handle.ptr != nil {
		t.Fatal("CaptureSafe fallback fail: expected zero GraphHandle")
	}
}

// TestCaptureSafe_OtherError verifies that when capture fails with a
// non-ErrCaptureIncompatibleAllocation error, fn is NOT retried and the
// error is returned as-is.
func TestCaptureSafe_OtherError(t *testing.T) {
	otherErr := errors.New("some other capture error")
	restore := stubCapturePipeline(
		func(_ *cuda.Stream) error {
			return otherErr
		},
		happyEnd,
		happyInstantiate,
		happyDestroy,
	)
	defer restore()

	e := &GPUEngine[float32]{}
	var calls atomic.Int32
	handle, err := CaptureSafe(e, func() error {
		calls.Add(1)
		return nil
	})
	if err == nil {
		t.Fatal("CaptureSafe other error: expected error, got nil")
	}
	if handle.ptr != nil {
		t.Fatal("CaptureSafe other error: expected zero GraphHandle")
	}
	// fn should not be called because BeginCapture fails before fn runs.
	if calls.Load() != 0 {
		t.Fatalf("CaptureSafe other error: fn called %d times, want 0", calls.Load())
	}
}

// TestCaptureSafe_FnCalledOnceOnSuccess verifies fn is called exactly once
// when capture succeeds (not called a second time).
func TestCaptureSafe_FnCalledOnceOnSuccess(t *testing.T) {
	restore := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restore()

	e := &GPUEngine[float32]{}
	var calls atomic.Int32
	_, err := CaptureSafe(e, func() error {
		calls.Add(1)
		return nil
	})
	if err != nil {
		t.Fatalf("CaptureSafe: unexpected error: %v", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("CaptureSafe: fn called %d times, want exactly 1", calls.Load())
	}
}

// TestCaptureSafe_FnCalledTwiceOnFallback verifies fn is called exactly twice
// when capture fails with ErrCaptureIncompatibleAllocation from fn: once
// during the capture attempt, then once for the uncaptured fallback.
func TestCaptureSafe_FnCalledTwiceOnFallback(t *testing.T) {
	restore := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restore()

	e := &GPUEngine[float32]{}
	var calls atomic.Int32
	_, err := CaptureSafe(e, func() error {
		n := calls.Add(1)
		if n == 1 {
			return ErrCaptureIncompatibleAllocation
		}
		return nil
	})
	if err != nil {
		t.Fatalf("CaptureSafe: unexpected error: %v", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("CaptureSafe: fn called %d times, want exactly 2", calls.Load())
	}
}

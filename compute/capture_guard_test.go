package compute

import (
	"errors"
	"testing"
)

// TestEnsureNotCapturing_NilStream verifies that ensureNotCapturing returns
// nil on an engine whose stream is nil (CPU-only runtime). This is the
// common path on machines without CUDA.
func TestEnsureNotCapturing_NilStream(t *testing.T) {
	e := &GPUEngine[float32]{}
	if err := e.ensureNotCapturing(); err != nil {
		t.Fatalf("ensureNotCapturing on nil-stream engine: got %v, want nil", err)
	}
}

// TestErrCaptureIncompatibleAllocation_Is verifies that
// ErrCaptureIncompatibleAllocation is a sentinel error usable with
// errors.Is, both directly and when wrapped.
func TestErrCaptureIncompatibleAllocation_Is(t *testing.T) {
	if !errors.Is(ErrCaptureIncompatibleAllocation, ErrCaptureIncompatibleAllocation) {
		t.Fatal("errors.Is should match sentinel against itself")
	}
	wrapped := wrapErr(ErrCaptureIncompatibleAllocation)
	if !errors.Is(wrapped, ErrCaptureIncompatibleAllocation) {
		t.Fatal("errors.Is should see sentinel through a wrapper")
	}
}

// wrapErr emulates a caller that wraps the sentinel error with %w.
// Kept local to the test to avoid leaking helpers into the package API.
func wrapErr(err error) error {
	return &wrappedErr{inner: err}
}

type wrappedErr struct{ inner error }

func (w *wrappedErr) Error() string { return "wrapped: " + w.inner.Error() }
func (w *wrappedErr) Unwrap() error { return w.inner }

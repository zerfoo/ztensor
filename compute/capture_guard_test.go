package compute

import (
	"errors"
	"fmt"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/internal/gpuapi"
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

// TestEnsureNotCapturing_NilPtr verifies that ensureNotCapturing returns nil
// when the engine has a stream whose Ptr() is nil. This can happen when a
// stream object is present but the underlying vendor handle was never
// assigned (CPU-shim runtimes).
func TestEnsureNotCapturing_NilPtr(t *testing.T) {
	e := &GPUEngine[float32]{stream: nilPtrStream{}}
	if err := e.ensureNotCapturing(); err != nil {
		t.Fatalf("ensureNotCapturing on nil-ptr stream: got %v, want nil", err)
	}
}

// TestEnsureNotCapturing_ProbeStatuses is a table-driven test that walks
// every cudaStreamCaptureStatus value through ensureNotCapturing and asserts
// the mapping to the guard's outcome:
//   - None          -> nil (allocation allowed)
//   - Active        -> ErrCaptureIncompatibleAllocation
//   - Invalidated   -> nil (guard only blocks Active; fallback logic handles Invalidated)
func TestEnsureNotCapturing_ProbeStatuses(t *testing.T) {
	tests := []struct {
		name   string
		status cuda.CaptureStatus
		want   error
	}{
		{name: "None allows allocation", status: cuda.CaptureStatusNone, want: nil},
		{name: "Active blocks allocation", status: cuda.CaptureStatusActive, want: ErrCaptureIncompatibleAllocation},
		{name: "Invalidated does not trip the active guard", status: cuda.CaptureStatusInvalidated, want: nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			restore := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
				return tc.status, nil
			})
			defer restore()

			e := &GPUEngine[float32]{stream: fakePtrStream{}}
			got := e.ensureNotCapturing()
			if !errors.Is(got, tc.want) && got != tc.want {
				t.Fatalf("ensureNotCapturing(status=%v): got %v, want %v", tc.status, got, tc.want)
			}
		})
	}
}

// TestEnsureNotCapturing_ProbeError verifies that when cudaStreamGetCaptureInfo
// itself fails, ensureNotCapturing returns that error (wrapped for context) and
// does NOT silently treat the stream as safe. Probe failure must propagate so
// callers fail loud instead of racing a hang on GB10.
func TestEnsureNotCapturing_ProbeError(t *testing.T) {
	probeErr := errors.New("cudaStreamGetCaptureInfo failed: synthetic")
	restore := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusNone, probeErr
	})
	defer restore()

	e := &GPUEngine[float32]{stream: fakePtrStream{}}
	err := e.ensureNotCapturing()
	if err == nil {
		t.Fatal("ensureNotCapturing: expected error from failing probe, got nil")
	}
	if !errors.Is(err, probeErr) {
		t.Fatalf("ensureNotCapturing: expected error to wrap probe error, got %v", err)
	}
	if errors.Is(err, ErrCaptureIncompatibleAllocation) {
		t.Fatal("ensureNotCapturing: probe error must not be surfaced as ErrCaptureIncompatibleAllocation")
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

// TestErrCaptureIncompatibleAllocation_FmtErrorfWrap verifies that the sentinel
// survives fmt.Errorf("...: %w", ...) wrapping — the idiom callers in
// allocWeight / uploadBytes use indirectly via ensureNotCapturing and that
// downstream callers use when adding their own context.
func TestErrCaptureIncompatibleAllocation_FmtErrorfWrap(t *testing.T) {
	wrapped := fmt.Errorf("upload layer %d: %w", 7, ErrCaptureIncompatibleAllocation)
	if !errors.Is(wrapped, ErrCaptureIncompatibleAllocation) {
		t.Fatalf("errors.Is through fmt.Errorf wrap: got false, want true (err=%v)", wrapped)
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

// swapCaptureStatusFn replaces the package-level captureStatusFn for a test
// and returns a restore closure. Callers defer restore() to keep tests hermetic.
func swapCaptureStatusFn(fn func(*cuda.Stream) (cuda.CaptureStatus, error)) func() {
	prev := captureStatusFn
	captureStatusFn = fn
	return func() { captureStatusFn = prev }
}

// fakeStreamSentinel backs fakePtrStream.Ptr() with a stable address so that
// escape-analysis does not re-allocate per call and returned pointers remain
// valid for the lifetime of the test binary. The probe is stubbed, so the
// handle is never dereferenced.
var fakeStreamSentinel byte

// fakePtrStream satisfies gpuapi.Stream and returns a non-nil Ptr so that
// ensureNotCapturing proceeds past the early-return guards and exercises the
// probe path. Synchronize / Destroy are never called by the guard.
type fakePtrStream struct{}

func (fakePtrStream) Synchronize() error  { return nil }
func (fakePtrStream) Destroy() error      { return nil }
func (fakePtrStream) Ptr() unsafe.Pointer { return unsafe.Pointer(&fakeStreamSentinel) }

// nilPtrStream satisfies gpuapi.Stream but returns a nil Ptr. Used to cover
// the "stream present but unbacked" branch of ensureNotCapturing.
type nilPtrStream struct{}

func (nilPtrStream) Synchronize() error  { return nil }
func (nilPtrStream) Destroy() error      { return nil }
func (nilPtrStream) Ptr() unsafe.Pointer { return nil }

// Compile-time assertions that the fakes satisfy gpuapi.Stream.
var (
	_ gpuapi.Stream = fakePtrStream{}
	_ gpuapi.Stream = nilPtrStream{}
)

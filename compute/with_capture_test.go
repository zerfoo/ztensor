package compute

import (
	"errors"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// stubCapturePipeline replaces the package-level capture indirection functions
// with the provided stubs and returns a restore closure. Callers must defer
// restore() to keep tests hermetic.
func stubCapturePipeline(
	begin func(*cuda.Stream) error,
	end func(*cuda.Stream) (*cuda.Graph, error),
	instantiate func(*cuda.Graph) (*cuda.GraphExec, error),
	destroy func(*cuda.Graph) error,
) func() {
	prevBegin := streamBeginCaptureFn
	prevEnd := streamEndCaptureFn
	prevInstantiate := graphInstantiateFn
	prevDestroy := graphDestroyFn

	streamBeginCaptureFn = begin
	streamEndCaptureFn = end
	graphInstantiateFn = instantiate
	graphDestroyFn = destroy

	return func() {
		streamBeginCaptureFn = prevBegin
		streamEndCaptureFn = prevEnd
		graphInstantiateFn = prevInstantiate
		graphDestroyFn = prevDestroy
	}
}

// happyBegin is a stub that always succeeds.
func happyBegin(_ *cuda.Stream) error { return nil }

// happyEnd returns a non-nil Graph stub so GraphInstantiate receives input.
func happyEnd(_ *cuda.Stream) (*cuda.Graph, error) { return &cuda.Graph{}, nil }

// happyInstantiate returns a non-nil GraphExec so the GraphHandle is valid.
func happyInstantiate(_ *cuda.Graph) (*cuda.GraphExec, error) { return &cuda.GraphExec{}, nil }

// happyDestroy always succeeds.
func happyDestroy(_ *cuda.Graph) error { return nil }

// TestWithCapture_NilStream_Succeeds verifies that WithCapture on an engine
// with no stream (CPU-only) successfully calls fn and returns a handle.
// BeginCapture/EndCapture are stubbed to succeed.
func TestWithCapture_NilStream_Succeeds(t *testing.T) {
	restore := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restore()

	e := &GPUEngine[float32]{}
	called := false
	handle, err := e.WithCapture(func() error {
		called = true
		return nil
	})
	if err != nil {
		t.Fatalf("WithCapture: unexpected error: %v", err)
	}
	if !called {
		t.Fatal("WithCapture: fn was not called")
	}
	if handle.ptr == nil {
		t.Fatal("WithCapture: expected non-nil graph handle")
	}
}

// TestWithCapture_PropagatesFnError verifies that when fn returns an error,
// WithCapture returns that error and EndCapture is still called. The returned
// GraphHandle should be zero.
func TestWithCapture_PropagatesFnError(t *testing.T) {
	endCalled := false
	restore := stubCapturePipeline(
		happyBegin,
		func(_ *cuda.Stream) (*cuda.Graph, error) {
			endCalled = true
			return &cuda.Graph{}, nil
		},
		happyInstantiate,
		happyDestroy,
	)
	defer restore()

	fnErr := errors.New("fn failed")
	e := &GPUEngine[float32]{}
	handle, err := e.WithCapture(func() error {
		return fnErr
	})
	if !errors.Is(err, fnErr) {
		t.Fatalf("WithCapture: expected fn error, got %v", err)
	}
	if !endCalled {
		t.Fatal("WithCapture: EndCapture was not called when fn errored")
	}
	if handle.ptr != nil {
		t.Fatal("WithCapture: expected zero GraphHandle on fn error")
	}
}

// TestWithCapture_PropagatesBeginCaptureError verifies that when BeginCapture
// fails, fn is never called and the error is returned.
func TestWithCapture_PropagatesBeginCaptureError(t *testing.T) {
	beginErr := errors.New("begin capture failed")
	restore := stubCapturePipeline(
		func(_ *cuda.Stream) error { return beginErr },
		happyEnd,
		happyInstantiate,
		happyDestroy,
	)
	defer restore()

	fnCalled := false
	e := &GPUEngine[float32]{}
	handle, err := e.WithCapture(func() error {
		fnCalled = true
		return nil
	})
	if err == nil {
		t.Fatal("WithCapture: expected error from failing BeginCapture, got nil")
	}
	if !errors.Is(err, beginErr) {
		t.Fatalf("WithCapture: expected wrapped begin error, got %v", err)
	}
	if fnCalled {
		t.Fatal("WithCapture: fn was called despite BeginCapture failure")
	}
	if handle.ptr != nil {
		t.Fatal("WithCapture: expected zero GraphHandle on BeginCapture error")
	}
}

// TestWithCapture_PropagatesEndCaptureError verifies that when EndCapture
// fails (and fn succeeds), the EndCapture error is returned.
func TestWithCapture_PropagatesEndCaptureError(t *testing.T) {
	endErr := errors.New("end capture failed")
	restore := stubCapturePipeline(
		happyBegin,
		func(_ *cuda.Stream) (*cuda.Graph, error) { return nil, endErr },
		happyInstantiate,
		happyDestroy,
	)
	defer restore()

	e := &GPUEngine[float32]{}
	handle, err := e.WithCapture(func() error {
		return nil
	})
	if err == nil {
		t.Fatal("WithCapture: expected error from failing EndCapture, got nil")
	}
	if !errors.Is(err, endErr) {
		t.Fatalf("WithCapture: expected wrapped end error, got %v", err)
	}
	if handle.ptr != nil {
		t.Fatal("WithCapture: expected zero GraphHandle on EndCapture error")
	}
}

// TestWithCapture_FnErrorTakesPrecedenceOverEndError verifies that when both
// fn and EndCapture return errors, the fn error is returned. This ensures
// callers see the root cause rather than a secondary cleanup failure.
func TestWithCapture_FnErrorTakesPrecedenceOverEndError(t *testing.T) {
	fnErr := errors.New("fn failed")
	endErr := errors.New("end capture failed")
	restore := stubCapturePipeline(
		happyBegin,
		func(_ *cuda.Stream) (*cuda.Graph, error) { return nil, endErr },
		happyInstantiate,
		happyDestroy,
	)
	defer restore()

	e := &GPUEngine[float32]{}
	_, err := e.WithCapture(func() error {
		return fnErr
	})
	if !errors.Is(err, fnErr) {
		t.Fatalf("WithCapture: expected fn error to take precedence, got %v", err)
	}
	if errors.Is(err, endErr) {
		t.Fatal("WithCapture: end error should not leak through when fn error exists")
	}
}

// TestWithCapture_EndCalledEvenWhenFnPanics is not tested because WithCapture
// uses a plain call (not defer) for EndCapture — callers that need panic safety
// should wrap fn themselves. This comment documents the intentional design choice.

// TestWithCapture_ReturnsValidHandle verifies that the returned GraphHandle
// contains a non-nil ptr when both fn and capture succeed.
func TestWithCapture_ReturnsValidHandle(t *testing.T) {
	restore := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restore()

	e := &GPUEngine[float32]{}
	handle, err := e.WithCapture(func() error { return nil })
	if err != nil {
		t.Fatalf("WithCapture: unexpected error: %v", err)
	}
	if handle.ptr == nil {
		t.Fatal("WithCapture: expected non-nil ptr in GraphHandle")
	}
	// Verify the handle contains a *cuda.GraphExec.
	if _, ok := handle.ptr.(*cuda.GraphExec); !ok {
		t.Fatalf("WithCapture: handle.ptr type = %T, want *cuda.GraphExec", handle.ptr)
	}
}

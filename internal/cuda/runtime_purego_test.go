package cuda

import "testing"

func TestStreamCaptureStatus_NoRuntime(t *testing.T) {
	// With or without CUDA, a stream that is not capturing must report None.
	// When CUDA is unavailable, the binding returns None without error.
	// When CUDA is available, a freshly created stream is not capturing.
	var s *Stream
	if Available() {
		var err error
		s, err = CreateStream()
		if err != nil {
			t.Fatalf("CreateStream failed: %v", err)
		}
		defer func() {
			if destroyErr := s.Destroy(); destroyErr != nil {
				t.Errorf("Stream.Destroy failed: %v", destroyErr)
			}
		}()
	} else {
		s = &Stream{}
	}

	status, err := StreamCaptureStatus(s)
	if err != nil {
		t.Fatalf("StreamCaptureStatus returned error: %v", err)
	}
	if status != CaptureStatusNone {
		t.Fatalf("expected CaptureStatusNone on a non-capturing stream, got %v", status)
	}
}

// TestStreamFromPtr_NilHandle verifies StreamFromPtr accepts a nil input and
// produces a Stream whose Ptr() reports nil. This is the path compute's
// ensureNotCapturing uses to short-circuit before invoking the CUDA probe on
// a stream that was never bound to a vendor handle.
func TestStreamFromPtr_NilHandle(t *testing.T) {
	s := StreamFromPtr(nil)
	if s == nil {
		t.Fatal("StreamFromPtr(nil) returned nil Stream")
	}
	if got := s.Ptr(); got != nil {
		t.Fatalf("StreamFromPtr(nil).Ptr(): got %p, want nil", got)
	}
}

// TestStreamCaptureStatus_ZeroStream exercises the path where the caller
// hands in a Stream whose handle is the zero value (e.g. a freshly wrapped
// nil pointer). When the CUDA runtime is unavailable, the binding must still
// return CaptureStatusNone with no error rather than panicking on the zero
// handle.
func TestStreamCaptureStatus_ZeroStream(t *testing.T) {
	if Available() {
		// On CUDA-enabled hosts the zero handle is invalid; skip instead of
		// probing the driver with an illegal argument.
		t.Skip("zero-handle probe is only safe when CUDA is unavailable")
	}
	var s Stream // handle == 0
	status, err := StreamCaptureStatus(&s)
	if err != nil {
		t.Fatalf("StreamCaptureStatus(zero stream) returned error: %v", err)
	}
	if status != CaptureStatusNone {
		t.Fatalf("StreamCaptureStatus(zero stream): got %v, want CaptureStatusNone", status)
	}
}

func TestCaptureStatus_EnumValues(t *testing.T) {
	// Compile-time exhaustive switch — ensures enum values stay stable and
	// every variant remains addressable from client code.
	cases := []CaptureStatus{
		CaptureStatusNone,
		CaptureStatusActive,
		CaptureStatusInvalidated,
	}
	for _, c := range cases {
		switch c {
		case CaptureStatusNone:
			if int(c) != 0 {
				t.Errorf("CaptureStatusNone = %d, want 0", int(c))
			}
		case CaptureStatusActive:
			if int(c) != 1 {
				t.Errorf("CaptureStatusActive = %d, want 1", int(c))
			}
		case CaptureStatusInvalidated:
			if int(c) != 2 {
				t.Errorf("CaptureStatusInvalidated = %d, want 2", int(c))
			}
		}
	}
}

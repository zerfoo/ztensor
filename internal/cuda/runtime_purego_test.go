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

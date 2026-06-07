package cuda

import (
	"errors"
	"testing"
	"unsafe"
)

// TestCaptureActive_SetSemantics verifies the capturing-stream registry behaves
// as an idempotent set: CaptureActive() is true while any handle is marked, and
// double-unmark (the watchdog force-end + normal end for one capture) does not
// underflow it.
func TestCaptureActive_SetSemantics(t *testing.T) {
	// Clean slate (other tests in this package may have left marks).
	resetCaptureStateForTest(t)

	if CaptureActive() {
		t.Fatal("CaptureActive() should be false with no marked streams")
	}

	const h1 = uintptr(0x1000)
	const h2 = uintptr(0x2000)

	markStreamCapturing(h1)
	if !CaptureActive() {
		t.Fatal("CaptureActive() should be true after marking a stream")
	}

	// A second stream begins; ending the first must not clear the active flag.
	markStreamCapturing(h2)
	unmarkStreamCapturing(h1)
	if !CaptureActive() {
		t.Fatal("CaptureActive() should remain true while a second stream captures")
	}

	// Double-unmark of h2 (watchdog force-end + normal end) must be a no-op the
	// second time, not an underflow that wrongly reports activity later.
	unmarkStreamCapturing(h2)
	unmarkStreamCapturing(h2)
	if CaptureActive() {
		t.Fatal("CaptureActive() should be false after all streams unmarked")
	}

	// After the double-unmark, a fresh capture must still register correctly.
	markStreamCapturing(h1)
	if !CaptureActive() {
		t.Fatal("registry corrupted by double-unmark: fresh mark not active")
	}
	unmarkStreamCapturing(h1)
}

// TestArenaPool_Alloc_CaptureGuard verifies the exhaustion fallback refuses a
// synchronous cudaMalloc while a capture is active and the fallback is not
// capture-aware (the issue #111 hang path), but proceeds normally otherwise.
func TestArenaPool_Alloc_CaptureGuard(t *testing.T) {
	resetCaptureStateForTest(t)

	const captureHandle = uintptr(0xCAFE)

	newExhaustedArena := func(fallback *MemPool) *ArenaPool {
		// capacity 0 + empty free-list => every Alloc takes the fallback path.
		return &ArenaPool{
			capacity:     0,
			fallback:     fallback,
			fallbackPtrs: make(map[unsafe.Pointer]int),
		}
	}

	t.Run("refuses sync fallback during capture", func(t *testing.T) {
		a := newExhaustedArena(NewMemPool())
		markStreamCapturing(captureHandle)
		defer unmarkStreamCapturing(captureHandle)

		_, err := a.Alloc(0, 256)
		if !errors.Is(err, ErrCaptureUnsafeAlloc) {
			t.Fatalf("expected ErrCaptureUnsafeAlloc during capture, got %v", err)
		}
		// The guard must short-circuit before touching the fallback pool.
		if got := a.misses.Load(); got != 1 {
			t.Fatalf("expected exactly one recorded miss, got %d", got)
		}
	})

	t.Run("allows fallback when no capture active", func(t *testing.T) {
		a := newExhaustedArena(NewMemPool())
		// No marked streams: the guard must not fire.
		_, err := a.Alloc(0, 256)
		if errors.Is(err, ErrCaptureUnsafeAlloc) {
			t.Fatal("guard fired with no capture active")
		}
		// On CPU the fallback itself fails ("cuda not available"); the point is
		// only that the guard did not pre-empt it.
		if err == nil {
			t.Fatal("expected a fallback error on CPU, got nil")
		}
	})

	t.Run("allows fallback when fallback is capture-aware", func(t *testing.T) {
		fallback := NewMemPool()
		// A capture-aware fallback routes through cudaMallocAsync, which is
		// graph-safe, so the guard must not refuse it. A non-nil stream is
		// enough to make IsCapturing() report true.
		fallback.SetCaptureStream(&Stream{})
		a := newExhaustedArena(fallback)

		markStreamCapturing(captureHandle)
		defer unmarkStreamCapturing(captureHandle)

		_, err := a.Alloc(0, 256)
		if errors.Is(err, ErrCaptureUnsafeAlloc) {
			t.Fatal("guard fired despite a capture-aware fallback")
		}
	})
}

// resetCaptureStateForTest clears the package capture registry so tests do not
// leak capture marks into one another.
func resetCaptureStateForTest(t *testing.T) {
	t.Helper()
	captureMu.Lock()
	capturingStreams = make(map[uintptr]struct{})
	captureMu.Unlock()
}

package cuda

import (
	"errors"
	"testing"
	"unsafe"
)

// newExhaustedArenaWithStream builds a capacity-0 arena (every Alloc takes the
// fallback path) with the given overflow stream and a real MemPool fallback.
func newExhaustedArenaWithStream(overflow *Stream) *ArenaPool {
	a := &ArenaPool{
		capacity:     0,
		fallback:     NewMemPool(),
		fallbackPtrs: make(map[unsafe.Pointer]int),
	}
	if overflow != nil {
		a.SetOverflowStream(overflow)
	}
	return a
}

// stubArenaAsync swaps the async indirections for the duration of a test and
// records calls. Returns recorders for malloc and free.
func stubArenaAsync(t *testing.T, mallocPtr unsafe.Pointer, mallocErr error) (mallocCalls *int, freed *[]unsafe.Pointer) {
	t.Helper()
	mc := 0
	var fr []unsafe.Pointer
	origMalloc := arenaMallocAsyncFn
	origFree := arenaFreeAsyncFn
	arenaMallocAsyncFn = func(_ int, _ *Stream) (unsafe.Pointer, error) {
		mc++
		return mallocPtr, mallocErr
	}
	arenaFreeAsyncFn = func(ptr unsafe.Pointer, _ *Stream) error {
		fr = append(fr, ptr)
		return nil
	}
	t.Cleanup(func() {
		arenaMallocAsyncFn = origMalloc
		arenaFreeAsyncFn = origFree
	})
	return &mc, &fr
}

// TestArenaPool_Overflow_RoutesAsyncWhenStreamSet verifies the issue #115 fix:
// with an overflow stream set and no capture active, an exhausted Alloc routes
// through cudaMallocAsync (the stub), NOT the synchronous MemPool fallback.
func TestArenaPool_Overflow_RoutesAsyncWhenStreamSet(t *testing.T) {
	resetCaptureStateForTest(t)
	var sentinel byte
	want := unsafe.Pointer(&sentinel)
	mallocCalls, _ := stubArenaAsync(t, want, nil)

	a := newExhaustedArenaWithStream(&Stream{})
	got, err := a.Alloc(0, 4096)
	if err != nil {
		t.Fatalf("async overflow Alloc: unexpected error %v", err)
	}
	if got != want {
		t.Fatalf("expected the async-stub pointer, got %v", got)
	}
	if *mallocCalls != 1 {
		t.Fatalf("expected exactly one async malloc, got %d", *mallocCalls)
	}
	// The pointer must be tracked for async free routing.
	if _, ok := a.asyncFallbackPtrs[got]; !ok {
		t.Fatal("async-allocated pointer was not recorded in asyncFallbackPtrs")
	}
}

// TestArenaPool_Overflow_CaptureGuardTakesPrecedence verifies the #111 guard
// still fires during graph-driven capture and is not bypassed by the new async
// overflow path.
func TestArenaPool_Overflow_CaptureGuardTakesPrecedence(t *testing.T) {
	resetCaptureStateForTest(t)
	var sentinel byte
	mallocCalls, _ := stubArenaAsync(t, unsafe.Pointer(&sentinel), nil)

	a := newExhaustedArenaWithStream(&Stream{}) // overflow stream set
	// Simulate a graph-driven capture: a stream is capturing, but the fallback
	// MemPool is NOT capture-aware (its captureStream is unset).
	const captureHandle = uintptr(0xC0FFEE)
	markStreamCapturing(captureHandle)
	defer unmarkStreamCapturing(captureHandle)

	_, err := a.Alloc(0, 4096)
	if !errors.Is(err, ErrCaptureUnsafeAlloc) {
		t.Fatalf("expected ErrCaptureUnsafeAlloc (guard precedence), got %v", err)
	}
	if *mallocCalls != 0 {
		t.Fatalf("async malloc must NOT run while the capture guard fires, ran %d", *mallocCalls)
	}
}

// TestArenaPool_Overflow_FreeRoutesAsync verifies an async-allocated overflow
// pointer is freed via cudaFreeAsync, not the synchronous MemPool.Free.
func TestArenaPool_Overflow_FreeRoutesAsync(t *testing.T) {
	resetCaptureStateForTest(t)
	var sentinel byte
	want := unsafe.Pointer(&sentinel)
	_, freed := stubArenaAsync(t, want, nil)

	a := newExhaustedArenaWithStream(&Stream{})
	got, err := a.Alloc(0, 4096)
	if err != nil {
		t.Fatalf("async overflow Alloc: %v", err)
	}
	a.Free(0, got, 4096)
	if len(*freed) != 1 || (*freed)[0] != got {
		t.Fatalf("expected async free of the overflow pointer, got %v", *freed)
	}
	if _, ok := a.asyncFallbackPtrs[got]; ok {
		t.Fatal("async pointer should be removed from asyncFallbackPtrs after Free")
	}
}

// TestArenaPool_Overflow_NoStreamUnchanged verifies that with no overflow stream
// set, behavior is unchanged: the synchronous MemPool fallback runs (and on CPU
// returns its not-available error), and the async path is not taken.
func TestArenaPool_Overflow_NoStreamUnchanged(t *testing.T) {
	resetCaptureStateForTest(t)
	var sentinel byte
	mallocCalls, _ := stubArenaAsync(t, unsafe.Pointer(&sentinel), nil)

	a := newExhaustedArenaWithStream(nil) // no overflow stream
	_, err := a.Alloc(0, 4096)
	if *mallocCalls != 0 {
		t.Fatalf("async malloc must not run without an overflow stream, ran %d", *mallocCalls)
	}
	// On CPU the synchronous fallback fails ("cuda not available"); the point is
	// only that the async path was not taken and the guard sentinel was not used.
	if errors.Is(err, ErrCaptureUnsafeAlloc) {
		t.Fatal("capture guard fired with no capture active")
	}
	if err == nil {
		t.Fatal("expected a synchronous fallback error on CPU, got nil")
	}
}

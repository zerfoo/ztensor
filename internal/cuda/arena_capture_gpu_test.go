package cuda

import (
	"errors"
	"testing"
)

// TestArenaPool_CaptureGuard_GPU is the on-hardware (GB10) validation for
// issue #111. With a REAL CUDA stream capture active, ArenaPool.Alloc must
// refuse the exhaustion fallback (returning ErrCaptureUnsafeAlloc) instead of
// issuing a synchronous cudaMalloc -- which on GB10 (sm_121) hangs the driver
// in an uninterruptible D-state. It also proves the capture marking works on
// real CUDA: CaptureActive() is true between a real StreamBeginCapture and
// StreamEndCapture, and false afterward.
//
// On CPU (Available() == false) it skips; the pure-logic guard is covered by
// TestArenaPool_Alloc_CaptureGuard. Run on the GB10 via Spark (exit-code
// guarded) -- see docs/bench/manifests/issue-111-capture-guard.yaml.
//
// This test is SAFE to run with the fix in place: the guard returns before any
// synchronous cudaMalloc, so no wedge can occur. Running it against the pre-fix
// arena code would instead exercise the synchronous fallback during capture and
// wedge the host -- do not do that on a shared GB10 without coordination.
func TestArenaPool_CaptureGuard_GPU(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}
	l := lib()
	if l == nil || !l.GraphAvailable() {
		t.Skip("CUDA graph capture not available")
	}

	const deviceID = 0
	if err := SetDevice(deviceID); err != nil {
		t.Fatalf("SetDevice: %v", err)
	}

	// A tiny arena so any real allocation overflows into the fallback path.
	const arenaCap = 4096
	fallback := NewMemPool()
	arena, err := NewArenaPool(deviceID, arenaCap, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	stream, err := CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	// Begin a real capture; this must mark the stream as capturing.
	if err := StreamBeginCapture(stream); err != nil {
		t.Fatalf("StreamBeginCapture: %v", err)
	}
	if !CaptureActive() {
		_, _ = StreamEndCapture(stream) // leave the stream clean before failing
		t.Fatal("CaptureActive() should be true after StreamBeginCapture on GPU")
	}

	// 1 MiB far exceeds the 4 KiB arena, forcing the exhaustion fallback.
	// Pre-fix this did a synchronous cudaMalloc during capture and wedged the
	// GB10. With the fix it must return ErrCaptureUnsafeAlloc without hanging.
	ptr, allocErr := arena.Alloc(deviceID, 1<<20)

	// End capture regardless and discard the (empty) graph so the stream is
	// clean for the post-capture assertions below.
	if g, endErr := StreamEndCapture(stream); endErr == nil && g != nil {
		_ = GraphDestroy(g)
	}

	if ptr != nil {
		t.Fatal("expected nil pointer from a refused alloc during capture")
	}
	if !errors.Is(allocErr, ErrCaptureUnsafeAlloc) {
		t.Fatalf("expected ErrCaptureUnsafeAlloc during capture, got %v", allocErr)
	}

	// After capture ends the guard must release: CaptureActive() is false and a
	// normal fallback allocation now succeeds, proving the guard only fires
	// during capture rather than permanently disabling the fallback.
	if CaptureActive() {
		t.Fatal("CaptureActive() should be false after StreamEndCapture")
	}
	ptr2, err2 := arena.Alloc(deviceID, 1<<20)
	if err2 != nil {
		t.Fatalf("post-capture fallback alloc should succeed, got %v", err2)
	}
	arena.Free(deviceID, ptr2, 1<<20)
}

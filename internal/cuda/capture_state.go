package cuda

import (
	"errors"
	"sync"
)

// ErrCaptureUnsafeAlloc is returned by an allocator when it would have to issue
// a synchronous cudaMalloc while a CUDA stream capture is active. On GB10
// (sm_121) a synchronous cudaMalloc during stream capture hangs the driver in an
// uninterruptible state (see docs/adr/004-capture-aware-arena-fallback.md and
// issue #111). Callers must treat this as a recoverable allocation failure --
// fall back to a CPU path, or let the capture fail cleanly and re-run
// uncaptured -- never as a hard error to retry in place.
var ErrCaptureUnsafeAlloc = errors.New("cuda: refusing synchronous device alloc during graph capture")

// capturingStreams tracks the set of CUDA stream handles that are currently
// capturing a graph. It is a set rather than a counter so that ending a capture
// is idempotent: the watchdog (graph/cuda_graph.go) and the normal path can both
// call StreamEndCapture for the same stream, and removing an absent handle is a
// no-op. A counter would underflow on that double-end.
var (
	captureMu        sync.Mutex
	capturingStreams = make(map[uintptr]struct{})
)

// markStreamCapturing records that the given stream handle has begun capturing.
// Called by StreamBeginCapture only after the runtime reports success.
func markStreamCapturing(handle uintptr) {
	captureMu.Lock()
	capturingStreams[handle] = struct{}{}
	captureMu.Unlock()
}

// unmarkStreamCapturing records that the given stream handle is no longer
// capturing. Idempotent: removing a handle that is not present is a no-op, so
// the watchdog force-end and the normal end can both call it for one capture.
func unmarkStreamCapturing(handle uintptr) {
	captureMu.Lock()
	delete(capturingStreams, handle)
	captureMu.Unlock()
}

// CaptureActive reports whether any CUDA stream is currently capturing a graph.
// Allocators consult this before issuing a synchronous cudaMalloc: such a malloc
// during capture hangs the GB10 driver. On CPU-only builds the CUDA runtime is
// unavailable, StreamBeginCapture never succeeds, and this is always false.
func CaptureActive() bool {
	captureMu.Lock()
	active := len(capturingStreams) > 0
	captureMu.Unlock()
	return active
}

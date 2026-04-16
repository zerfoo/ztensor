package compute

import "errors"

// ErrCaptureIncompatibleAllocation is returned when a weight allocation
// or upload is attempted while a CUDA graph capture is active on the
// engine's stream. Allocations during capture are not supported and
// would silently hang on GB10. Callers should either upload weights
// before BeginCapture, or catch this error and fall back to an
// uncaptured run.
var ErrCaptureIncompatibleAllocation = errors.New("compute: allocation attempted during active CUDA graph capture")

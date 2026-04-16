package compute

import (
	"errors"

	"github.com/zerfoo/ztensor/tensor"
)

// CaptureSafe attempts to execute fn under CUDA graph capture using
// engine.WithCapture. If capture succeeds, it returns the GraphHandle
// for replay. If capture fails with ErrCaptureIncompatibleAllocation
// (an allocation was attempted that is incompatible with capture),
// fn is re-executed uncaptured on the same stream and a zero
// GraphHandle is returned with a nil error.
//
// Any other error from WithCapture is returned as-is.
func CaptureSafe[T tensor.Numeric](engine *GPUEngine[T], fn func() error) (GraphHandle, error) {
	handle, err := engine.WithCapture(fn)
	if err == nil {
		return handle, nil
	}
	if errors.Is(err, ErrCaptureIncompatibleAllocation) {
		return GraphHandle{}, fn()
	}
	return GraphHandle{}, err
}

package compute

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/internal/gpuapi"
	"github.com/zerfoo/ztensor/tensor"
)

// f32Size and the GPU helpers (getDevicePtr, tryReuseDstPtr, finishReusedDst,
// makeGPUResult, isFloat32) are defined in gpu_kernels.go.

// Dropout applies inverted dropout to a on the GPU using a deterministic Philox
// mask keyed by (seed, element offset) -- the same Philox the CPU engine uses,
// so the masks (and outputs, given identical inputs) are bit-identical across
// CPU and GPU. In eval mode (training==false) or p==0 the kernel performs an
// exact identity copy. p must be in [0, 1). The mask is recomputed in
// DropoutBackward from (seed, p) rather than cached, keeping the op
// capture-safe (no save pinned across an arena reset; ztensor ADR 006).
func (e *GPUEngine[T]) Dropout(ctx context.Context, a *tensor.TensorNumeric[T], p float64, seed uint64, training bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.dropoutGPU(ctx, a, p, seed, training, dst...)
}

// DropoutBackward propagates g through the same mask Dropout used for the
// identical (p, seed, training). Dropout is linear in its input given the mask,
// so the backward is the same masked-and-scaled map applied to g.
func (e *GPUEngine[T]) DropoutBackward(ctx context.Context, g *tensor.TensorNumeric[T], p float64, seed uint64, training bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.dropoutGPU(ctx, g, p, seed, training, dst...)
}

// dropoutGPU is the shared forward/backward GPU launcher.
func (e *GPUEngine[T]) dropoutGPU(_ context.Context, a *tensor.TensorNumeric[T], p float64, seed uint64, training bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, fmt.Errorf("dropout: input tensor cannot be nil")
	}
	if p < 0 || p >= 1 {
		return nil, fmt.Errorf("dropout: p must be in [0, 1), got %g", p)
	}
	if !isFloat32[T]() {
		return nil, fmt.Errorf("GPU dropout: unsupported type, only float32")
	}
	dk, ok := e.kernels.(gpuapi.Dropouter)
	if !ok {
		return nil, fmt.Errorf("GPU dropout: kernel runner does not provide the dropout kernel")
	}

	n := a.GetStorage().Len()
	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return nil, err
	}
	defer cleanupA()

	byteSize := n * f32Size
	devC, reused := tryReuseDstPtr[T](n, dst)
	if !reused {
		devC, err = e.pool.Alloc(e.deviceID, byteSize)
		if err != nil {
			return nil, err
		}
	}

	invKeep := float32(1.0 / (1.0 - p))
	if err := dk.DropoutF32(devA, devC, n, float32(p), seed, training, invKeep, e.stream); err != nil {
		if !reused {
			e.pool.Free(e.deviceID, devC, byteSize)
		}
		return nil, err
	}

	if reused {
		return finishReusedDst[T](dst[0], a.Shape()), nil
	}
	return makeGPUResult[T](e, a.Shape(), devC, n, dst...)
}

// Compile-time assertion: GPUEngine[float32] satisfies the Dropouter capability.
var _ Dropouter[float32] = (*GPUEngine[float32])(nil)

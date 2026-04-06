package compute

// gpu_engine_elementwise.go contains all element-wise operations, scalar
// operations, fused element-wise kernels (RoPE, SwiGLU, RMSNorm), and
// pairwise operations (CosineSimilarity, HadamardTransform) for GPUEngine.

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// UnaryOp applies an arbitrary unary function element-wise (CPU fallback).
func (e *GPUEngine[T]) UnaryOp(ctx context.Context, a *tensor.TensorNumeric[T], op func(T) T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.UnaryOp(ctx, a, op, dst...)
}

// Add performs element-wise addition.
func (e *GPUEngine[T]) Add(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuAdd(ctx, a, b, dst...)
}

// Sub performs element-wise subtraction.
func (e *GPUEngine[T]) Sub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuSub(ctx, a, b, dst...)
}

// Mul performs element-wise multiplication.
func (e *GPUEngine[T]) Mul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuMul(ctx, a, b, dst...)
}

// Div performs element-wise division.
func (e *GPUEngine[T]) Div(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuDiv(ctx, a, b, dst...)
}

// Exp computes the element-wise exponential.
func (e *GPUEngine[T]) Exp(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuExp(ctx, a, dst...)
}

// Log computes the element-wise natural logarithm.
func (e *GPUEngine[T]) Log(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuLog(ctx, a, dst...)
}

// Sin computes the element-wise sine.
func (e *GPUEngine[T]) Sin(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuSin(ctx, a, dst...)
}

// Cos computes the element-wise cosine.
func (e *GPUEngine[T]) Cos(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuCos(ctx, a, dst...)
}

// Tanh computes the element-wise hyperbolic tangent.
func (e *GPUEngine[T]) Tanh(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuTanh(ctx, a, dst...)
}

// TanhPrime computes the element-wise gradient of tanh.
func (e *GPUEngine[T]) TanhPrime(ctx context.Context, a, upstream *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuTanhPrime(ctx, a, upstream, dst...)
}

// Pow raises each element to the given power.
func (e *GPUEngine[T]) Pow(ctx context.Context, base, exponent *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuPow(ctx, base, exponent, dst...)
}

// MulScalar multiplies each element by a scalar.
func (e *GPUEngine[T]) MulScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuMulScalar(ctx, a, scalar, dst...)
}

// DivScalar divides each element by a scalar.
func (e *GPUEngine[T]) DivScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuDivScalar(ctx, a, scalar, dst...)
}

// AddScalar adds a scalar to each element.
func (e *GPUEngine[T]) AddScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuAddScalar(ctx, a, scalar, dst...)
}

// Sqrt computes the element-wise square root.
func (e *GPUEngine[T]) Sqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuSqrt(ctx, a, dst...)
}

// Rsqrt computes the element-wise reciprocal square root.
func (e *GPUEngine[T]) Rsqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuRsqrt(ctx, a, dst...)
}

// CosineSimilarity computes pairwise cosine similarity between rows of two 2D tensors.
// a has shape [M, D], b has shape [N, D]. Result has shape [M, N].
// Currently delegates to CPUEngine; a dedicated GPU kernel will be added later.
func (e *GPUEngine[T]) CosineSimilarity(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.CosineSimilarity(ctx, a, b, dst...)
}

// HadamardTransform delegates to the CPU engine.
func (e *GPUEngine[T]) HadamardTransform(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.HadamardTransform(ctx, a, dst...)
}

// GPUFusedRoPE applies rotary position embeddings in a single GPU kernel launch.
// This replaces Split + 4 Mul + Sub + Add + Concat (8 operations, ~10 D2D memcpy) with 1 kernel.
func (e *GPUEngine[T]) GPUFusedRoPE(input, cosAngles, sinAngles *tensor.TensorNumeric[T], rotaryDim int) (*tensor.TensorNumeric[T], error) {
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("GPUFusedRoPE: expected 3D input [batch, seq, dim], got %dD", len(shape))
	}

	batch := shape[0]
	seqLen := shape[1]
	headDim := shape[2]
	halfRotary := rotaryDim / 2

	cosShape := cosAngles.Shape()
	if len(cosShape) != 2 || cosShape[0] < seqLen || cosShape[1] < halfRotary {
		return nil, fmt.Errorf("GPUFusedRoPE: cos shape %v incompatible with seq_len=%d half_rotary=%d", cosShape, seqLen, halfRotary)
	}
	cosStride := cosShape[1]

	// Get device pointers for input, cos, sin.
	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedRoPE input: %w", err)
	}
	defer inCleanup()

	cosPtr, cosCleanup, err := getDevicePtr(e, cosAngles)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedRoPE cos: %w", err)
	}
	defer cosCleanup()

	sinPtr, sinCleanup, err := getDevicePtr(e, sinAngles)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedRoPE sin: %w", err)
	}
	defer sinCleanup()

	// Allocate output.
	outElems := batch * seqLen * headDim
	outBytes := outElems * f32Size
	e.setDevice()
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedRoPE alloc: %w", err)
	}

	if err := e.kernels.FusedRoPEF32(inPtr, cosPtr, sinPtr, devOut, batch, seqLen, headDim, halfRotary, cosStride, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}

	return makeGPUResult[T](e, shape, devOut, outElems)
}

// GPUFusedSwiGLU computes SwiGLU(w1, w3) = w1 * sigmoid(w1) * w3 in a single GPU kernel.
// This replaces Concat + Split + sigmoid + Mul + Mul (5 operations, ~4 D2D memcpy per layer) with 1 kernel.
func (e *GPUEngine[T]) GPUFusedSwiGLU(w1, w3 *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	w1Shape := w1.Shape()
	w3Shape := w3.Shape()
	if len(w1Shape) == 0 || len(w3Shape) == 0 {
		return nil, fmt.Errorf("GPUFusedSwiGLU: empty shape")
	}

	// Validate shapes match.
	n1 := 1
	for _, d := range w1Shape {
		n1 *= d
	}
	n3 := 1
	for _, d := range w3Shape {
		n3 *= d
	}
	if n1 != n3 {
		return nil, fmt.Errorf("GPUFusedSwiGLU: w1 (%d elems) and w3 (%d elems) size mismatch", n1, n3)
	}

	w1Ptr, w1Cleanup, err := getDevicePtr(e, w1)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedSwiGLU w1: %w", err)
	}
	defer w1Cleanup()

	w3Ptr, w3Cleanup, err := getDevicePtr(e, w3)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedSwiGLU w3: %w", err)
	}
	defer w3Cleanup()

	outBytes := n1 * f32Size
	e.setDevice()
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedSwiGLU alloc: %w", err)
	}

	if err := e.kernels.FusedSwiGLUF32(w1Ptr, w3Ptr, devOut, n1, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}

	return makeGPUResult[T](e, w1Shape, devOut, n1)
}

// GPUFusedAddRMSNorm computes sum = input + residual and
// normed = rmsnorm(sum, weight, eps) in a single GPU kernel launch.
// Both inputs are read-only; outputs go to separate buffers.
// This replaces Add + RMSNorm (2 kernel launches) with 1.
func (e *GPUEngine[T]) GPUFusedAddRMSNorm(
	input, residual *tensor.TensorNumeric[T],
	weight *tensor.TensorNumeric[T],
	eps float32,
) (normed *tensor.TensorNumeric[T], residualOut *tensor.TensorNumeric[T], scales *tensor.TensorNumeric[T], err error) {
	// FP16 paths — skip entirely for F32 compute.
	if e.dtype != DTypeF32 {
		// Native FP16 path: input and residual already have Float16Storage — no conversion needed.
		inFS, inOK := any(input.GetStorage()).(*tensor.Float16Storage)
		resFS, resOK := any(residual.GetStorage()).(*tensor.Float16Storage)
		if inOK && resOK {
			return fp16FusedAddRMSNormNative(e, inFS, resFS, input, weight, eps)
		}
		// FP16 path: decompose into F32 Add + FP16 RMSNorm.
		return fp16FusedAddRMSNorm(e, input, residual, weight, eps)
	}

	inShape := input.Shape()
	if len(inShape) < 2 {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm: input must be at least 2D, got %v", inShape)
	}
	D := inShape[len(inShape)-1]
	rows := 1
	for i := 0; i < len(inShape)-1; i++ {
		rows *= inShape[i]
	}

	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm input: %w", err)
	}
	defer inCleanup()

	// Residual is updated in-place. We need a mutable device pointer.
	resPtr, resCleanup, err := getDevicePtr(e, residual)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm residual: %w", err)
	}
	defer resCleanup()

	wPtr, wCleanup, err := getDevicePtr(e, weight)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm weight: %w", err)
	}
	defer wCleanup()

	outBytes := rows * D * f32Size
	e.setDevice()
	devNormed, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm alloc normed: %w", err)
	}

	devSum, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		e.pool.Free(e.deviceID, devNormed, outBytes)
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm alloc sum: %w", err)
	}

	if err := e.kernels.FusedAddRMSNormF32(inPtr, resPtr, wPtr, devNormed, devSum, eps, rows, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, devNormed, outBytes)
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, err
	}

	normed, err = makeGPUResult[T](e, inShape, devNormed, rows*D)
	if err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, err
	}

	residualOut, err = makeGPUResult[T](e, inShape, devSum, rows*D)
	if err != nil {
		return nil, nil, nil, err
	}

	return normed, residualOut, nil, nil
}

// GPUFusedNormAdd computes output = rmsnorm(input, weight, eps) + residual
// in a single GPU kernel launch. Replaces separate RMSNorm + Add (2 launches → 1).
func (e *GPUEngine[T]) GPUFusedNormAdd(input, weight, residual *tensor.TensorNumeric[T], eps float32) (*tensor.TensorNumeric[T], error) {
	inShape := input.Shape()
	if len(inShape) < 2 {
		return nil, fmt.Errorf("GPUFusedNormAdd: input must be at least 2D, got %v", inShape)
	}
	D := inShape[len(inShape)-1]
	rows := 1
	for i := 0; i < len(inShape)-1; i++ {
		rows *= inShape[i]
	}

	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd input: %w", err)
	}
	defer inCleanup()

	wPtr, wCleanup, err := getDevicePtr(e, weight)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd weight: %w", err)
	}
	defer wCleanup()

	resPtr, resCleanup, err := getDevicePtr(e, residual)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd residual: %w", err)
	}
	defer resCleanup()

	outElems := rows * D
	outBytes := outElems * f32Size
	e.setDevice()
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd alloc: %w", err)
	}

	if err := e.kernels.FusedNormAddF32(inPtr, wPtr, resPtr, devOut, eps, rows, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}

	return makeGPUResult[T](e, inShape, devOut, outElems)
}

// GPUFusedQKNormRoPE applies per-head RMSNorm + RoPE to combined Q+K heads
// in a single GPU kernel launch. This replaces 4 kernel launches per GQA layer.
// input: [totalHeads, headDim], weightQ/weightK: [headDim],
// cosAngles/sinAngles: [halfRotary], output: [totalHeads, headDim].
func (e *GPUEngine[T]) GPUFusedQKNormRoPE(
	input *tensor.TensorNumeric[T],
	weightQ, weightK *tensor.TensorNumeric[T],
	cosAngles, sinAngles *tensor.TensorNumeric[T],
	eps float32,
	totalHeads, headDim, numQHeads, halfRotary int,
) (*tensor.TensorNumeric[T], error) {
	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE input: %w", err)
	}
	defer inCleanup()

	wqPtr, wqCleanup, err := getDevicePtr(e, weightQ)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE weightQ: %w", err)
	}
	defer wqCleanup()

	wkPtr, wkCleanup, err := getDevicePtr(e, weightK)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE weightK: %w", err)
	}
	defer wkCleanup()

	cosPtr, cosCleanup, err := getDevicePtr(e, cosAngles)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE cos: %w", err)
	}
	defer cosCleanup()

	sinPtr, sinCleanup, err := getDevicePtr(e, sinAngles)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE sin: %w", err)
	}
	defer sinCleanup()

	outElems := totalHeads * headDim
	outBytes := outElems * f32Size
	e.setDevice()
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE alloc: %w", err)
	}

	if err := e.kernels.FusedQKNormRoPEF32(inPtr, wqPtr, wkPtr, cosPtr, sinPtr, devOut, eps, totalHeads, headDim, numQHeads, halfRotary, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}

	return makeGPUResult[T](e, []int{totalHeads, headDim}, devOut, outElems)
}

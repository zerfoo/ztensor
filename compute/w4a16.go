package compute

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// W4A16Precision represents the mixed-precision configuration where weights
// are stored in 4-bit quantized format and activations are in FP16.
type W4A16Precision struct {
	// WeightFormat describes which 4-bit quantization is used.
	WeightFormat string // "Q4_0", "GPTQ_4", "AWQ"
}

// w4Storage is a unified interface for 4-bit weight storage types that support
// dequantization to float32.
type w4Storage interface {
	Len() int
	Slice() []float32
}

// detectW4A16 checks if A has 4-bit quantized weights and B has FP16 activations
// (or vice versa). Returns the quantized storage, the FP16 storage, and which
// operand holds the weights ("A" or "B").
func detectW4A16[T tensor.Numeric](a, b *tensor.TensorNumeric[T]) (w4 w4Storage, fp16Act *tensor.Float16Storage, weightSide string) {
	storA := a.GetStorage()
	storB := b.GetStorage()

	// Check A=weights(4-bit), B=activations(FP16).
	if w4s := extractW4Storage(storA); w4s != nil {
		if fp16s, ok := any(storB).(*tensor.Float16Storage); ok {
			return w4s, fp16s, "A"
		}
	}

	// Check B=weights(4-bit), A=activations(FP16).
	if w4s := extractW4Storage(storB); w4s != nil {
		if fp16s, ok := any(storA).(*tensor.Float16Storage); ok {
			return w4s, fp16s, "B"
		}
	}

	return nil, nil, ""
}

// extractW4Storage returns a w4Storage if the given storage is a recognized
// 4-bit quantization format.
func extractW4Storage[T tensor.Numeric](s tensor.Storage[T]) w4Storage {
	switch qs := any(s).(type) {
	case *tensor.Q4Storage:
		return qs
	case *tensor.GPTQStorage:
		if qs.Bits() == 4 {
			return qs
		}
	case *tensor.AWQStorage:
		return qs
	}
	return nil
}

// MatMulW4A16 performs mixed-precision matrix multiplication with 4-bit
// quantized weights and FP16 activations. The 4-bit weights are dequantized
// to float32 and the FP16 activations are decoded to float32 for computation.
// The result is a float32 tensor.
//
// This is the CPU fallback path. GPU engines can override with fused
// dequant-GEMM kernels for better performance.
func MatMulW4A16[T tensor.Numeric](
	ctx context.Context,
	eng Engine[T],
	a, b *tensor.TensorNumeric[T],
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	w4, fp16Act, side := detectW4A16(a, b)
	if w4 == nil || fp16Act == nil {
		return nil, fmt.Errorf("MatMulW4A16: inputs are not W4A16 — need 4-bit weights and FP16 activations")
	}

	// Dequantize 4-bit weights to float32.
	wF32 := w4.Slice()

	// Decode FP16 activations to float32.
	actF32 := fp16Act.Slice()

	// Build float32 tensors for the engine MatMul.
	var weightTensor, actTensor *tensor.TensorNumeric[T]
	var err error

	if side == "A" {
		// A=weights, B=activations.
		wStorage := anyStorage[T](wF32, a.Shape())
		aStorage := anyStorage[T](actF32, b.Shape())
		weightTensor, err = tensor.NewWithStorage[T](a.Shape(), wStorage)
		if err != nil {
			return nil, fmt.Errorf("MatMulW4A16: create weight tensor: %w", err)
		}
		actTensor, err = tensor.NewWithStorage[T](b.Shape(), aStorage)
		if err != nil {
			return nil, fmt.Errorf("MatMulW4A16: create activation tensor: %w", err)
		}
		return eng.MatMul(ctx, weightTensor, actTensor, dst...)
	}

	// B=weights, A=activations.
	aStorage := anyStorage[T](actF32, a.Shape())
	wStorage := anyStorage[T](wF32, b.Shape())
	actTensor, err = tensor.NewWithStorage[T](a.Shape(), aStorage)
	if err != nil {
		return nil, fmt.Errorf("MatMulW4A16: create activation tensor: %w", err)
	}
	weightTensor, err = tensor.NewWithStorage[T](b.Shape(), wStorage)
	if err != nil {
		return nil, fmt.Errorf("MatMulW4A16: create weight tensor: %w", err)
	}
	return eng.MatMul(ctx, actTensor, weightTensor, dst...)
}

// anyStorage creates a Storage[T] from a float32 slice, converting if needed.
// This handles the type assertion between float32 data and the generic T parameter.
func anyStorage[T tensor.Numeric](f32 []float32, shape []int) tensor.Storage[T] {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]T, size)
	for i := range size {
		if i < len(f32) {
			data[i] = any(f32[i]).(T)
		}
	}
	return tensor.NewCPUStorage(data)
}

// W4A16MatMuler is an optional interface for engines that support W4A16
// mixed-precision matrix multiplication with fused dequantization.
type W4A16MatMuler[T tensor.Numeric] interface {
	// MatMulW4A16 performs C = dequant(W_4bit) * A_fp16 with the weight and
	// activation operands identified by the caller.
	MatMulW4A16(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
}

// TryW4A16MatMul attempts to dispatch a W4A16 mixed-precision MatMul.
// Returns (result, true) if the inputs matched the W4A16 pattern, or
// (nil, false) if the inputs are not a W4A16 combination.
func TryW4A16MatMul[T tensor.Numeric](
	ctx context.Context,
	eng Engine[T],
	a, b *tensor.TensorNumeric[T],
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], bool, error) {
	w4, fp16Act, _ := detectW4A16(a, b)
	if w4 == nil || fp16Act == nil {
		return nil, false, nil
	}

	// If the engine natively supports W4A16, use it.
	if w4eng, ok := any(eng).(W4A16MatMuler[T]); ok {
		result, err := w4eng.MatMulW4A16(ctx, a, b, dst...)
		return result, true, err
	}

	// CPU fallback: dequantize and compute in float32.
	result, err := MatMulW4A16(ctx, eng, a, b, dst...)
	return result, true, err
}

// DequantW4ToFP16 dequantizes 4-bit quantized weights to FP16 format.
// This is useful for GPU paths that want FP16 weight data for cuBLAS HGEMM.
func DequantW4ToFP16(w4 w4Storage) *tensor.Float16Storage {
	f32 := w4.Slice()
	return tensor.NewFloat16StorageFromF32(f32)
}

// QuantFormat returns a string identifying the 4-bit quantization format
// of the given storage, or "" if it's not a recognized 4-bit format.
func QuantFormat[T tensor.Numeric](s tensor.Storage[T]) string {
	switch qs := any(s).(type) {
	case *tensor.Q4Storage:
		return "Q4_0"
	case *tensor.GPTQStorage:
		if qs.Bits() == 4 {
			return "GPTQ_4"
		}
	case *tensor.AWQStorage:
		return "AWQ"
	}
	return ""
}

// IsW4A16 returns true if the two tensors form a W4A16 mixed-precision pair
// (one operand has 4-bit weights, the other has FP16 activations).
func IsW4A16[T tensor.Numeric](a, b *tensor.TensorNumeric[T]) bool {
	w4, fp16, _ := detectW4A16(a, b)
	return w4 != nil && fp16 != nil
}

// W4A16Info returns metadata about a W4A16 mixed-precision pair.
// Returns zero value if the inputs are not a W4A16 combination.
func W4A16Info[T tensor.Numeric](a, b *tensor.TensorNumeric[T]) W4A16Precision {
	storA := a.GetStorage()
	storB := b.GetStorage()

	if fmt := QuantFormat[T](storA); fmt != "" {
		if _, ok := any(storB).(*tensor.Float16Storage); ok {
			return W4A16Precision{WeightFormat: fmt}
		}
	}
	if fmt := QuantFormat[T](storB); fmt != "" {
		if _, ok := any(storA).(*tensor.Float16Storage); ok {
			return W4A16Precision{WeightFormat: fmt}
		}
	}
	return W4A16Precision{}
}

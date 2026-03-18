package compute

import (
	"context"
	"errors"
	"math"

	float16 "github.com/zerfoo/float16"
	float8 "github.com/zerfoo/float8"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// fp8E4M3Max is the maximum representable value of FP8 E4M3FN format.
const fp8E4M3Max = float32(448.0)

// ComputeAmax returns the maximum absolute value of all elements in t as float32.
// It scans the tensor data on the CPU. For GPU tensors, data must be accessible
// on the host (e.g. via a prior device-to-host copy).
// Returns 0 for empty tensors.
func ComputeAmax[T tensor.Numeric](_ context.Context, ops numeric.Arithmetic[T], t *tensor.TensorNumeric[T]) (float32, error) {
	if t == nil {
		return 0, errors.New("ComputeAmax: tensor is nil")
	}
	data := t.Data()
	if len(data) == 0 {
		return 0, nil
	}

	maxVal := ops.Abs(data[0])
	for i := 1; i < len(data); i++ {
		absVal := ops.Abs(data[i])
		if ops.GreaterThan(absVal, maxVal) {
			maxVal = absVal
		}
	}

	return numericToFloat32(maxVal), nil
}

// ScaleForFP8 returns the scale factor for FP8 E4M3FN quantization: 448.0 / amax.
// Returns an error if the tensor is nil. Returns +Inf if amax is zero (all-zero tensor).
func ScaleForFP8[T tensor.Numeric](ctx context.Context, ops numeric.Arithmetic[T], t *tensor.TensorNumeric[T]) (float32, error) {
	amax, err := ComputeAmax(ctx, ops, t)
	if err != nil {
		return 0, err
	}
	if amax == 0 {
		return float32(math.Inf(1)), nil
	}
	return fp8E4M3Max / amax, nil
}

// numericToFloat32 converts a tensor.Numeric value to float32 via type assertion.
func numericToFloat32[T tensor.Numeric](v T) float32 {
	switch val := any(v).(type) {
	case float32:
		return val
	case float64:
		return float32(val)
	case float16.Float16:
		return val.ToFloat32()
	case float8.Float8:
		return val.ToFloat32()
	case int:
		return float32(val)
	case int8:
		return float32(val)
	case int16:
		return float32(val)
	case int32:
		return float32(val)
	case int64:
		return float32(val)
	case uint:
		return float32(val)
	case uint8:
		return float32(val)
	case uint32:
		return float32(val)
	case uint64:
		return float32(val)
	default:
		return 0
	}
}

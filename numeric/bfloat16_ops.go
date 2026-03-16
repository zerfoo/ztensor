package numeric

import (
	"math"

	"github.com/zerfoo/float16"
)

// BFloat16Ops provides the implementation of the Arithmetic interface for the float16.BFloat16 type.
type BFloat16Ops struct{}

// Add performs element-wise addition.
func (ops BFloat16Ops) Add(a, b float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16Add(a, b)
}

// Sub performs element-wise subtraction.
func (ops BFloat16Ops) Sub(a, b float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16Sub(a, b)
}

// Mul performs element-wise multiplication.
func (ops BFloat16Ops) Mul(a, b float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16Mul(a, b)
}

// Div performs element-wise division.
func (ops BFloat16Ops) Div(a, b float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16Div(a, b)
}

// Tanh computes the hyperbolic tangent of x.
func (ops BFloat16Ops) Tanh(x float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16FromFloat32(float32(math.Tanh(float64(x.ToFloat32()))))
}

// Sigmoid computes the sigmoid function of x.
func (ops BFloat16Ops) Sigmoid(x float16.BFloat16) float16.BFloat16 {
	f := x.ToFloat32()
	return float16.BFloat16FromFloat32(1.0 / (1.0 + float32(math.Exp(float64(-f)))))
}

// TanhGrad computes the gradient of the hyperbolic tangent function.
func (ops BFloat16Ops) TanhGrad(x float16.BFloat16) float16.BFloat16 {
	t := ops.Tanh(x)
	return ops.Sub(ops.One(), ops.Mul(t, t))
}

// SigmoidGrad computes the gradient of the sigmoid function.
func (ops BFloat16Ops) SigmoidGrad(x float16.BFloat16) float16.BFloat16 {
	s := ops.Sigmoid(x)
	return ops.Mul(s, ops.Sub(ops.One(), s))
}

// ReLU computes the Rectified Linear Unit function.
func (ops BFloat16Ops) ReLU(x float16.BFloat16) float16.BFloat16 {
	if x.ToFloat32() > 0 {
		return x
	}
	return float16.BFloat16FromFloat32(0)
}

// LeakyReLU computes the Leaky Rectified Linear Unit function.
func (ops BFloat16Ops) LeakyReLU(x float16.BFloat16, alpha float64) float16.BFloat16 {
	if x.ToFloat32() > 0 {
		return x
	}
	return ops.Mul(x, float16.BFloat16FromFloat32(float32(alpha)))
}

// ReLUGrad computes the gradient of the Rectified Linear Unit function.
func (ops BFloat16Ops) ReLUGrad(x float16.BFloat16) float16.BFloat16 {
	if x.ToFloat32() > 0 {
		return ops.One()
	}
	return float16.BFloat16FromFloat32(0)
}

// LeakyReLUGrad computes the gradient of the Leaky Rectified Linear Unit function.
func (ops BFloat16Ops) LeakyReLUGrad(x float16.BFloat16, alpha float64) float16.BFloat16 {
	if x.ToFloat32() > 0 {
		return ops.One()
	}
	return float16.BFloat16FromFloat32(float32(alpha))
}

// FromFloat32 converts a float32 to a float16.BFloat16.
func (ops BFloat16Ops) FromFloat32(f float32) float16.BFloat16 {
	return float16.BFloat16FromFloat32(f)
}

// FromFloat64 converts a float64 to a float16.BFloat16.
func (ops BFloat16Ops) FromFloat64(f float64) float16.BFloat16 {
	return float16.BFloat16FromFloat64(f)
}

// One returns a float16.BFloat16 with value 1.
func (ops BFloat16Ops) One() float16.BFloat16 {
	return float16.BFloat16FromFloat32(1)
}

// IsZero checks if the given float16.BFloat16 value is zero.
func (ops BFloat16Ops) IsZero(v float16.BFloat16) bool {
	return v.IsZero()
}

// Abs computes the absolute value of x.
func (ops BFloat16Ops) Abs(x float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16Abs(x)
}

// Sum computes the sum of elements in a slice.
func (ops BFloat16Ops) Sum(s []float16.BFloat16) float16.BFloat16 {
	var sum float16.BFloat16
	for _, v := range s {
		sum = float16.BFloat16Add(sum, v)
	}
	return sum
}

// Exp computes the exponential of x.
func (ops BFloat16Ops) Exp(x float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16FromFloat32(float32(math.Exp(float64(x.ToFloat32()))))
}

// Log computes the natural logarithm of x.
func (ops BFloat16Ops) Log(x float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16FromFloat32(float32(math.Log(float64(x.ToFloat32()))))
}

// Pow computes base raised to the power of exponent.
func (ops BFloat16Ops) Pow(base, exponent float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16FromFloat32(float32(math.Pow(float64(base.ToFloat32()), float64(exponent.ToFloat32()))))
}

// Sqrt computes the square root of x.
func (ops BFloat16Ops) Sqrt(x float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16FromFloat32(float32(math.Sqrt(float64(x.ToFloat32()))))
}

// GreaterThan checks if a is greater than b.
func (ops BFloat16Ops) GreaterThan(a, b float16.BFloat16) bool {
	return float16.BFloat16Greater(a, b)
}

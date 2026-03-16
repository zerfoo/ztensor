package numeric

import (
	"math"

	"github.com/zerfoo/float16"
)

// BFloat16Ops provides the implementation of the Arithmetic interface for the float16.BFloat16 type.
type BFloat16Ops struct{}

func (ops BFloat16Ops) Add(a, b float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16Add(a, b)
}

func (ops BFloat16Ops) Sub(a, b float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16Sub(a, b)
}

func (ops BFloat16Ops) Mul(a, b float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16Mul(a, b)
}

func (ops BFloat16Ops) Div(a, b float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16Div(a, b)
}

func (ops BFloat16Ops) Tanh(x float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16FromFloat32(float32(math.Tanh(float64(x.ToFloat32()))))
}

func (ops BFloat16Ops) Sigmoid(x float16.BFloat16) float16.BFloat16 {
	f := x.ToFloat32()
	return float16.BFloat16FromFloat32(1.0 / (1.0 + float32(math.Exp(float64(-f)))))
}

func (ops BFloat16Ops) TanhGrad(x float16.BFloat16) float16.BFloat16 {
	t := ops.Tanh(x)
	return ops.Sub(ops.One(), ops.Mul(t, t))
}

func (ops BFloat16Ops) SigmoidGrad(x float16.BFloat16) float16.BFloat16 {
	s := ops.Sigmoid(x)
	return ops.Mul(s, ops.Sub(ops.One(), s))
}

func (ops BFloat16Ops) ReLU(x float16.BFloat16) float16.BFloat16 {
	if x.ToFloat32() > 0 {
		return x
	}
	return float16.BFloat16FromFloat32(0)
}

func (ops BFloat16Ops) LeakyReLU(x float16.BFloat16, alpha float64) float16.BFloat16 {
	if x.ToFloat32() > 0 {
		return x
	}
	return ops.Mul(x, float16.BFloat16FromFloat32(float32(alpha)))
}

func (ops BFloat16Ops) ReLUGrad(x float16.BFloat16) float16.BFloat16 {
	if x.ToFloat32() > 0 {
		return ops.One()
	}
	return float16.BFloat16FromFloat32(0)
}

func (ops BFloat16Ops) LeakyReLUGrad(x float16.BFloat16, alpha float64) float16.BFloat16 {
	if x.ToFloat32() > 0 {
		return ops.One()
	}
	return float16.BFloat16FromFloat32(float32(alpha))
}

func (ops BFloat16Ops) FromFloat32(f float32) float16.BFloat16 {
	return float16.BFloat16FromFloat32(f)
}

func (ops BFloat16Ops) FromFloat64(f float64) float16.BFloat16 {
	return float16.BFloat16FromFloat64(f)
}

func (ops BFloat16Ops) One() float16.BFloat16 {
	return float16.BFloat16FromFloat32(1)
}

func (ops BFloat16Ops) IsZero(v float16.BFloat16) bool {
	return v.IsZero()
}

func (ops BFloat16Ops) Abs(x float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16Abs(x)
}

func (ops BFloat16Ops) Sum(s []float16.BFloat16) float16.BFloat16 {
	var sum float16.BFloat16
	for _, v := range s {
		sum = float16.BFloat16Add(sum, v)
	}
	return sum
}

func (ops BFloat16Ops) Exp(x float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16FromFloat32(float32(math.Exp(float64(x.ToFloat32()))))
}

func (ops BFloat16Ops) Log(x float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16FromFloat32(float32(math.Log(float64(x.ToFloat32()))))
}

func (ops BFloat16Ops) Pow(base, exponent float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16FromFloat32(float32(math.Pow(float64(base.ToFloat32()), float64(exponent.ToFloat32()))))
}

func (ops BFloat16Ops) Sqrt(x float16.BFloat16) float16.BFloat16 {
	return float16.BFloat16FromFloat32(float32(math.Sqrt(float64(x.ToFloat32()))))
}

func (ops BFloat16Ops) GreaterThan(a, b float16.BFloat16) bool {
	return float16.BFloat16Greater(a, b)
}

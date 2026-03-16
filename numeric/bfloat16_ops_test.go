package numeric

import (
	"math"
	"testing"

	"github.com/zerfoo/float16"
)

func TestBFloat16Ops_Arithmetic(t *testing.T) {
	ops := BFloat16Ops{}
	a := float16.BFloat16FromFloat32(3.0)
	b := float16.BFloat16FromFloat32(2.0)

	if got := ops.Add(a, b).ToFloat32(); math.Abs(float64(got-5.0)) > 0.1 {
		t.Errorf("Add = %v, want ~5.0", got)
	}
	if got := ops.Sub(a, b).ToFloat32(); math.Abs(float64(got-1.0)) > 0.1 {
		t.Errorf("Sub = %v, want ~1.0", got)
	}
	if got := ops.Mul(a, b).ToFloat32(); math.Abs(float64(got-6.0)) > 0.1 {
		t.Errorf("Mul = %v, want ~6.0", got)
	}
	if got := ops.Div(a, b).ToFloat32(); math.Abs(float64(got-1.5)) > 0.1 {
		t.Errorf("Div = %v, want ~1.5", got)
	}
}

func TestBFloat16Ops_Activations(t *testing.T) {
	ops := BFloat16Ops{}
	x := float16.BFloat16FromFloat32(0.5)

	tanh := ops.Tanh(x).ToFloat32()
	if math.Abs(float64(tanh)-math.Tanh(0.5)) > 0.05 {
		t.Errorf("Tanh(0.5) = %v, want ~%v", tanh, math.Tanh(0.5))
	}

	sig := ops.Sigmoid(x).ToFloat32()
	if math.Abs(float64(sig)-0.6224) > 0.05 {
		t.Errorf("Sigmoid(0.5) = %v, want ~0.6224", sig)
	}

	relu := ops.ReLU(x).ToFloat32()
	if relu != 0.5 {
		t.Errorf("ReLU(0.5) = %v, want 0.5", relu)
	}

	neg := float16.BFloat16FromFloat32(-1.0)
	reluNeg := ops.ReLU(neg).ToFloat32()
	if reluNeg != 0 {
		t.Errorf("ReLU(-1) = %v, want 0", reluNeg)
	}
}

func TestBFloat16Ops_Conversions(t *testing.T) {
	ops := BFloat16Ops{}

	one := ops.One()
	if one.ToFloat32() != 1.0 {
		t.Errorf("One() = %v, want 1.0", one.ToFloat32())
	}

	f32 := ops.FromFloat32(3.14)
	if math.Abs(float64(f32.ToFloat32()-3.14)) > 0.02 {
		t.Errorf("FromFloat32(3.14) = %v", f32.ToFloat32())
	}

	f64 := ops.FromFloat64(2.71)
	if math.Abs(float64(f64.ToFloat32())-2.71) > 0.02 {
		t.Errorf("FromFloat64(2.71) = %v", f64.ToFloat32())
	}

	zero := float16.BFloat16FromFloat32(0)
	if !ops.IsZero(zero) {
		t.Error("IsZero(0) = false")
	}
	if ops.IsZero(one) {
		t.Error("IsZero(1) = true")
	}
}

func TestBFloat16Ops_Math(t *testing.T) {
	ops := BFloat16Ops{}
	x := float16.BFloat16FromFloat32(2.0)

	exp := ops.Exp(x).ToFloat32()
	if math.Abs(float64(exp)-math.E*math.E) > 0.5 {
		t.Errorf("Exp(2) = %v, want ~%v", exp, math.E*math.E)
	}

	log := ops.Log(x).ToFloat32()
	if math.Abs(float64(log)-math.Ln2) > 0.05 {
		t.Errorf("Log(2) = %v, want ~%v", log, math.Ln2)
	}

	sqrt := ops.Sqrt(float16.BFloat16FromFloat32(4.0)).ToFloat32()
	if math.Abs(float64(sqrt-2.0)) > 0.1 {
		t.Errorf("Sqrt(4) = %v, want 2.0", sqrt)
	}

	pow := ops.Pow(float16.BFloat16FromFloat32(2.0), float16.BFloat16FromFloat32(3.0)).ToFloat32()
	if math.Abs(float64(pow-8.0)) > 0.5 {
		t.Errorf("Pow(2,3) = %v, want 8.0", pow)
	}

	abs := ops.Abs(float16.BFloat16FromFloat32(-5.0)).ToFloat32()
	if math.Abs(float64(abs-5.0)) > 0.1 {
		t.Errorf("Abs(-5) = %v, want 5.0", abs)
	}

	sum := ops.Sum([]float16.BFloat16{
		float16.BFloat16FromFloat32(1.0),
		float16.BFloat16FromFloat32(2.0),
		float16.BFloat16FromFloat32(3.0),
	}).ToFloat32()
	if math.Abs(float64(sum-6.0)) > 0.5 {
		t.Errorf("Sum([1,2,3]) = %v, want 6.0", sum)
	}

	if !ops.GreaterThan(float16.BFloat16FromFloat32(2.0), float16.BFloat16FromFloat32(1.0)) {
		t.Error("GreaterThan(2,1) = false")
	}
}

func TestBFloat16Ops_Grads(t *testing.T) {
	ops := BFloat16Ops{}
	x := float16.BFloat16FromFloat32(0.5)

	tg := ops.TanhGrad(x).ToFloat32()
	if tg <= 0 || tg > 1 {
		t.Errorf("TanhGrad(0.5) = %v, want (0,1)", tg)
	}

	sg := ops.SigmoidGrad(x).ToFloat32()
	if sg <= 0 || sg > 0.3 {
		t.Errorf("SigmoidGrad(0.5) = %v, want ~0.235", sg)
	}

	rg := ops.ReLUGrad(x).ToFloat32()
	if rg != 1.0 {
		t.Errorf("ReLUGrad(0.5) = %v, want 1.0", rg)
	}

	rgNeg := ops.ReLUGrad(float16.BFloat16FromFloat32(-1.0)).ToFloat32()
	if rgNeg != 0 {
		t.Errorf("ReLUGrad(-1) = %v, want 0", rgNeg)
	}

	lrg := ops.LeakyReLUGrad(float16.BFloat16FromFloat32(-1.0), 0.01).ToFloat32()
	if math.Abs(float64(lrg-0.01)) > 0.005 {
		t.Errorf("LeakyReLUGrad(-1, 0.01) = %v, want ~0.01", lrg)
	}

	lr := ops.LeakyReLU(float16.BFloat16FromFloat32(-2.0), 0.1).ToFloat32()
	if math.Abs(float64(lr+0.2)) > 0.05 {
		t.Errorf("LeakyReLU(-2, 0.1) = %v, want ~-0.2", lr)
	}
}

//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// AddBF16 launches the bf16 elementwise add kernel: c = a + b.
func AddBF16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("add_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchAddBF16, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "add_bf16")
}

// SubBF16 launches the bf16 elementwise subtract kernel: c = a - b.
func SubBF16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sub_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchSubBF16, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "sub_bf16")
}

// MulBF16 launches the bf16 elementwise multiply kernel: c = a * b.
func MulBF16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("mul_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchMulBF16, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "mul_bf16")
}

// DivBF16 launches the bf16 elementwise divide kernel: c = a / b.
func DivBF16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("div_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchDivBF16, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "div_bf16")
}

// TanhBF16 launches the bf16 elementwise tanh kernel (FP32 transcendental): c = tanh(a).
func TanhBF16(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("tanh_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchTanhBF16, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "tanh_bf16")
}

// SqrtBF16 launches the bf16 elementwise sqrt kernel (FP32 transcendental): c = sqrt(a).
func SqrtBF16(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sqrt_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchSqrtBF16, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "sqrt_bf16")
}

// RsqrtBF16 launches the bf16 elementwise rsqrt kernel (FP32 transcendental): c = 1/sqrt(a).
func RsqrtBF16(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("rsqrt_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchRsqrtBF16, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "rsqrt_bf16")
}

// ExpBF16 launches the bf16 elementwise exp kernel (FP32 transcendental): c = exp(a).
func ExpBF16(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("exp_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchExpBF16, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "exp_bf16")
}

// LogBF16 launches the bf16 elementwise log kernel (FP32 transcendental): c = log(a).
func LogBF16(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("log_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchLogBF16, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "log_bf16")
}

// F32ToBF16 converts n float32 elements to bf16 on GPU.
func F32ToBF16(src, dst unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("f32_to_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchF32ToBF16, uintptr(src), uintptr(dst), uintptr(n), uintptr(s))
	return checkKernel(ret, "f32_to_bf16")
}

// BF16ToF32 converts n bf16 elements to float32 on GPU.
func BF16ToF32(src, dst unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("bf16_to_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchBF16ToF32, uintptr(src), uintptr(dst), uintptr(n), uintptr(s))
	return checkKernel(ret, "bf16_to_f32")
}

// ScaledSoftmaxBF16 applies fused scaled softmax on bf16 data with FP32 accumulation.
func ScaledSoftmaxBF16(
	input, output unsafe.Pointer,
	outer, inner, axisSize int,
	scale float32,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("scaled_softmax_bf16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchScaledSoftmaxBF16,
		uintptr(input), uintptr(output),
		uintptr(outer), uintptr(inner), uintptr(axisSize),
		floatBits(scale),
		uintptr(stream))
	return checkKernel(ret, "scaled_softmax_bf16")
}

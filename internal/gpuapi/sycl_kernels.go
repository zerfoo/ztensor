package gpuapi

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/sycl"
)

// SYCLKernels implements the KernelRunner interface for SYCL devices.
// Operations will be implemented incrementally as SYCL kernel support
// is developed.
type SYCLKernels struct{}

// NewSYCLKernels returns a new SYCL kernel runner.
func NewSYCLKernels() *SYCLKernels {
	return &SYCLKernels{}
}

func (k *SYCLKernels) Add(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Add: not yet implemented for SYCL")
}

func (k *SYCLKernels) Sub(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Sub: not yet implemented for SYCL")
}

func (k *SYCLKernels) Mul(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Mul: not yet implemented for SYCL")
}

func (k *SYCLKernels) Div(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Div: not yet implemented for SYCL")
}

func (k *SYCLKernels) Pow(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Pow: not yet implemented for SYCL")
}

func (k *SYCLKernels) Exp(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Exp: not yet implemented for SYCL")
}

func (k *SYCLKernels) Log(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Log: not yet implemented for SYCL")
}

func (k *SYCLKernels) Sqrt(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Sqrt: not yet implemented for SYCL")
}

func (k *SYCLKernels) Rsqrt(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Rsqrt: not yet implemented for SYCL")
}

func (k *SYCLKernels) Sin(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Sin: not yet implemented for SYCL")
}

func (k *SYCLKernels) Cos(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Cos: not yet implemented for SYCL")
}

func (k *SYCLKernels) Tanh(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Tanh: not yet implemented for SYCL")
}

func (k *SYCLKernels) TanhPrime(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("TanhPrime: not yet implemented for SYCL")
}

func (k *SYCLKernels) AddScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("AddScalar: not yet implemented for SYCL")
}

func (k *SYCLKernels) MulScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("MulScalar: not yet implemented for SYCL")
}

func (k *SYCLKernels) DivScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("DivScalar: not yet implemented for SYCL")
}

func (k *SYCLKernels) SubScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("SubScalar: not yet implemented for SYCL")
}

func (k *SYCLKernels) PowScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("PowScalar: not yet implemented for SYCL")
}

func (k *SYCLKernels) Fill(_ unsafe.Pointer, _ float32, _ int, _ Stream) error {
	return fmt.Errorf("Fill: not yet implemented for SYCL")
}

func (k *SYCLKernels) SumAxis(_, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("SumAxis: not yet implemented for SYCL")
}

func (k *SYCLKernels) Softmax(_, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("Softmax: not yet implemented for SYCL")
}

func (k *SYCLKernels) GemmQ4F32(_, _, _ unsafe.Pointer, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("GemmQ4F32: not implemented for SYCL")
}

func (k *SYCLKernels) GemvQ4KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ4KF32: not implemented for SYCL")
}

func (k *SYCLKernels) GemvQ5KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ5KF32: not implemented for SYCL")
}

func (k *SYCLKernels) GemvQ6KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ6KF32: not implemented for SYCL")
}

func (k *SYCLKernels) GemvQ5_0F32(_, _, _ unsafe.Pointer, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ5_0F32: not implemented for SYCL")
}

func (k *SYCLKernels) DequantQ4KF32(_, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("DequantQ4KF32: not implemented for SYCL")
}

func (k *SYCLKernels) GemmQ8F32(_, _, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("GemmQ8F32: not implemented for SYCL")
}

func (k *SYCLKernels) AddBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("AddBroadcast: not implemented for SYCL")
}

func (k *SYCLKernels) SubBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("SubBroadcast: not implemented for SYCL")
}

func (k *SYCLKernels) MulBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("MulBroadcast: not implemented for SYCL")
}

func (k *SYCLKernels) DivBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("DivBroadcast: not implemented for SYCL")
}

func (k *SYCLKernels) AddBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("AddBroadcast4D: not implemented for SYCL")
}

func (k *SYCLKernels) SubBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("SubBroadcast4D: not implemented for SYCL")
}

func (k *SYCLKernels) MulBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("MulBroadcast4D: not implemented for SYCL")
}

func (k *SYCLKernels) DivBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("DivBroadcast4D: not implemented for SYCL")
}

func (k *SYCLKernels) Transpose2D(_, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("Transpose2D: not implemented for SYCL")
}

func (k *SYCLKernels) TransposeND(_, _ unsafe.Pointer, _, _, _ []int32, _, _ int, _ Stream) error {
	return fmt.Errorf("TransposeND: not implemented for SYCL")
}

func (k *SYCLKernels) Gather(_, _, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("Gather: not implemented for SYCL")
}

func (k *SYCLKernels) RMSNorm(_, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("RMSNorm: not implemented for SYCL")
}

func (k *SYCLKernels) Repeat(_, _ unsafe.Pointer, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("Repeat: not implemented for SYCL")
}

func (k *SYCLKernels) RepeatInterleaveF32(_, _ unsafe.Pointer, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("RepeatInterleaveF32: not implemented for SYCL")
}

func (k *SYCLKernels) Argmax(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Argmax: not implemented for SYCL")
}

func (k *SYCLKernels) FusedRoPEF32(_, _, _, _ unsafe.Pointer, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedRoPEF32: not implemented for SYCL")
}

func (k *SYCLKernels) FusedSwiGLUF32(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("FusedSwiGLUF32: not implemented for SYCL")
}

func (k *SYCLKernels) FusedAddRMSNormF32(_, _, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedAddRMSNormF32: not implemented for SYCL")
}

func (k *SYCLKernels) FusedNormAddF32(_, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedNormAddF32: not implemented for SYCL")
}

func (k *SYCLKernels) FusedQKNormRoPEF32(_, _, _, _, _, _ unsafe.Pointer, _ float32, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedQKNormRoPEF32: not implemented for SYCL")
}

func (k *SYCLKernels) FusedEncoderFwdF32(_ unsafe.Pointer, _, _ *[16]unsafe.Pointer, _, _ unsafe.Pointer, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedEncoderFwdF32: not implemented for SYCL")
}

func (k *SYCLKernels) FusedEncoderBwdF32(_ unsafe.Pointer, _, _ *[16]unsafe.Pointer, _ *[16]unsafe.Pointer, _ *[15]unsafe.Pointer, _ *[16]unsafe.Pointer, _, _, _ unsafe.Pointer, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedEncoderBwdF32: not implemented for SYCL")
}

func (k *SYCLKernels) FusedEncoderFwdAvailable() bool {
	return false
}

func (k *SYCLKernels) ScaledSoftmaxF32(input, output unsafe.Pointer, outer, inner, axisSize int, scale float32, s Stream) error {
	if !sycl.ScaledSoftmaxF32Available() {
		return fmt.Errorf("ScaledSoftmaxF32: SYCL kernel not available")
	}
	return sycl.ScaledSoftmaxF32(input, output, outer, inner, axisSize, scale, streamPtr(s))
}

func (k *SYCLKernels) AddFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("AddFP16: not implemented for SYCL")
}

func (k *SYCLKernels) SubFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("SubFP16: not implemented for SYCL")
}

func (k *SYCLKernels) MulFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("MulFP16: not implemented for SYCL")
}

func (k *SYCLKernels) DivFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("DivFP16: not implemented for SYCL")
}

func (k *SYCLKernels) RMSNormFP16(_, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("RMSNormFP16: not implemented for SYCL")
}

func (k *SYCLKernels) ScaledSoftmaxFP16(_, _ unsafe.Pointer, _, _, _ int, _ float32, _ Stream) error {
	return fmt.Errorf("ScaledSoftmaxFP16: not implemented for SYCL")
}

func (k *SYCLKernels) F32ToFP16(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("F32ToFP16: not implemented for SYCL")
}

func (k *SYCLKernels) FP16ToF32(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("FP16ToF32: not implemented for SYCL")
}

func (k *SYCLKernels) DequantFP8E4M3ToFP16(_, _ unsafe.Pointer, _ float32, _ int, _ Stream) error {
	return fmt.Errorf("DequantFP8E4M3ToFP16: not implemented for SYCL")
}

func (k *SYCLKernels) FP8Gemm(_, _, _ unsafe.Pointer, _, _, _ int, _, _ float32, _ Stream) error {
	return fmt.Errorf("FP8Gemm: not implemented for SYCL")
}

func (k *SYCLKernels) IsFP8GemmSupported() bool {
	return false
}

func (k *SYCLKernels) IncrementCounter(_ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("IncrementCounter: not implemented for SYCL")
}

func (k *SYCLKernels) ResetCounter(_ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("ResetCounter: not implemented for SYCL")
}

func (k *SYCLKernels) OffsetMemcpy(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("OffsetMemcpy: not implemented for SYCL")
}

func (k *SYCLKernels) OffsetMemcpyFP16(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("OffsetMemcpyFP16: not implemented for SYCL")
}

func (k *SYCLKernels) RoPESelect(_, _, _, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("RoPESelect: not implemented for SYCL")
}

func (k *SYCLKernels) SgemvM1(y, A, x unsafe.Pointer, M, N int, s Stream) error {
	if !sycl.SgemvM1Available() {
		return fmt.Errorf("SgemvM1: SYCL kernel not available")
	}
	return sycl.SgemvM1(y, A, x, M, N, streamPtr(s))
}

func (k *SYCLKernels) FusedSoftmaxVMulF32(_, _, _ unsafe.Pointer, _ float32, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedSoftmaxVMulF32: not implemented for SYCL")
}

// Compile-time interface assertion.
var _ KernelRunner = (*SYCLKernels)(nil)

func (k *SYCLKernels) GatherQ8F32(_, _, _ unsafe.Pointer, _, _, _ int, _ Stream) error { return fmt.Errorf("GatherQ8F32 not implemented") }

func (k *SYCLKernels) DequantQ5KF32(_, _ unsafe.Pointer, _, _ int, _ Stream) error { return fmt.Errorf("not implemented") }
func (k *SYCLKernels) DequantQ6KF32(_, _ unsafe.Pointer, _, _ int, _ Stream) error { return fmt.Errorf("not implemented") }
func (k *SYCLKernels) DequantQ5_0F32(_, _ unsafe.Pointer, _, _ int, _ Stream) error { return fmt.Errorf("not implemented") }

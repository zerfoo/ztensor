package gpuapi

import (
	"fmt"
	"unsafe"
)

// MetalKernels implements the KernelRunner interface using Metal compute shaders.
// Advanced operations return "not implemented" and will be added incrementally.
type MetalKernels struct{}

// NewMetalKernels returns a new Metal kernel runner.
func NewMetalKernels() *MetalKernels {
	return &MetalKernels{}
}

func (k *MetalKernels) Add(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Add: not yet implemented for Metal")
}

func (k *MetalKernels) Sub(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Sub: not yet implemented for Metal")
}

func (k *MetalKernels) Mul(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Mul: not yet implemented for Metal")
}

func (k *MetalKernels) Div(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Div: not yet implemented for Metal")
}

func (k *MetalKernels) Pow(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Pow: not yet implemented for Metal")
}

func (k *MetalKernels) Exp(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Exp: not yet implemented for Metal")
}

func (k *MetalKernels) Log(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Log: not yet implemented for Metal")
}

func (k *MetalKernels) Sqrt(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Sqrt: not yet implemented for Metal")
}

func (k *MetalKernels) Rsqrt(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Rsqrt: not yet implemented for Metal")
}

func (k *MetalKernels) Sin(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Sin: not yet implemented for Metal")
}

func (k *MetalKernels) Cos(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Cos: not yet implemented for Metal")
}

func (k *MetalKernels) Tanh(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Tanh: not yet implemented for Metal")
}

func (k *MetalKernels) TanhPrime(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("TanhPrime: not yet implemented for Metal")
}

func (k *MetalKernels) AddScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("AddScalar: not yet implemented for Metal")
}

func (k *MetalKernels) MulScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("MulScalar: not yet implemented for Metal")
}

func (k *MetalKernels) DivScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("DivScalar: not yet implemented for Metal")
}

func (k *MetalKernels) SubScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("SubScalar: not yet implemented for Metal")
}

func (k *MetalKernels) PowScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("PowScalar: not yet implemented for Metal")
}

func (k *MetalKernels) Fill(_ unsafe.Pointer, _ float32, _ int, _ Stream) error {
	return fmt.Errorf("Fill: not yet implemented for Metal")
}

func (k *MetalKernels) SumAxis(_, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("SumAxis: not yet implemented for Metal")
}

func (k *MetalKernels) Softmax(_, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("Softmax: not yet implemented for Metal")
}

func (k *MetalKernels) GemmQ4F32(_, _, _ unsafe.Pointer, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("GemmQ4F32: not implemented for Metal")
}

func (k *MetalKernels) GemvQ4KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ4KF32: not implemented for Metal")
}

func (k *MetalKernels) GemvQ5KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ5KF32: not implemented for Metal")
}

func (k *MetalKernels) GemvQ6KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ6KF32: not implemented for Metal")
}

func (k *MetalKernels) GemvQ5_0F32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ5_0F32: not implemented for Metal")
}

func (k *MetalKernels) DequantQ4KF32(_, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("DequantQ4KF32: not implemented for Metal")
}

func (k *MetalKernels) GemmQ8F32(_, _, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("GemmQ8F32: not implemented for Metal")
}

func (k *MetalKernels) AddBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("AddBroadcast: not implemented for Metal")
}

func (k *MetalKernels) SubBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("SubBroadcast: not implemented for Metal")
}

func (k *MetalKernels) MulBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("MulBroadcast: not implemented for Metal")
}

func (k *MetalKernels) DivBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("DivBroadcast: not implemented for Metal")
}

func (k *MetalKernels) AddBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("AddBroadcast4D: not implemented for Metal")
}

func (k *MetalKernels) SubBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("SubBroadcast4D: not implemented for Metal")
}

func (k *MetalKernels) MulBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("MulBroadcast4D: not implemented for Metal")
}

func (k *MetalKernels) DivBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("DivBroadcast4D: not implemented for Metal")
}

func (k *MetalKernels) Transpose2D(_, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("Transpose2D: not implemented for Metal")
}

func (k *MetalKernels) TransposeND(_, _ unsafe.Pointer, _, _, _ []int32, _, _ int, _ Stream) error {
	return fmt.Errorf("TransposeND: not implemented for Metal")
}

func (k *MetalKernels) Gather(_, _, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("Gather: not implemented for Metal")
}

func (k *MetalKernels) RMSNorm(_, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("RMSNorm: not implemented for Metal")
}

func (k *MetalKernels) Repeat(_, _ unsafe.Pointer, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("Repeat: not implemented for Metal")
}

func (k *MetalKernels) Argmax(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Argmax: not implemented for Metal")
}

func (k *MetalKernels) FusedRoPEF32(_, _, _, _ unsafe.Pointer, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedRoPEF32: not implemented for Metal")
}

func (k *MetalKernels) FusedSwiGLUF32(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("FusedSwiGLUF32: not implemented for Metal")
}

func (k *MetalKernels) FusedAddRMSNormF32(_, _, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedAddRMSNormF32: not implemented for Metal")
}

func (k *MetalKernels) FusedNormAddF32(_, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedNormAddF32: not implemented for Metal")
}

func (k *MetalKernels) FusedQKNormRoPEF32(_, _, _, _, _, _ unsafe.Pointer, _ float32, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedQKNormRoPEF32: not implemented for Metal")
}

func (k *MetalKernels) ScaledSoftmaxF32(_, _ unsafe.Pointer, _, _, _ int, _ float32, _ Stream) error {
	return fmt.Errorf("ScaledSoftmaxF32: not implemented for Metal")
}

func (k *MetalKernels) AddFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("AddFP16: not implemented for Metal")
}

func (k *MetalKernels) SubFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("SubFP16: not implemented for Metal")
}

func (k *MetalKernels) MulFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("MulFP16: not implemented for Metal")
}

func (k *MetalKernels) DivFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("DivFP16: not implemented for Metal")
}

func (k *MetalKernels) RMSNormFP16(_, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("RMSNormFP16: not implemented for Metal")
}

func (k *MetalKernels) ScaledSoftmaxFP16(_, _ unsafe.Pointer, _, _, _ int, _ float32, _ Stream) error {
	return fmt.Errorf("ScaledSoftmaxFP16: not implemented for Metal")
}

func (k *MetalKernels) F32ToFP16(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("F32ToFP16: not implemented for Metal")
}

func (k *MetalKernels) FP16ToF32(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("FP16ToF32: not implemented for Metal")
}

func (k *MetalKernels) DequantFP8E4M3ToFP16(_, _ unsafe.Pointer, _ float32, _ int, _ Stream) error {
	return fmt.Errorf("DequantFP8E4M3ToFP16: not implemented for Metal")
}

func (k *MetalKernels) FP8Gemm(_, _, _ unsafe.Pointer, _, _, _ int, _, _ float32, _ Stream) error {
	return fmt.Errorf("FP8Gemm: not implemented for Metal")
}

func (k *MetalKernels) IsFP8GemmSupported() bool {
	return false
}

func (k *MetalKernels) IncrementCounter(_ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("IncrementCounter: not implemented for Metal")
}

func (k *MetalKernels) ResetCounter(_ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("ResetCounter: not implemented for Metal")
}

func (k *MetalKernels) OffsetMemcpy(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("OffsetMemcpy: not implemented for Metal")
}

func (k *MetalKernels) OffsetMemcpyFP16(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("OffsetMemcpyFP16: not implemented for Metal")
}

func (k *MetalKernels) RoPESelect(_, _, _, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("RoPESelect: not implemented for Metal")
}

func (k *MetalKernels) SgemvM1(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("SgemvM1: not implemented for Metal")
}

// Compile-time interface assertion.
var _ KernelRunner = (*MetalKernels)(nil)

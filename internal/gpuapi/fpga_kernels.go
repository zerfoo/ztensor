package gpuapi

import (
	"fmt"
	"unsafe"
)

// FPGAKernels implements the KernelRunner interface for FPGA accelerators.
// Basic Add operation is supported; advanced operations return "not implemented"
// and will be added incrementally as FPGA bitstreams are developed.
type FPGAKernels struct{}

// NewFPGAKernels returns a new FPGA kernel runner.
func NewFPGAKernels() *FPGAKernels {
	return &FPGAKernels{}
}

func (k *FPGAKernels) Add(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Add: not yet implemented for FPGA")
}

func (k *FPGAKernels) Sub(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Sub: not yet implemented for FPGA")
}

func (k *FPGAKernels) Mul(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Mul: not yet implemented for FPGA")
}

func (k *FPGAKernels) Div(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Div: not yet implemented for FPGA")
}

func (k *FPGAKernels) Pow(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Pow: not yet implemented for FPGA")
}

func (k *FPGAKernels) Exp(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Exp: not yet implemented for FPGA")
}

func (k *FPGAKernels) Log(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Log: not yet implemented for FPGA")
}

func (k *FPGAKernels) Sqrt(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Sqrt: not yet implemented for FPGA")
}

func (k *FPGAKernels) Rsqrt(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Rsqrt: not yet implemented for FPGA")
}

func (k *FPGAKernels) Sin(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Sin: not yet implemented for FPGA")
}

func (k *FPGAKernels) Cos(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Cos: not yet implemented for FPGA")
}

func (k *FPGAKernels) Tanh(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Tanh: not yet implemented for FPGA")
}

func (k *FPGAKernels) TanhPrime(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("TanhPrime: not yet implemented for FPGA")
}

func (k *FPGAKernels) AddScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("AddScalar: not yet implemented for FPGA")
}

func (k *FPGAKernels) MulScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("MulScalar: not yet implemented for FPGA")
}

func (k *FPGAKernels) DivScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("DivScalar: not yet implemented for FPGA")
}

func (k *FPGAKernels) SubScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("SubScalar: not yet implemented for FPGA")
}

func (k *FPGAKernels) PowScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("PowScalar: not yet implemented for FPGA")
}

func (k *FPGAKernels) Fill(_ unsafe.Pointer, _ float32, _ int, _ Stream) error {
	return fmt.Errorf("Fill: not yet implemented for FPGA")
}

func (k *FPGAKernels) SumAxis(_, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("SumAxis: not yet implemented for FPGA")
}

func (k *FPGAKernels) Softmax(_, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("Softmax: not yet implemented for FPGA")
}

func (k *FPGAKernels) GemmQ4F32(_, _, _ unsafe.Pointer, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("GemmQ4F32: not implemented for FPGA")
}

func (k *FPGAKernels) GemvQ4KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ4KF32: not implemented for FPGA")
}

func (k *FPGAKernels) GemvQ5KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ5KF32: not implemented for FPGA")
}

func (k *FPGAKernels) GemvQ6KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ6KF32: not implemented for FPGA")
}

func (k *FPGAKernels) GemvQ5_0F32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ5_0F32: not implemented for FPGA")
}

func (k *FPGAKernels) DequantQ4KF32(_, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("DequantQ4KF32: not implemented for FPGA")
}

func (k *FPGAKernels) GemmQ8F32(_, _, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("GemmQ8F32: not implemented for FPGA")
}

func (k *FPGAKernels) AddBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("AddBroadcast: not implemented for FPGA")
}

func (k *FPGAKernels) SubBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("SubBroadcast: not implemented for FPGA")
}

func (k *FPGAKernels) MulBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("MulBroadcast: not implemented for FPGA")
}

func (k *FPGAKernels) DivBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("DivBroadcast: not implemented for FPGA")
}

func (k *FPGAKernels) AddBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("AddBroadcast4D: not implemented for FPGA")
}

func (k *FPGAKernels) SubBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("SubBroadcast4D: not implemented for FPGA")
}

func (k *FPGAKernels) MulBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("MulBroadcast4D: not implemented for FPGA")
}

func (k *FPGAKernels) DivBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("DivBroadcast4D: not implemented for FPGA")
}

func (k *FPGAKernels) Transpose2D(_, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("Transpose2D: not implemented for FPGA")
}

func (k *FPGAKernels) TransposeND(_, _ unsafe.Pointer, _, _, _ []int32, _, _ int, _ Stream) error {
	return fmt.Errorf("TransposeND: not implemented for FPGA")
}

func (k *FPGAKernels) Gather(_, _, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("Gather: not implemented for FPGA")
}

func (k *FPGAKernels) RMSNorm(_, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("RMSNorm: not implemented for FPGA")
}

func (k *FPGAKernels) Repeat(_, _ unsafe.Pointer, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("Repeat: not implemented for FPGA")
}

func (k *FPGAKernels) RepeatInterleaveF32(_, _ unsafe.Pointer, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("RepeatInterleaveF32: not implemented for FPGA")
}

func (k *FPGAKernels) Argmax(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Argmax: not implemented for FPGA")
}

func (k *FPGAKernels) FusedRoPEF32(_, _, _, _ unsafe.Pointer, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedRoPEF32: not implemented for FPGA")
}

func (k *FPGAKernels) FusedSwiGLUF32(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("FusedSwiGLUF32: not implemented for FPGA")
}

func (k *FPGAKernels) FusedAddRMSNormF32(_, _, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedAddRMSNormF32: not implemented for FPGA")
}

func (k *FPGAKernels) FusedNormAddF32(_, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedNormAddF32: not implemented for FPGA")
}

func (k *FPGAKernels) FusedQKNormRoPEF32(_, _, _, _, _, _ unsafe.Pointer, _ float32, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedQKNormRoPEF32: not implemented for FPGA")
}

func (k *FPGAKernels) ScaledSoftmaxF32(_, _ unsafe.Pointer, _, _, _ int, _ float32, _ Stream) error {
	return fmt.Errorf("ScaledSoftmaxF32: not implemented for FPGA")
}

func (k *FPGAKernels) AddFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("AddFP16: not implemented for FPGA")
}

func (k *FPGAKernels) SubFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("SubFP16: not implemented for FPGA")
}

func (k *FPGAKernels) MulFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("MulFP16: not implemented for FPGA")
}

func (k *FPGAKernels) DivFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("DivFP16: not implemented for FPGA")
}

func (k *FPGAKernels) RMSNormFP16(_, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("RMSNormFP16: not implemented for FPGA")
}

func (k *FPGAKernels) ScaledSoftmaxFP16(_, _ unsafe.Pointer, _, _, _ int, _ float32, _ Stream) error {
	return fmt.Errorf("ScaledSoftmaxFP16: not implemented for FPGA")
}

func (k *FPGAKernels) F32ToFP16(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("F32ToFP16: not implemented for FPGA")
}

func (k *FPGAKernels) FP16ToF32(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("FP16ToF32: not implemented for FPGA")
}

func (k *FPGAKernels) DequantFP8E4M3ToFP16(_, _ unsafe.Pointer, _ float32, _ int, _ Stream) error {
	return fmt.Errorf("DequantFP8E4M3ToFP16: not implemented for FPGA")
}

func (k *FPGAKernels) FP8Gemm(_, _, _ unsafe.Pointer, _, _, _ int, _, _ float32, _ Stream) error {
	return fmt.Errorf("FP8Gemm: not implemented for FPGA")
}

func (k *FPGAKernels) IsFP8GemmSupported() bool {
	return false
}

func (k *FPGAKernels) IncrementCounter(_ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("IncrementCounter: not implemented for FPGA")
}

func (k *FPGAKernels) ResetCounter(_ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("ResetCounter: not implemented for FPGA")
}

func (k *FPGAKernels) OffsetMemcpy(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("OffsetMemcpy: not implemented for FPGA")
}

func (k *FPGAKernels) OffsetMemcpyFP16(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("OffsetMemcpyFP16: not implemented for FPGA")
}

func (k *FPGAKernels) RoPESelect(_, _, _, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("RoPESelect: not implemented for FPGA")
}

func (k *FPGAKernels) SgemvM1(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("SgemvM1: not implemented for FPGA")
}

// Compile-time interface assertion.
var _ KernelRunner = (*FPGAKernels)(nil)

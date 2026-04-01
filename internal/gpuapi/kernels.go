package gpuapi

import "unsafe"

// KernelRunner abstracts GPU compute kernels for elementwise, scalar,
// reduction, and utility operations. Each vendor provides an implementation
// using its own kernel compilation toolchain (CUDA .cu, HIP .hip, OpenCL .cl).
type KernelRunner interface {
	// Binary elementwise operations: c[i] = op(a[i], b[i])
	Add(a, b, c unsafe.Pointer, n int, stream Stream) error
	Sub(a, b, c unsafe.Pointer, n int, stream Stream) error
	Mul(a, b, c unsafe.Pointer, n int, stream Stream) error
	Div(a, b, c unsafe.Pointer, n int, stream Stream) error
	Pow(base, exp, c unsafe.Pointer, n int, stream Stream) error

	// Unary elementwise operations: c[i] = op(a[i])
	Exp(a, c unsafe.Pointer, n int, stream Stream) error
	Log(a, c unsafe.Pointer, n int, stream Stream) error
	Sqrt(a, c unsafe.Pointer, n int, stream Stream) error
	Rsqrt(a, c unsafe.Pointer, n int, stream Stream) error
	Sin(a, c unsafe.Pointer, n int, stream Stream) error
	Cos(a, c unsafe.Pointer, n int, stream Stream) error
	Tanh(a, c unsafe.Pointer, n int, stream Stream) error

	// TanhPrime: c[i] = (1 - tanh(a[i])^2) * upstream[i]
	TanhPrime(a, upstream, c unsafe.Pointer, n int, stream Stream) error

	// Scalar operations: c[i] = op(a[i], scalar)
	AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error
	MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error
	DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error

	SubScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error
	PowScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error

	// Fill sets all n elements to value.
	Fill(data unsafe.Pointer, value float32, n int, stream Stream) error

	// SumAxis reduces along one axis: output[outer][inner] = sum(input[outer][k][inner], k=0..axisSize-1).
	SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, stream Stream) error

	// Softmax computes softmax along one axis.
	Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, stream Stream) error

	// GemmQ4F32 performs Q4_0 dequant-GEMM: C = dequant(A_q4) * B.
	// A_q4 is in GPU separated layout (scales then data), B is [K,N] float32, C is [M,N] float32.
	// dataOffset is the byte offset from A_q4 to the packed data region.
	GemmQ4F32(aQ4, b, c unsafe.Pointer, m, k, n, dataOffset int, stream Stream) error

	// GemvQ4KF32 performs Q4_K fused dequant-GEMV: y = dequant(W_q4k) * x.
	// W_q4k is raw Q4_K super-blocks for matrix [M, K]. x is [K] float32.
	// y is [M] float32. K must be a multiple of 256. Batch=1 only.
	GemvQ4KF32(wQ4K, x, y unsafe.Pointer, M, K int, stream Stream) error

	// GemvQ5KF32 performs Q5_K fused dequant-GEMV: y = dequant(W_q5k) * x.
	// W_q5k is raw Q5_K super-blocks for matrix [M, K]. x is [K] float32.
	// y is [M] float32. K must be a multiple of 256. Batch=1 only.
	GemvQ5KF32(wQ5K, x, y unsafe.Pointer, M, K int, stream Stream) error

	// GemvQ6KF32 performs Q6_K fused dequant-GEMV: y = dequant(W_q6k) * x.
	// W_q6k is raw Q6_K super-blocks for matrix [M, K]. x is [K] float32.
	// y is [M] float32. K must be a multiple of 256. Batch=1 only.
	GemvQ6KF32(wQ6K, x, y unsafe.Pointer, M, K int, stream Stream) error

	// GemvQ5_0F32 performs Q5_0 fused dequant-GEMV: y = dequant(W_q5_0) * x.
	// W_q5_0 is raw Q5_0 blocks for matrix [M, K]. x is [K] float32.
	// y is [M] float32. K must be a multiple of 32. Batch=1 only.
	GemvQ5_0F32(wQ5_0, x, y unsafe.Pointer, M, K int, stream Stream) error

	// DequantQ4KF32 dequantizes Q4_K super-blocks to FP32 in global memory.
	// src is raw Q4_K super-blocks for matrix [rows, K]. dst is [rows, K] float32.
	// K must be a multiple of 256. Used for non-GEMV cuBLAS path.
	DequantQ4KF32(src, dst unsafe.Pointer, rows, K int, stream Stream) error
	DequantQ5KF32(src, dst unsafe.Pointer, rows, K int, stream Stream) error
	DequantQ6KF32(src, dst unsafe.Pointer, rows, K int, stream Stream) error
	DequantQ5_0F32(src, dst unsafe.Pointer, rows, K int, stream Stream) error

	// GemmQ8F32 performs Q8_0 dequant-GEMM: C = dequant(A_q8) * B.
	// A_q8 is packed Q8_0 blocks (36 bytes per 32 values), B is [K,N] float32, C is [M,N] float32.
	GemmQ8F32(aQ8, b, c unsafe.Pointer, m, k, n int, stream Stream) error

	// Broadcast binary ops: c[r,c] = op(a[r*saRow+c*saCol], b[r*sbRow+c*sbCol]).
	// Strides encode broadcasting: D for full row, 1 for full col, 0 for broadcast.
	AddBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error
	SubBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error
	MulBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error
	DivBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error

	// 4D broadcast binary ops: c[i0,i1,i2,i3] = op(a[...], b[...]) with per-dim strides.
	// d0-d3 are output dims; sa0-sa3 and sb0-sb3 are per-dim strides (0 = broadcast).
	AddBroadcast4D(a, b, c unsafe.Pointer, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3 int, stream Stream) error
	SubBroadcast4D(a, b, c unsafe.Pointer, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3 int, stream Stream) error
	MulBroadcast4D(a, b, c unsafe.Pointer, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3 int, stream Stream) error
	DivBroadcast4D(a, b, c unsafe.Pointer, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3 int, stream Stream) error

	// Transpose2D transposes a [rows, cols] matrix to [cols, rows] using tiled shared memory.
	Transpose2D(input, output unsafe.Pointer, rows, cols int, stream Stream) error

	// TransposeND permutes dimensions of an N-D tensor.
	// inStrides/outStrides/perm are int32 slices on host.
	TransposeND(input, output unsafe.Pointer, inStrides, outStrides, perm []int32, ndim, total int, stream Stream) error

	// Gather performs embedding table lookup: output[i,:] = table[indices[i],:].
	// table: [V, D], indices: [N] int64 on device, output: [N, D].
	Gather(table, indices, output unsafe.Pointer, N, D, V int, stream Stream) error
	// GatherQ8F32 performs Q8_0 embedding gather: dequantizes only the requested rows.
	// q8Table: raw Q8_0 bytes on device, indices: [N] int32, output: [N, D] float32.
	GatherQ8F32(q8Table, indices, output unsafe.Pointer, N, D, V int, stream Stream) error

	// RMSNorm computes fused RMSNorm: output = input * rsqrt(mean(input^2) + eps) * weight.
	// input: [rows, D], weight: [D], output: [rows, D], scales: [rows] (per-row rsqrt values for backward).
	RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, stream Stream) error

	// Repeat replicates elements along an axis.
	// outerSize = product of dims before axis, axisDim = size of axis,
	// innerSize = product of dims after axis, reps = number of repetitions.
	Repeat(src, dst unsafe.Pointer, outerSize, axisDim, innerSize, reps int, stream Stream) error

	// RepeatInterleaveF32 expands [B, numKV, S, D] to [B, numQ, S, D] for GQA
	// key/value head expansion. Each KV head is repeated rep times along the
	// head dimension (numQ = numKV * rep). Replaces Reshape->Repeat->Reshape.
	RepeatInterleaveF32(input, output unsafe.Pointer, B, numKV, S, D, rep int, stream Stream) error

	// Argmax finds the index of the maximum element in a float32 array on device.
	// input: [n] float32, result: single int32 on device, scratch: temp storage.
	// scratch must be at least 2*ceil(n/256)*4 bytes.
	Argmax(input, result, scratch unsafe.Pointer, n int, stream Stream) error

	// FusedRoPEF32 applies rotary positional embedding in one kernel launch.
	// input/output: [batch * seqLen * headDim], cos/sin: [seqLen * cosStride].
	FusedRoPEF32(input, cosAngles, sinAngles, output unsafe.Pointer, batch, seqLen, headDim, halfRotary, cosStride int, stream Stream) error

	// FusedSwiGLUF32 applies SwiGLU activation in one kernel launch.
	// output[i] = w1[i] * sigmoid(w1[i]) * w3[i]. All arrays have n elements.
	FusedSwiGLUF32(w1, w3, output unsafe.Pointer, n int, stream Stream) error

	// FusedAddRMSNormF32 fuses residual addition and RMSNorm into one kernel launch.
	// sum_out = input + residual, normed_out = rmsnorm(sum_out, weight, eps).
	// input: [rows, D], residual: [rows, D], weight: [D],
	// normedOut: [rows, D], sumOut: [rows, D].
	FusedAddRMSNormF32(input, residual, weight, normedOut, sumOut unsafe.Pointer, eps float32, rows, D int, stream Stream) error

	// FusedNormAddF32 applies RMSNorm then adds residual in one kernel launch.
	// output = rmsnorm(input, weight, eps) + residual.
	// input: [rows, D], weight: [D], residual: [rows, D], output: [rows, D].
	FusedNormAddF32(input, weight, residual, output unsafe.Pointer, eps float32, rows, D int, stream Stream) error

	// FusedQKNormRoPEF32 applies per-head RMSNorm + RoPE to combined Q+K heads.
	// Replaces 4 kernel launches (Q_norm + K_norm + Q_RoPE + K_RoPE) with 1.
	// input: [totalHeads, headDim], weightQ/weightK: [headDim],
	// cosAngles/sinAngles: [halfRotary], output: [totalHeads, headDim].
	FusedQKNormRoPEF32(input, weightQ, weightK, cosAngles, sinAngles, output unsafe.Pointer, eps float32, totalHeads, headDim, numQHeads, halfRotary int, stream Stream) error

	// ScaledSoftmaxF32 computes softmax(input * scale) in one kernel launch,
	// replacing the MulScalar + Softmax chain (saves 1 kernel launch per call).
	ScaledSoftmaxF32(input, output unsafe.Pointer, outer, inner, axisSize int, scale float32, stream Stream) error

	// FP16 elementwise operations: inputs and outputs are __half (2 bytes each).
	AddFP16(a, b, c unsafe.Pointer, n int, stream Stream) error
	SubFP16(a, b, c unsafe.Pointer, n int, stream Stream) error
	MulFP16(a, b, c unsafe.Pointer, n int, stream Stream) error
	DivFP16(a, b, c unsafe.Pointer, n int, stream Stream) error

	// RMSNormFP16 computes RMSNorm on FP16 data with FP32 accumulation.
	RMSNormFP16(input, weight, output unsafe.Pointer, eps float32, rows, D int, stream Stream) error

	// ScaledSoftmaxFP16 computes softmax(input * scale) on FP16 data with FP32 accumulation.
	ScaledSoftmaxFP16(input, output unsafe.Pointer, outer, inner, axisSize int, scale float32, stream Stream) error

	// F32ToFP16 converts n float32 elements to FP16 on device.
	F32ToFP16(src, dst unsafe.Pointer, n int, stream Stream) error

	// FP16ToF32 converts n FP16 elements to float32 on device.
	FP16ToF32(src, dst unsafe.Pointer, n int, stream Stream) error

	// DequantFP8E4M3ToFP16 dequantizes n FP8 E4M3 bytes to FP16 on device.
	// output[i] = fp8_to_fp16(input[i]) * scale.
	DequantFP8E4M3ToFP16(input, output unsafe.Pointer, scale float32, n int, stream Stream) error

	// FP8Gemm performs FP8 E4M3 GEMM using cublasLt.
	// A: [M, K] FP8 E4M3, B: [K, N] FP8 E4M3, C: [M, N] FP16 output.
	// scaleA and scaleB are per-tensor dequantization scales.
	// Returns false for supported if the GPU does not support FP8 GEMM (sm_89+).
	FP8Gemm(a, b, c unsafe.Pointer, m, k, n int, scaleA, scaleB float32, stream Stream) error

	// IsFP8GemmSupported returns true if the GPU supports native FP8 GEMM (sm_89+).
	IsFP8GemmSupported() bool

	// GPU-resident counter operations for CUDA graph position tracking.
	IncrementCounter(counter unsafe.Pointer, delta int, stream Stream) error
	ResetCounter(counter unsafe.Pointer, value int, stream Stream) error

	// OffsetMemcpy copies dim floats from src to dst at offset counter*dim.
	// counter is a GPU-resident int32. Used for GPU-driven KV cache append.
	OffsetMemcpy(dst, src, counter unsafe.Pointer, dim, maxSeqLen int, stream Stream) error

	// OffsetMemcpyFP16 copies dim floats from F32 src to FP16 dst at offset counter*dim.
	// counter is a GPU-resident int32. Used for GPU-driven FP16 KV cache append.
	OffsetMemcpyFP16(dst, src, counter unsafe.Pointer, dim, maxSeqLen int, stream Stream) error

	// RoPESelect copies halfRotary cos/sin values from the precomputed table
	// at position counter[0]. Used for GPU-driven RoPE angle selection.
	RoPESelect(cosTable, sinTable, cosOut, sinOut, counter unsafe.Pointer, halfRotary int, stream Stream) error

	// SgemvM1 computes y = A*x for M=1 decode (single-token GEMV).
	// y[M], A[M x N] row-major, x[N].
	SgemvM1(y, A, x unsafe.Pointer, M, N int, stream Stream) error

	// FusedSoftmaxVMulF32 computes softmax(scores * scale) @ V in one kernel.
	// Decode-optimized (seqQ=1): avoids materializing the attention weights tensor.
	// scores: [BH, seqKV], V: [BH, seqKV, D], output: [BH, D].
	FusedSoftmaxVMulF32(scores, V, output unsafe.Pointer, scale float32, BH, seqKV, D int, stream Stream) error
}

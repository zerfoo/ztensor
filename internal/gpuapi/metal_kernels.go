package gpuapi

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/metal"
)

const metalThreadgroupSize = 256

// MetalKernels implements the KernelRunner interface using Metal compute shaders.
type MetalKernels struct {
	cc *metal.ComputeContext
}

// NewMetalKernels returns a new Metal kernel runner.
// If cc is nil, all operations fall back to "not implemented" errors.
func NewMetalKernels() *MetalKernels {
	return &MetalKernels{}
}

// NewMetalKernelsWithCompute returns a Metal kernel runner backed by a compute context.
func NewMetalKernelsWithCompute(cc *metal.ComputeContext) *MetalKernels {
	return &MetalKernels{cc: cc}
}

// ready returns an error if the compute context is not initialized.
func (k *MetalKernels) ready() error {
	if k.cc == nil {
		return fmt.Errorf("metal compute context not initialized")
	}
	return nil
}

// uint32Bytes encodes a uint32 as little-endian bytes.
func uint32Bytes(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}

// float32Bytes encodes a float32 as little-endian bytes.
func float32Bytes(v float32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, math.Float32bits(v))
	return b
}

// divCeil returns ceil(a/b).
func divCeil(a, b int) int {
	return (a + b - 1) / b
}

// dispatchSimple dispatches a 1D kernel with n total threads.
func (k *MetalKernels) dispatchSimple(name string, n int, buffers map[int]metal.BufferBinding, bytesArgs map[int][]byte) error {
	if err := k.ready(); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	pipeline, err := k.cc.GetPipeline(name)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	tpg := metalThreadgroupSize
	numGroups := divCeil(n, tpg)
	return k.cc.Dispatch(pipeline,
		metal.MTLSize{Width: uint64(numGroups), Height: 1, Depth: 1},
		metal.MTLSize{Width: uint64(tpg), Height: 1, Depth: 1},
		buffers, bytesArgs)
}

// dispatchPerRow dispatches a kernel with one threadgroup per row and shared memory.
// Allocates threadgroupSize * sizeof(float32) bytes of shared memory at index 0.
func (k *MetalKernels) dispatchPerRow(name string, rows int, buffers map[int]metal.BufferBinding, bytesArgs map[int][]byte) error {
	if err := k.ready(); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	pipeline, err := k.cc.GetPipeline(name)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	tpg := metalThreadgroupSize
	sharedBytes := tpg * 4 // float32 per thread
	return k.cc.Dispatch(pipeline,
		metal.MTLSize{Width: uint64(rows), Height: 1, Depth: 1},
		metal.MTLSize{Width: uint64(tpg), Height: 1, Depth: 1},
		buffers, bytesArgs, map[int]int{0: sharedBytes})
}

// --- Binary element-wise ops ---

func (k *MetalKernels) Add(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_add", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(b)}, 2: {Buffer: uintptr(c)}},
		map[int][]byte{3: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) Sub(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_sub", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(b)}, 2: {Buffer: uintptr(c)}},
		map[int][]byte{3: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) Mul(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_mul", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(b)}, 2: {Buffer: uintptr(c)}},
		map[int][]byte{3: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) Div(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_div", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(b)}, 2: {Buffer: uintptr(c)}},
		map[int][]byte{3: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) Pow(base, exp, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_pow", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(base)}, 1: {Buffer: uintptr(exp)}, 2: {Buffer: uintptr(c)}},
		map[int][]byte{3: uint32Bytes(uint32(n))})
}

// --- Unary element-wise ops ---

func (k *MetalKernels) Exp(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_exp", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) Log(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_log", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) Sqrt(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_sqrt", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) Rsqrt(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_rsqrt", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) Sin(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_sin", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) Cos(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_cos", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) Tanh(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_tanh", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) TanhPrime(a, upstream, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_tanh_prime", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(upstream)}, 2: {Buffer: uintptr(c)}},
		map[int][]byte{3: uint32Bytes(uint32(n))})
}

// --- Scalar ops ---

func (k *MetalKernels) AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_add_scalar", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: float32Bytes(scalar), 3: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) SubScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_sub_scalar", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: float32Bytes(scalar), 3: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_mul_scalar", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: float32Bytes(scalar), 3: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_div_scalar", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: float32Bytes(scalar), 3: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) PowScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_pow_scalar", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(c)}},
		map[int][]byte{2: float32Bytes(scalar), 3: uint32Bytes(uint32(n))})
}

// --- Fill ---

func (k *MetalKernels) Fill(data unsafe.Pointer, value float32, n int, _ Stream) error {
	return k.dispatchSimple("kernel_fill", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(data)}},
		map[int][]byte{1: float32Bytes(value), 2: uint32Bytes(uint32(n))})
}

// --- Reductions ---

func (k *MetalKernels) SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, _ Stream) error {
	total := outer * inner
	return k.dispatchSimple("kernel_sum_axis", total,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(input)}, 1: {Buffer: uintptr(output)}},
		map[int][]byte{2: uint32Bytes(uint32(outer)), 3: uint32Bytes(uint32(inner)), 4: uint32Bytes(uint32(axisSize))})
}

func (k *MetalKernels) Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, _ Stream) error {
	numGroups := outer * inner
	return k.dispatchPerRow("kernel_softmax", numGroups,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(input)}, 1: {Buffer: uintptr(output)}},
		map[int][]byte{2: uint32Bytes(uint32(outer)), 3: uint32Bytes(uint32(inner)), 4: uint32Bytes(uint32(axisSize))})
}

func (k *MetalKernels) ScaledSoftmaxF32(input, output unsafe.Pointer, outer, inner, axisSize int, scale float32, _ Stream) error {
	numGroups := outer * inner
	return k.dispatchPerRow("kernel_scaled_softmax", numGroups,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(input)}, 1: {Buffer: uintptr(output)}},
		map[int][]byte{
			2: uint32Bytes(uint32(outer)), 3: uint32Bytes(uint32(inner)),
			4: uint32Bytes(uint32(axisSize)), 5: float32Bytes(scale),
		})
}

// --- RMSNorm ---

func (k *MetalKernels) RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, _ Stream) error {
	return k.dispatchPerRow("kernel_rmsnorm", rows,
		map[int]metal.BufferBinding{
			0: {Buffer: uintptr(input)}, 1: {Buffer: uintptr(weight)},
			2: {Buffer: uintptr(output)}, 3: {Buffer: uintptr(scales)},
		},
		map[int][]byte{4: float32Bytes(eps), 5: uint32Bytes(uint32(D))})
}

// --- Fused ops ---

func (k *MetalKernels) FusedRoPEF32(input, cosAngles, sinAngles, output unsafe.Pointer, batch, seqLen, headDim, halfRotary, cosStride int, _ Stream) error {
	total := batch * seqLen * headDim
	b := map[int][]byte{
		4: uint32Bytes(uint32(batch)), 5: uint32Bytes(uint32(seqLen)),
		6: uint32Bytes(uint32(headDim)), 7: uint32Bytes(uint32(halfRotary)),
		8: uint32Bytes(uint32(cosStride)),
	}
	return k.dispatchSimple("kernel_fused_rope", total,
		map[int]metal.BufferBinding{
			0: {Buffer: uintptr(input)}, 1: {Buffer: uintptr(cosAngles)},
			2: {Buffer: uintptr(sinAngles)}, 3: {Buffer: uintptr(output)},
		}, b)
}

func (k *MetalKernels) FusedSwiGLUF32(w1, w3, output unsafe.Pointer, n int, _ Stream) error {
	return k.dispatchSimple("kernel_fused_swiglu", n,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(w1)}, 1: {Buffer: uintptr(w3)}, 2: {Buffer: uintptr(output)}},
		map[int][]byte{3: uint32Bytes(uint32(n))})
}

func (k *MetalKernels) FusedAddRMSNormF32(input, residual, weight, normedOut, sumOut unsafe.Pointer, eps float32, rows, D int, _ Stream) error {
	return k.dispatchPerRow("kernel_fused_add_rmsnorm", rows,
		map[int]metal.BufferBinding{
			0: {Buffer: uintptr(input)}, 1: {Buffer: uintptr(residual)},
			2: {Buffer: uintptr(weight)}, 3: {Buffer: uintptr(normedOut)},
			4: {Buffer: uintptr(sumOut)},
		},
		map[int][]byte{5: float32Bytes(eps), 6: uint32Bytes(uint32(D))})
}

func (k *MetalKernels) FusedNormAddF32(input, weight, residual, output unsafe.Pointer, eps float32, rows, D int, _ Stream) error {
	return k.dispatchPerRow("kernel_fused_norm_add", rows,
		map[int]metal.BufferBinding{
			0: {Buffer: uintptr(input)}, 1: {Buffer: uintptr(weight)},
			2: {Buffer: uintptr(residual)}, 3: {Buffer: uintptr(output)},
		},
		map[int][]byte{4: float32Bytes(eps), 5: uint32Bytes(uint32(D))})
}

func (k *MetalKernels) FusedQKNormRoPEF32(input, weightQ, weightK, cosAngles, sinAngles, output unsafe.Pointer, eps float32, totalHeads, headDim, numQHeads, halfRotary int, _ Stream) error {
	return k.dispatchPerRow("kernel_fused_qk_norm_rope", totalHeads,
		map[int]metal.BufferBinding{
			0: {Buffer: uintptr(input)}, 1: {Buffer: uintptr(weightQ)},
			2: {Buffer: uintptr(weightK)}, 3: {Buffer: uintptr(cosAngles)},
			4: {Buffer: uintptr(sinAngles)}, 5: {Buffer: uintptr(output)},
		},
		map[int][]byte{
			6: float32Bytes(eps), 7: uint32Bytes(uint32(totalHeads)),
			8: uint32Bytes(uint32(headDim)), 9: uint32Bytes(uint32(numQHeads)),
			10: uint32Bytes(uint32(halfRotary)),
		})
}

// --- GEMV ---

func (k *MetalKernels) SgemvM1(y, A, x unsafe.Pointer, M, N int, _ Stream) error {
	return k.dispatchPerRow("kernel_sgemv_m1", M,
		map[int]metal.BufferBinding{
			0: {Buffer: uintptr(y)}, 1: {Buffer: uintptr(A)}, 2: {Buffer: uintptr(x)},
		},
		map[int][]byte{3: uint32Bytes(uint32(M)), 4: uint32Bytes(uint32(N))})
}

func (k *MetalKernels) FusedSoftmaxVMulF32(_, _, _ unsafe.Pointer, _ float32, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedSoftmaxVMulF32: not yet implemented for Metal")
}

// --- Gather ---

func (k *MetalKernels) Gather(table, indices, output unsafe.Pointer, N, D, _ int, _ Stream) error {
	total := N * D
	return k.dispatchSimple("kernel_gather", total,
		map[int]metal.BufferBinding{
			0: {Buffer: uintptr(table)}, 1: {Buffer: uintptr(indices)}, 2: {Buffer: uintptr(output)},
		},
		map[int][]byte{3: uint32Bytes(uint32(N)), 4: uint32Bytes(uint32(D))})
}

// --- Transpose ---

func (k *MetalKernels) Transpose2D(input, output unsafe.Pointer, rows, cols int, _ Stream) error {
	total := rows * cols
	return k.dispatchSimple("kernel_transpose2d", total,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(input)}, 1: {Buffer: uintptr(output)}},
		map[int][]byte{2: uint32Bytes(uint32(rows)), 3: uint32Bytes(uint32(cols))})
}

func (k *MetalKernels) TransposeND(_, _ unsafe.Pointer, _, _, _ []int32, _, _ int, _ Stream) error {
	return fmt.Errorf("TransposeND: not yet implemented for Metal")
}

// --- Broadcast ops ---

func (k *MetalKernels) AddBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	total := M * D
	return k.dispatchSimple("kernel_add_broadcast", total,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(b)}, 2: {Buffer: uintptr(c)}},
		map[int][]byte{
			3: uint32Bytes(uint32(saRow)), 4: uint32Bytes(uint32(saCol)),
			5: uint32Bytes(uint32(sbRow)), 6: uint32Bytes(uint32(sbCol)),
			7: uint32Bytes(uint32(M)), 8: uint32Bytes(uint32(D)),
		})
}

func (k *MetalKernels) SubBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	total := M * D
	return k.dispatchSimple("kernel_sub_broadcast", total,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(b)}, 2: {Buffer: uintptr(c)}},
		map[int][]byte{
			3: uint32Bytes(uint32(saRow)), 4: uint32Bytes(uint32(saCol)),
			5: uint32Bytes(uint32(sbRow)), 6: uint32Bytes(uint32(sbCol)),
			7: uint32Bytes(uint32(M)), 8: uint32Bytes(uint32(D)),
		})
}

func (k *MetalKernels) MulBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	total := M * D
	return k.dispatchSimple("kernel_mul_broadcast", total,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(b)}, 2: {Buffer: uintptr(c)}},
		map[int][]byte{
			3: uint32Bytes(uint32(saRow)), 4: uint32Bytes(uint32(saCol)),
			5: uint32Bytes(uint32(sbRow)), 6: uint32Bytes(uint32(sbCol)),
			7: uint32Bytes(uint32(M)), 8: uint32Bytes(uint32(D)),
		})
}

func (k *MetalKernels) DivBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	total := M * D
	return k.dispatchSimple("kernel_div_broadcast", total,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(a)}, 1: {Buffer: uintptr(b)}, 2: {Buffer: uintptr(c)}},
		map[int][]byte{
			3: uint32Bytes(uint32(saRow)), 4: uint32Bytes(uint32(saCol)),
			5: uint32Bytes(uint32(sbRow)), 6: uint32Bytes(uint32(sbCol)),
			7: uint32Bytes(uint32(M)), 8: uint32Bytes(uint32(D)),
		})
}

// --- Repeat ---

func (k *MetalKernels) Repeat(src, dst unsafe.Pointer, outerSize, axisDim, innerSize, reps int, _ Stream) error {
	total := outerSize * axisDim * reps * innerSize
	return k.dispatchSimple("kernel_repeat", total,
		map[int]metal.BufferBinding{0: {Buffer: uintptr(src)}, 1: {Buffer: uintptr(dst)}},
		map[int][]byte{
			2: uint32Bytes(uint32(outerSize)), 3: uint32Bytes(uint32(axisDim)),
			4: uint32Bytes(uint32(innerSize)), 5: uint32Bytes(uint32(reps)),
		})
}

// --- RepeatInterleaveF32 (GQA head expansion) ---

func (k *MetalKernels) RepeatInterleaveF32(_, _ unsafe.Pointer, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("RepeatInterleaveF32: not implemented for Metal")
}

// --- Argmax (two-pass) ---

func (k *MetalKernels) Argmax(input, result, scratch unsafe.Pointer, n int, _ Stream) error {
	if err := k.ready(); err != nil {
		return fmt.Errorf("Argmax: %w", err)
	}

	tpg := metalThreadgroupSize
	nBlocks := divCeil(n, tpg)
	// scratch layout: [nBlocks float32 vals] [nBlocks int32 idxs]
	// vals at offset 0, idxs at offset nBlocks*4 in scratch buffer.

	sharedMem := map[int]int{0: tpg * 4, 1: tpg * 4} // float32 vals + int32 idxs

	// Pass 1: per-block reduction.
	p1, err := k.cc.GetPipeline("kernel_argmax_pass1")
	if err != nil {
		return fmt.Errorf("Argmax pass1: %w", err)
	}
	err = k.cc.Dispatch(p1,
		metal.MTLSize{Width: uint64(nBlocks), Height: 1, Depth: 1},
		metal.MTLSize{Width: uint64(tpg), Height: 1, Depth: 1},
		map[int]metal.BufferBinding{
			0: {Buffer: uintptr(input)},
			1: {Buffer: uintptr(scratch), Offset: 0},
			2: {Buffer: uintptr(scratch), Offset: nBlocks * 4},
		},
		map[int][]byte{3: uint32Bytes(uint32(n))},
		sharedMem)
	if err != nil {
		return err
	}

	// Pass 2: final reduction across blocks.
	p2, err := k.cc.GetPipeline("kernel_argmax_pass2")
	if err != nil {
		return fmt.Errorf("Argmax pass2: %w", err)
	}
	return k.cc.Dispatch(p2,
		metal.MTLSize{Width: 1, Height: 1, Depth: 1},
		metal.MTLSize{Width: uint64(tpg), Height: 1, Depth: 1},
		map[int]metal.BufferBinding{
			0: {Buffer: uintptr(scratch), Offset: 0},
			1: {Buffer: uintptr(scratch), Offset: nBlocks * 4},
			2: {Buffer: uintptr(result)},
		},
		map[int][]byte{3: uint32Bytes(uint32(nBlocks))},
		sharedMem)
}

// --- 4D Broadcast ops (not yet implemented) ---

func (k *MetalKernels) AddBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("AddBroadcast4D: not yet implemented for Metal")
}

func (k *MetalKernels) SubBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("SubBroadcast4D: not yet implemented for Metal")
}

func (k *MetalKernels) MulBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("MulBroadcast4D: not yet implemented for Metal")
}

func (k *MetalKernels) DivBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("DivBroadcast4D: not yet implemented for Metal")
}

// --- Quantized GEMM/GEMV (not yet implemented — requires Q4/Q8 dequant MSL) ---

func (k *MetalKernels) GemmQ4F32(_, _, _ unsafe.Pointer, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("GemmQ4F32: not yet implemented for Metal")
}

func (k *MetalKernels) GemvQ4KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ4KF32: not yet implemented for Metal")
}

func (k *MetalKernels) GemvQ5KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ5KF32: not yet implemented for Metal")
}

func (k *MetalKernels) GemvQ6KF32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ6KF32: not yet implemented for Metal")
}

func (k *MetalKernels) GemvQ5_0F32(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("GemvQ5_0F32: not yet implemented for Metal")
}

func (k *MetalKernels) DequantQ4KF32(_, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("DequantQ4KF32: not yet implemented for Metal")
}

func (k *MetalKernels) GemmQ8F32(_, _, _ unsafe.Pointer, _, _, _ int, _ Stream) error {
	return fmt.Errorf("GemmQ8F32: not yet implemented for Metal")
}

// --- FP16 ops (not yet implemented — requires half-precision MSL kernels) ---

func (k *MetalKernels) AddFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("AddFP16: not yet implemented for Metal")
}

func (k *MetalKernels) SubFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("SubFP16: not yet implemented for Metal")
}

func (k *MetalKernels) MulFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("MulFP16: not yet implemented for Metal")
}

func (k *MetalKernels) DivFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("DivFP16: not yet implemented for Metal")
}

func (k *MetalKernels) RMSNormFP16(_, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("RMSNormFP16: not yet implemented for Metal")
}

func (k *MetalKernels) ScaledSoftmaxFP16(_, _ unsafe.Pointer, _, _, _ int, _ float32, _ Stream) error {
	return fmt.Errorf("ScaledSoftmaxFP16: not yet implemented for Metal")
}

func (k *MetalKernels) F32ToFP16(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("F32ToFP16: not yet implemented for Metal")
}

func (k *MetalKernels) FP16ToF32(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("FP16ToF32: not yet implemented for Metal")
}

// --- FP8 ops (not supported on Metal) ---

func (k *MetalKernels) DequantFP8E4M3ToFP16(_, _ unsafe.Pointer, _ float32, _ int, _ Stream) error {
	return fmt.Errorf("DequantFP8E4M3ToFP16: not supported on Metal")
}

func (k *MetalKernels) FP8Gemm(_, _, _ unsafe.Pointer, _, _, _ int, _, _ float32, _ Stream) error {
	return fmt.Errorf("FP8Gemm: not supported on Metal")
}

func (k *MetalKernels) IsFP8GemmSupported() bool {
	return false
}

// --- GPU-resident counter ops (not yet implemented) ---

func (k *MetalKernels) IncrementCounter(_ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("IncrementCounter: not yet implemented for Metal")
}

func (k *MetalKernels) ResetCounter(_ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("ResetCounter: not yet implemented for Metal")
}

func (k *MetalKernels) OffsetMemcpy(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("OffsetMemcpy: not yet implemented for Metal")
}

func (k *MetalKernels) OffsetMemcpyFP16(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("OffsetMemcpyFP16: not yet implemented for Metal")
}

func (k *MetalKernels) RoPESelect(_, _, _, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("RoPESelect: not yet implemented for Metal")
}

// Compile-time interface assertion.
var _ KernelRunner = (*MetalKernels)(nil)

func (k *MetalKernels) GatherQ8F32(_, _, _ unsafe.Pointer, _, _, _ int, _ Stream) error { return fmt.Errorf("GatherQ8F32 not implemented") }

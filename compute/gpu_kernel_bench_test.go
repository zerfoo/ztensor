package compute

// gpu_kernel_bench_test.go: per-kernel GPU micro-benchmarks for the
// transcendental/division elementwise kernels and softmax -- exactly the ops
// whose codegen changed when the global --use_fast_math NVCC flag was removed
// (zerfoo plan-gpu-training-hardening T3.1). Run before/after a kernel
// rebuild on the GB10 to record the perf delta the plan requires:
//
//	go test -run '^$' -bench BenchmarkGPUKernel -benchtime 100x ./compute
//
// The measured path is engine-level (H2D upload + kernel + arena bookkeeping),
// the same for both kernel builds, so the delta isolates the kernel change.
// Skips without CUDA.

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// benchN is sized so kernel time is visible over launch overhead (4M floats,
// 16 MiB) while staying far below the arena minimum.
const benchN = 4 << 20

func newBenchEngine(b *testing.B) *GPUEngine[float32] {
	b.Helper()
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = eng.Close() })
	return eng
}

// benchInput returns a deterministic input covering the ranges that matter
// for the accurate-vs-fast comparison: positive values in (lo, hi].
func benchInput(b *testing.B, n int, lo, hi float32) *tensor.TensorNumeric[float32] {
	b.Helper()
	data := make([]float32, n)
	span := hi - lo
	for i := range data {
		data[i] = lo + span*float32(i%1000)/1000.0
	}
	t, err := tensor.New[float32]([]int{n}, data)
	if err != nil {
		b.Fatal(err)
	}
	return t
}

func benchUnary(b *testing.B, lo, hi float32,
	call func(ctx context.Context, eng *GPUEngine[float32], a *tensor.TensorNumeric[float32]) error,
) {
	eng := newBenchEngine(b)
	a := benchInput(b, benchN, lo, hi)
	ctx := context.Background()
	b.SetBytes(int64(benchN * 4))
	b.ResetTimer()
	for range b.N {
		if err := call(ctx, eng, a); err != nil {
			b.Fatal(err)
		}
		// Rewind the arena so b.N iterations cannot exhaust it and silently
		// divert the op to the CPU fallback path mid-benchmark.
		eng.ResetPool()
	}
}

func BenchmarkGPUKernel_Exp(b *testing.B) {
	benchUnary(b, -4, 4, func(ctx context.Context, eng *GPUEngine[float32], a *tensor.TensorNumeric[float32]) error {
		_, err := eng.Exp(ctx, a)
		return err
	})
}

func BenchmarkGPUKernel_Log(b *testing.B) {
	benchUnary(b, 0.001, 8, func(ctx context.Context, eng *GPUEngine[float32], a *tensor.TensorNumeric[float32]) error {
		_, err := eng.Log(ctx, a)
		return err
	})
}

func BenchmarkGPUKernel_Sqrt(b *testing.B) {
	benchUnary(b, 0.001, 8, func(ctx context.Context, eng *GPUEngine[float32], a *tensor.TensorNumeric[float32]) error {
		_, err := eng.Sqrt(ctx, a)
		return err
	})
}

func BenchmarkGPUKernel_Rsqrt(b *testing.B) {
	benchUnary(b, 0.001, 8, func(ctx context.Context, eng *GPUEngine[float32], a *tensor.TensorNumeric[float32]) error {
		_, err := eng.Rsqrt(ctx, a)
		return err
	})
}

func BenchmarkGPUKernel_Sin(b *testing.B) {
	benchUnary(b, -3, 3, func(ctx context.Context, eng *GPUEngine[float32], a *tensor.TensorNumeric[float32]) error {
		_, err := eng.Sin(ctx, a)
		return err
	})
}

func BenchmarkGPUKernel_Cos(b *testing.B) {
	benchUnary(b, -3, 3, func(ctx context.Context, eng *GPUEngine[float32], a *tensor.TensorNumeric[float32]) error {
		_, err := eng.Cos(ctx, a)
		return err
	})
}

// Tanh covers the saturation-clamped tanh path (ztensor#125) that GELU's
// tanh approximation rides on.
func BenchmarkGPUKernel_Tanh(b *testing.B) {
	benchUnary(b, -30, 30, func(ctx context.Context, eng *GPUEngine[float32], a *tensor.TensorNumeric[float32]) error {
		_, err := eng.Tanh(ctx, a)
		return err
	})
}

// Div exercises the IEEE division codegen (fast-math compiled it to a
// reciprocal approximation).
func BenchmarkGPUKernel_Div(b *testing.B) {
	eng := newBenchEngine(b)
	a := benchInput(b, benchN, 0.5, 8)
	d := benchInput(b, benchN, 0.25, 4)
	ctx := context.Background()
	b.SetBytes(int64(benchN * 4))
	b.ResetTimer()
	for range b.N {
		if _, err := eng.Div(ctx, a, d); err != nil {
			b.Fatal(err)
		}
		eng.ResetPool()
	}
}

// Softmax exercises kernel_softmax, the one kernel that keeps a selective
// fast intrinsic (__expf after max-subtraction).
func BenchmarkGPUKernel_Softmax(b *testing.B) {
	eng := newBenchEngine(b)
	const rows, cols = 4096, 1024
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = -8 + 16*float32(i%997)/997.0
	}
	a, err := tensor.New[float32]([]int{rows, cols}, data)
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	b.SetBytes(int64(rows * cols * 4))
	b.ResetTimer()
	for range b.N {
		if _, err := eng.Softmax(ctx, a, 1); err != nil {
			b.Fatal(err)
		}
		eng.ResetPool()
	}
}

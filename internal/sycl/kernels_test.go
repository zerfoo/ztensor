//go:build sycl

package sycl_test

import (
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/sycl"
)

func skipUnlessSYCLKernels(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("SYCL runtime is only available on linux")
	}
	if !sycl.Available() {
		t.Skip("SYCL runtime not available")
	}
	if !sycl.KernelsAvailable() {
		t.Skip("SYCL kernel library not available")
	}
}

// cpuSgemvM1 computes y = A*x on CPU for reference.
// A is [M x N] row-major, x is [N], y is [M].
func cpuSgemvM1(A []float32, x []float32, M, N int) []float32 {
	y := make([]float32, M)
	for i := 0; i < M; i++ {
		var sum float32
		for j := 0; j < N; j++ {
			sum += A[i*N+j] * x[j]
		}
		y[i] = sum
	}
	return y
}

// cpuScaledSoftmax computes softmax(input * scale) on CPU for reference.
// input is [outer x axisSize], output is [outer x axisSize].
func cpuScaledSoftmax(input []float32, outer, axisSize int, scale float32) []float32 {
	output := make([]float32, outer*axisSize)
	for i := 0; i < outer; i++ {
		// Find max for numerical stability.
		maxVal := float32(-math.MaxFloat32)
		for j := 0; j < axisSize; j++ {
			v := input[i*axisSize+j] * scale
			if v > maxVal {
				maxVal = v
			}
		}
		// Compute exp and sum.
		var sum float32
		for j := 0; j < axisSize; j++ {
			v := float32(math.Exp(float64(input[i*axisSize+j]*scale - maxVal)))
			output[i*axisSize+j] = v
			sum += v
		}
		// Normalize.
		for j := 0; j < axisSize; j++ {
			output[i*axisSize+j] /= sum
		}
	}
	return output
}

func TestSYCLGEMV_Correctness(t *testing.T) {
	skipUnlessSYCLKernels(t)

	if !sycl.SgemvM1Available() {
		t.Skip("SYCL SgemvM1 kernel not available")
	}

	tests := []struct {
		name string
		M, N int
	}{
		{"small_4x8", 4, 8},
		{"medium_16x32", 16, 32},
		{"large_64x128", 64, 128},
		{"non_power_of_2_5x7", 5, 7},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			M, N := tt.M, tt.N

			// Prepare input data on host.
			A := make([]float32, M*N)
			x := make([]float32, N)
			for i := range A {
				A[i] = float32(i%7) * 0.1
			}
			for i := range x {
				x[i] = float32(i%5) * 0.2
			}

			// Compute CPU reference.
			expected := cpuSgemvM1(A, x, M, N)

			// Allocate device memory and copy.
			ctx, err := sycl.NewContext(0)
			if err != nil {
				t.Fatalf("NewContext: %v", err)
			}
			defer ctx.Destroy()

			aBuf := M * N * 4
			xBuf := N * 4
			yBuf := M * 4

			devA, err := ctx.Malloc(aBuf)
			if err != nil {
				t.Fatalf("Malloc A: %v", err)
			}
			defer ctx.Free(devA)

			devX, err := ctx.Malloc(xBuf)
			if err != nil {
				t.Fatalf("Malloc x: %v", err)
			}
			defer ctx.Free(devX)

			devY, err := ctx.Malloc(yBuf)
			if err != nil {
				t.Fatalf("Malloc y: %v", err)
			}
			defer ctx.Free(devY)

			if err := ctx.Memcpy(devA, unsafe.Pointer(&A[0]), aBuf, sycl.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy A: %v", err)
			}
			if err := ctx.Memcpy(devX, unsafe.Pointer(&x[0]), xBuf, sycl.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy x: %v", err)
			}

			// Run SYCL GEMV kernel.
			if err := sycl.SgemvM1(devY, devA, devX, M, N, nil); err != nil {
				t.Fatalf("SgemvM1: %v", err)
			}

			// Copy result back.
			result := make([]float32, M)
			if err := ctx.Memcpy(unsafe.Pointer(&result[0]), devY, yBuf, sycl.MemcpyDeviceToHost); err != nil {
				t.Fatalf("Memcpy result: %v", err)
			}

			// Compare.
			const tol = 1e-4
			for i := 0; i < M; i++ {
				diff := math.Abs(float64(result[i] - expected[i]))
				if diff > tol {
					t.Errorf("y[%d] = %f, want %f (diff=%e)", i, result[i], expected[i], diff)
				}
			}
		})
	}
}

func TestSYCLAttention_Correctness(t *testing.T) {
	skipUnlessSYCLKernels(t)

	if !sycl.ScaledSoftmaxF32Available() {
		t.Skip("SYCL ScaledSoftmaxF32 kernel not available")
	}

	tests := []struct {
		name     string
		outer    int
		axisSize int
		scale    float32
	}{
		{"heads_2_dim_8", 2, 8, 0.125},
		{"heads_4_dim_16", 4, 16, 0.25},
		{"heads_8_dim_64", 8, 64, 0.125},
		{"single_head_dim_32", 1, 32, 0.1767},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outer, axisSize := tt.outer, tt.axisSize
			n := outer * axisSize

			// Prepare input data on host.
			input := make([]float32, n)
			for i := range input {
				input[i] = float32(i%11)*0.3 - 1.5
			}

			// Compute CPU reference.
			expected := cpuScaledSoftmax(input, outer, axisSize, tt.scale)

			// Allocate device memory and copy.
			ctx, err := sycl.NewContext(0)
			if err != nil {
				t.Fatalf("NewContext: %v", err)
			}
			defer ctx.Destroy()

			bufSize := n * 4

			devIn, err := ctx.Malloc(bufSize)
			if err != nil {
				t.Fatalf("Malloc input: %v", err)
			}
			defer ctx.Free(devIn)

			devOut, err := ctx.Malloc(bufSize)
			if err != nil {
				t.Fatalf("Malloc output: %v", err)
			}
			defer ctx.Free(devOut)

			if err := ctx.Memcpy(devIn, unsafe.Pointer(&input[0]), bufSize, sycl.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy input: %v", err)
			}

			// Run SYCL ScaledSoftmax kernel.
			// inner=1 for contiguous row layout.
			if err := sycl.ScaledSoftmaxF32(devIn, devOut, outer, 1, axisSize, tt.scale, nil); err != nil {
				t.Fatalf("ScaledSoftmaxF32: %v", err)
			}

			// Copy result back.
			result := make([]float32, n)
			if err := ctx.Memcpy(unsafe.Pointer(&result[0]), devOut, bufSize, sycl.MemcpyDeviceToHost); err != nil {
				t.Fatalf("Memcpy result: %v", err)
			}

			// Compare.
			const tol = 1e-4
			for i := 0; i < n; i++ {
				diff := math.Abs(float64(result[i] - expected[i]))
				if diff > tol {
					t.Errorf("output[%d] = %f, want %f (diff=%e)", i, result[i], expected[i], diff)
				}
			}

			// Verify each row sums to ~1.0.
			for i := 0; i < outer; i++ {
				var sum float32
				for j := 0; j < axisSize; j++ {
					sum += result[i*axisSize+j]
				}
				if math.Abs(float64(sum-1.0)) > 1e-3 {
					t.Errorf("row %d sum = %f, want ~1.0", i, sum)
				}
			}
		})
	}
}

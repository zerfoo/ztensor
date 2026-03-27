package compute

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestTernaryGEMVGPU(t *testing.T) {
	t.Run("fallback_known_values", func(t *testing.T) {
		// 3x4 weight matrix:
		//  [ 1,  0, -1,  1]
		//  [-1,  1,  0,  0]
		//  [ 0, -1,  1, -1]
		weights := []int8{1, 0, -1, 1, -1, 1, 0, 0, 0, -1, 1, -1}
		x := []float32{2.0, 3.0, 4.0, 5.0}
		expected := []float32{3.0, 1.0, -4.0}

		ts := tensor.NewTernaryStorageFrom(weights)
		y := TernaryGEMVGPU(ts, x, 3, 4)

		for i, want := range expected {
			if y[i] != want {
				t.Fatalf("y[%d] = %f, want %f", i, y[i], want)
			}
		}
	})

	t.Run("fallback_matches_cpu", func(t *testing.T) {
		rows, cols := 128, 256
		rng := rand.New(rand.NewPCG(42, 0))

		ternaryVals := make([]int8, rows*cols)
		for i := range ternaryVals {
			ternaryVals[i] = int8(rng.IntN(3) - 1)
		}

		x := make([]float32, cols)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}

		ts := tensor.NewTernaryStorageFrom(ternaryVals)
		gpuResult := TernaryGEMVGPU(ts, x, rows, cols)
		cpuResult := TernaryGEMV(ts, x, rows, cols)

		for i := 0; i < rows; i++ {
			diff := math.Abs(float64(gpuResult[i] - cpuResult[i]))
			if diff > 1e-6 {
				t.Fatalf("row %d: gpu=%f, cpu=%f, diff=%e", i, gpuResult[i], cpuResult[i], diff)
			}
		}
	})

	t.Run("fallback_non_multiple_of_4", func(t *testing.T) {
		// 2x5 matrix — cols not a multiple of 4
		weights := []int8{1, -1, 0, 1, -1, 0, 1, -1, 0, 1}
		x := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
		expected := []float32{-2.0, 4.0}

		ts := tensor.NewTernaryStorageFrom(weights)
		y := TernaryGEMVGPU(ts, x, 2, 5)

		for i, want := range expected {
			if y[i] != want {
				t.Fatalf("y[%d] = %f, want %f", i, y[i], want)
			}
		}
	})

	t.Run("fallback_single_element", func(t *testing.T) {
		for _, w := range []int8{-1, 0, 1} {
			ts := tensor.NewTernaryStorageFrom([]int8{w})
			y := TernaryGEMVGPU(ts, []float32{7.5}, 1, 1)
			want := float32(w) * 7.5
			if y[0] != want {
				t.Fatalf("w=%d: got %f, want %f", w, y[0], want)
			}
		}
	})

	t.Run("fallback_all_zeros", func(t *testing.T) {
		rows, cols := 16, 32
		vals := make([]int8, rows*cols)
		ts := tensor.NewTernaryStorageFrom(vals)
		x := make([]float32, cols)
		for i := range x {
			x[i] = float32(i + 1)
		}
		y := TernaryGEMVGPU(ts, x, rows, cols)
		for i, v := range y {
			if v != 0 {
				t.Fatalf("y[%d] = %f, want 0", i, v)
			}
		}
	})
}

package compute

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func denseGEMV(weights []float32, x []float32, rows, cols int) []float32 {
	y := make([]float32, rows)
	for i := 0; i < rows; i++ {
		var sum float32
		for j := 0; j < cols; j++ {
			sum += weights[i*cols+j] * x[j]
		}
		y[i] = sum
	}
	return y
}

func TestTernaryGEMV(t *testing.T) {
	t.Run("known_values", func(t *testing.T) {
		// 3x4 weight matrix:
		//  [ 1,  0, -1,  1]
		//  [-1,  1,  0,  0]
		//  [ 0, -1,  1, -1]
		weights := []int8{1, 0, -1, 1, -1, 1, 0, 0, 0, -1, 1, -1}
		x := []float32{2.0, 3.0, 4.0, 5.0}
		// y[0] = 2 + 0 - 4 + 5 = 3
		// y[1] = -2 + 3 + 0 + 0 = 1
		// y[2] = 0 - 3 + 4 - 5 = -4
		expected := []float32{3.0, 1.0, -4.0}

		ts := tensor.NewTernaryStorageFrom(weights)
		y := TernaryGEMV(ts, x, 3, 4)

		for i, want := range expected {
			if y[i] != want {
				t.Fatalf("y[%d] = %f, want %f", i, y[i], want)
			}
		}
	})

	t.Run("matches_dense_float32", func(t *testing.T) {
		rows, cols := 128, 256
		rng := rand.New(rand.NewPCG(42, 0))

		ternaryVals := make([]int8, rows*cols)
		denseWeights := make([]float32, rows*cols)
		for i := range ternaryVals {
			v := int8(rng.IntN(3) - 1)
			ternaryVals[i] = v
			denseWeights[i] = float32(v)
		}

		x := make([]float32, cols)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}

		ts := tensor.NewTernaryStorageFrom(ternaryVals)
		got := TernaryGEMV(ts, x, rows, cols)
		want := denseGEMV(denseWeights, x, rows, cols)

		for i := 0; i < rows; i++ {
			diff := float64(got[i] - want[i])
			if math.Abs(diff) > 1e-6 {
				t.Fatalf("row %d: got %f, want %f, diff %e", i, got[i], want[i], diff)
			}
		}
	})

	t.Run("all_zeros", func(t *testing.T) {
		rows, cols := 16, 32
		vals := make([]int8, rows*cols) // all zero by default
		ts := tensor.NewTernaryStorageFrom(vals)
		x := make([]float32, cols)
		for i := range x {
			x[i] = float32(i + 1)
		}
		y := TernaryGEMV(ts, x, rows, cols)
		for i, v := range y {
			if v != 0 {
				t.Fatalf("y[%d] = %f, want 0", i, v)
			}
		}
	})

	t.Run("single_element", func(t *testing.T) {
		for _, w := range []int8{-1, 0, 1} {
			ts := tensor.NewTernaryStorageFrom([]int8{w})
			y := TernaryGEMV(ts, []float32{7.5}, 1, 1)
			want := float32(w) * 7.5
			if y[0] != want {
				t.Fatalf("w=%d: got %f, want %f", w, y[0], want)
			}
		}
	})

	t.Run("identity_diagonal", func(t *testing.T) {
		// NxN matrix with 1 on diagonal, 0 elsewhere — y should equal x.
		n := 33 // intentionally not a multiple of 4
		vals := make([]int8, n*n)
		for i := 0; i < n; i++ {
			vals[i*n+i] = 1
		}
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(i)*0.5 - 8.0
		}

		ts := tensor.NewTernaryStorageFrom(vals)
		y := TernaryGEMV(ts, x, n, n)

		for i := 0; i < n; i++ {
			if y[i] != x[i] {
				t.Fatalf("y[%d] = %f, want %f", i, y[i], x[i])
			}
		}
	})

	t.Run("non_multiple_of_4_cols", func(t *testing.T) {
		// 2x5 matrix — cols not a multiple of 4
		weights := []int8{1, -1, 0, 1, -1, 0, 1, -1, 0, 1}
		x := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
		// y[0] = 1 - 2 + 0 + 4 - 5 = -2
		// y[1] = 0 + 2 - 3 + 0 + 5 = 4
		expected := []float32{-2.0, 4.0}

		ts := tensor.NewTernaryStorageFrom(weights)
		y := TernaryGEMV(ts, x, 2, 5)

		for i, want := range expected {
			if y[i] != want {
				t.Fatalf("y[%d] = %f, want %f", i, y[i], want)
			}
		}
	})
}

func BenchmarkTernaryGEMV(b *testing.B) {
	for _, size := range []struct {
		rows, cols int
	}{
		{256, 256},
		{1024, 1024},
		{4096, 4096},
	} {
		rows, cols := size.rows, size.cols
		rng := rand.New(rand.NewPCG(42, 0))

		ternaryVals := make([]int8, rows*cols)
		denseWeights := make([]float32, rows*cols)
		for i := range ternaryVals {
			v := int8(rng.IntN(3) - 1)
			ternaryVals[i] = v
			denseWeights[i] = float32(v)
		}

		x := make([]float32, cols)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}

		ts := tensor.NewTernaryStorageFrom(ternaryVals)

		b.Run("ternary_"+sizeStr(rows, cols), func(b *testing.B) {
			for range b.N {
				TernaryGEMV(ts, x, rows, cols)
			}
		})

		b.Run("dense_f32_"+sizeStr(rows, cols), func(b *testing.B) {
			for range b.N {
				denseGEMV(denseWeights, x, rows, cols)
			}
		})
	}
}

func sizeStr(rows, cols int) string {
	return fmt.Sprintf("%dx%d", rows, cols)
}

package compute

import (
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// HadamardMatrix generates a normalized Walsh-Hadamard matrix of size n x n,
// where n must be a power of 2 in the range [1, 512]. The matrix is normalized
// by 1/sqrt(n) so that H * H^T = I.
func HadamardMatrix[T tensor.Float](n int) (*tensor.TensorNumeric[T], error) {
	if n <= 0 || n&(n-1) != 0 {
		return nil, fmt.Errorf("hadamard: n must be a power of 2, got %d", n)
	}
	if n > 512 {
		return nil, fmt.Errorf("hadamard: n must be <= 512, got %d", n)
	}

	// Build unnormalized Walsh-Hadamard matrix using recursive doubling.
	// H_1 = [1]
	// H_2k = [[H_k, H_k], [H_k, -H_k]]
	data := make([]T, n*n)

	// Start with H_1 = [1].
	data[0] = 1

	for size := 1; size < n; size *= 2 {
		newSize := size * 2
		// Work backwards to avoid overwriting data we still need.
		for i := newSize - 1; i >= 0; i-- {
			for j := newSize - 1; j >= 0; j-- {
				// Map (i, j) in H_2k to the corresponding element in H_k.
				si := i % size
				sj := j % size
				val := data[si*size+sj]
				// Bottom-right quadrant gets negated.
				if i >= size && j >= size {
					val = -val
				}
				data[i*newSize+j] = val
			}
		}
	}

	// Normalize by 1/sqrt(n).
	scale := T(1.0 / math.Sqrt(float64(n)))
	for i := range data {
		data[i] *= scale
	}

	return tensor.New[T]([]int{n, n}, data)
}

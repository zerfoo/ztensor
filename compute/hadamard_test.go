package compute

import (
	"math"
	"testing"
)

func TestHadamardMatrix(t *testing.T) {
	t.Run("orthogonality", func(t *testing.T) {
		sizes := []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
		for _, n := range sizes {
			t.Run("", func(t *testing.T) {
				h, err := HadamardMatrix[float64](n)
				if err != nil {
					t.Fatalf("HadamardMatrix(%d): %v", n, err)
				}

				shape := h.Shape()
				if shape[0] != n || shape[1] != n {
					t.Fatalf("expected shape [%d %d], got %v", n, n, shape)
				}

				data := h.Data()

				// Verify H * H^T = I within tolerance.
				const tol = 1e-6
				for i := 0; i < n; i++ {
					for j := 0; j < n; j++ {
						var dot float64
						for k := 0; k < n; k++ {
							dot += data[i*n+k] * data[j*n+k]
						}
						expected := 0.0
						if i == j {
							expected = 1.0
						}
						if math.Abs(dot-expected) > tol {
							t.Fatalf("(H*H^T)[%d][%d] = %v, want %v (n=%d)", i, j, dot, expected, n)
						}
					}
				}
			})
		}
	})

	t.Run("entries are pm 1/sqrt(n)", func(t *testing.T) {
		sizes := []int{4, 16, 64, 256}
		for _, n := range sizes {
			h, err := HadamardMatrix[float64](n)
			if err != nil {
				t.Fatalf("HadamardMatrix(%d): %v", n, err)
			}
			expected := 1.0 / math.Sqrt(float64(n))
			for i, v := range h.Data() {
				if math.Abs(math.Abs(v)-expected) > 1e-12 {
					t.Fatalf("entry %d = %v, want ±%v (n=%d)", i, v, expected, n)
				}
			}
		}
	})

	t.Run("float32", func(t *testing.T) {
		h, err := HadamardMatrix[float32](64)
		if err != nil {
			t.Fatal(err)
		}
		data := h.Data()
		// Verify H * H^T ≈ I with float32 tolerance.
		const tol = 1e-5
		n := 64
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				var dot float32
				for k := 0; k < n; k++ {
					dot += data[i*n+k] * data[j*n+k]
				}
				expected := float32(0)
				if i == j {
					expected = 1
				}
				if math.Abs(float64(dot-expected)) > tol {
					t.Fatalf("(H*H^T)[%d][%d] = %v, want %v", i, j, dot, expected)
				}
			}
		}
	})

	t.Run("errors", func(t *testing.T) {
		tests := []struct {
			n       int
			wantErr string
		}{
			{0, "power of 2"},
			{-1, "power of 2"},
			{3, "power of 2"},
			{7, "power of 2"},
			{1024, "<= 512"},
		}
		for _, tt := range tests {
			_, err := HadamardMatrix[float64](tt.n)
			if err == nil {
				t.Fatalf("HadamardMatrix(%d): expected error containing %q", tt.n, tt.wantErr)
			}
		}
	})
}

package tensor

import (
	"math"
	"math/rand"
	"testing"

	"github.com/zerfoo/ztensor/device"
)

func TestNF4Quantization(t *testing.T) {
	t.Run("round-trip basic", func(t *testing.T) {
		input := nf4Linspace(-1, 1, 256)
		s := NewNF4Storage(input, []int{256})

		if s.Len() != 256 {
			t.Fatalf("Len() = %d, want 256", s.Len())
		}
		if want := (256 + nf4BlockSize - 1) / nf4BlockSize; s.NumBlocks() != want {
			t.Fatalf("NumBlocks() = %d, want %d", s.NumBlocks(), want)
		}

		out := s.Dequantize()
		if len(out) != 256 {
			t.Fatalf("Dequantize() len = %d, want 256", len(out))
		}

		mse := nf4ComputeMSE(input, out)
		if mse >= 0.01 {
			t.Errorf("round-trip MSE = %f, want < 0.01", mse)
		}
	})

	t.Run("round-trip normal distribution", func(t *testing.T) {
		// NF4 is optimized for normally distributed weights.
		rng := rand.New(rand.NewSource(42))
		n := 1024
		input := make([]float32, n)
		for i := range input {
			input[i] = float32(rng.NormFloat64()) * 0.1
		}

		s := NewNF4Storage(input, []int{n})
		out := s.Dequantize()

		mse := nf4ComputeMSE(input, out)
		if mse >= 0.01 {
			t.Errorf("normal distribution round-trip MSE = %f, want < 0.01", mse)
		}
	})

	t.Run("round-trip large", func(t *testing.T) {
		// Test with > 256 blocks to exercise double quantization meta-blocks.
		rng := rand.New(rand.NewSource(99))
		n := nf4BlockSize * 300 // 300 blocks → 2 meta-blocks
		input := make([]float32, n)
		for i := range input {
			input[i] = float32(rng.NormFloat64()) * 0.5
		}

		s := NewNF4Storage(input, []int{n})

		wantMeta := (300 + 255) / 256
		if len(s.MetaScales) != wantMeta {
			t.Fatalf("MetaScales len = %d, want %d", len(s.MetaScales), wantMeta)
		}

		out := s.Dequantize()
		mse := nf4ComputeMSE(input, out)
		if mse >= 0.01 {
			t.Errorf("large round-trip MSE = %f, want < 0.01", mse)
		}
	})

	t.Run("zeros", func(t *testing.T) {
		input := make([]float32, 128)
		s := NewNF4Storage(input, []int{128})
		out := s.Dequantize()
		for i, v := range out {
			if v != 0 {
				t.Fatalf("expected zero at index %d, got %f", i, v)
			}
		}
	})

	t.Run("empty", func(t *testing.T) {
		s := &NF4Storage{}
		if err := s.Quantize(nil); err != nil {
			t.Fatal(err)
		}
		if s.Len() != 0 {
			t.Fatalf("Len() = %d, want 0", s.Len())
		}
		if out := s.Dequantize(); out != nil {
			t.Fatalf("Dequantize() = %v, want nil", out)
		}
	})

	t.Run("non-block-aligned", func(t *testing.T) {
		// Length not a multiple of block size.
		input := nf4Linspace(-0.5, 0.5, 100)
		s := NewNF4Storage(input, []int{100})

		if s.Len() != 100 {
			t.Fatalf("Len() = %d, want 100", s.Len())
		}
		out := s.Dequantize()
		if len(out) != 100 {
			t.Fatalf("Dequantize() len = %d, want 100", len(out))
		}

		mse := nf4ComputeMSE(input, out)
		if mse >= 0.01 {
			t.Errorf("non-aligned round-trip MSE = %f, want < 0.01", mse)
		}
	})

	t.Run("Storage interface", func(t *testing.T) {
		var _ Storage[float32] = (*NF4Storage)(nil)

		input := nf4Linspace(-1, 1, 64)
		s := NewNF4Storage(input, []int{64})

		if s.DeviceType() != device.CPU {
			t.Errorf("DeviceType() = %v, want CPU", s.DeviceType())
		}

		slice := s.Slice()
		if len(slice) != 64 {
			t.Fatalf("Slice() len = %d, want 64", len(slice))
		}

		if bs := s.ByteSize(); bs <= 0 {
			t.Errorf("ByteSize() = %d, want > 0", bs)
		}
	})

	t.Run("Set re-quantizes", func(t *testing.T) {
		s := NewNF4Storage(nf4Linspace(-1, 1, 64), []int{64})
		newData := nf4Linspace(0, 1, 64)
		s.Set(newData)

		out := s.Dequantize()
		mse := nf4ComputeMSE(newData, out)
		if mse >= 0.01 {
			t.Errorf("Set() re-quantize MSE = %f, want < 0.01", mse)
		}
	})

	t.Run("codebook symmetry", func(t *testing.T) {
		// Verify the codebook is sorted and has the expected endpoints.
		if nf4Codebook[0] != -1.0 {
			t.Errorf("codebook[0] = %f, want -1.0", nf4Codebook[0])
		}
		if nf4Codebook[15] != 1.0 {
			t.Errorf("codebook[15] = %f, want 1.0", nf4Codebook[15])
		}
		if nf4Codebook[7] != 0.0 {
			t.Errorf("codebook[7] = %f, want 0.0", nf4Codebook[7])
		}
		for i := 1; i < 16; i++ {
			if nf4Codebook[i] <= nf4Codebook[i-1] {
				t.Errorf("codebook not sorted: [%d]=%f <= [%d]=%f", i, nf4Codebook[i], i-1, nf4Codebook[i-1])
			}
		}
	})
}

// nf4Linspace generates n evenly spaced values in [lo, hi].
func nf4Linspace(lo, hi float32, n int) []float32 {
	out := make([]float32, n)
	if n == 1 {
		out[0] = lo
		return out
	}
	step := (hi - lo) / float32(n-1)
	for i := range n {
		out[i] = lo + step*float32(i)
	}
	return out
}

// nf4ComputeMSE computes mean squared error between two slices.
func nf4ComputeMSE(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := range n {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return float32(sum / math.Max(float64(n), 1))
}

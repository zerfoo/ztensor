package tensor

import (
	"math"
	"math/rand"
	"testing"

	"github.com/zerfoo/ztensor/device"
)

func TestNVFloat4Storage(t *testing.T) {
	t.Run("round-trip basic", func(t *testing.T) {
		input := linspace(-1, 1, 64)
		s := NewNVFloat4Storage(input, []int{64})

		if s.Len() != 64 {
			t.Fatalf("Len() = %d, want 64", s.Len())
		}
		if s.NumBlocks() != 4 {
			t.Fatalf("NumBlocks() = %d, want 4", s.NumBlocks())
		}

		out := s.Dequantize()
		if len(out) != 64 {
			t.Fatalf("Dequantize() len = %d, want 64", len(out))
		}

		mse := computeMSE(input, out)
		maxVal := absMax(input)
		relMSE := mse / (maxVal * maxVal)
		t.Logf("basic round-trip: MSE=%e, relMSE=%.4f%%", mse, relMSE*100)
		if relMSE > 0.005 {
			t.Errorf("relative MSE %.4f%% exceeds 0.5%% threshold", relMSE*100)
		}
	})

	t.Run("round-trip random float16 range", func(t *testing.T) {
		rng := rand.New(rand.NewSource(42))
		n := 1024
		input := make([]float32, n)
		for i := range input {
			// float16 range: roughly [-65504, 65504]
			input[i] = (rng.Float32()*2 - 1) * 65504
		}

		s := NewNVFloat4Storage(input, []int{n})
		out := s.Dequantize()

		mse := computeMSE(input, out)
		maxVal := absMax(input)
		relMSE := mse / (maxVal * maxVal)
		t.Logf("random float16 round-trip (n=%d): MSE=%e, relMSE=%.4f%%", n, mse, relMSE*100)
		if relMSE > 0.005 {
			t.Errorf("relative MSE %.4f%% exceeds 0.5%% threshold", relMSE*100)
		}
	})

	t.Run("zeros", func(t *testing.T) {
		input := make([]float32, 32)
		s := NewNVFloat4Storage(input, []int{32})
		out := s.Dequantize()
		for i, v := range out {
			if v != 0 {
				t.Errorf("index %d: got %v, want 0", i, v)
			}
		}
	})

	t.Run("non-block-aligned", func(t *testing.T) {
		// 17 elements: 2 blocks, second block partial
		input := linspace(-2, 2, 17)
		s := NewNVFloat4Storage(input, []int{17})
		if s.Len() != 17 {
			t.Fatalf("Len() = %d, want 17", s.Len())
		}
		if s.NumBlocks() != 2 {
			t.Fatalf("NumBlocks() = %d, want 2", s.NumBlocks())
		}
		out := s.Dequantize()
		if len(out) != 17 {
			t.Fatalf("Dequantize() len = %d, want 17", len(out))
		}
	})

	t.Run("single element", func(t *testing.T) {
		input := []float32{3.14}
		s := NewNVFloat4Storage(input, []int{1})
		if s.Len() != 1 {
			t.Fatalf("Len() = %d, want 1", s.Len())
		}
		out := s.Dequantize()
		// With only 8 representable magnitudes, we accept coarse quantization.
		diff := float32(math.Abs(float64(out[0] - input[0])))
		if diff > 1.0 {
			t.Errorf("single element: got %v, want ~%v (diff=%v)", out[0], input[0], diff)
		}
	})

	t.Run("negative values", func(t *testing.T) {
		input := []float32{-1.0, -0.5, -0.25, -2.0, -4.0, -6.0, -3.0, -1.5,
			-1.0, -0.5, -0.25, -2.0, -4.0, -6.0, -3.0, -1.5}
		s := NewNVFloat4Storage(input, []int{16})
		out := s.Dequantize()
		for i, v := range out {
			if v > 0 {
				t.Errorf("index %d: expected negative, got %v", i, v)
			}
		}
	})

	t.Run("Storage interface", func(t *testing.T) {
		input := linspace(-1, 1, 16)
		s := NewNVFloat4Storage(input, []int{16})

		if s.DeviceType() != device.CPU {
			t.Errorf("DeviceType() = %v, want CPU", s.DeviceType())
		}

		slice := s.Slice()
		if len(slice) != 16 {
			t.Fatalf("Slice() len = %d, want 16", len(slice))
		}

		if s.ByteSize() <= 0 {
			t.Errorf("ByteSize() = %d, want > 0", s.ByteSize())
		}
	})

	t.Run("Set re-quantizes", func(t *testing.T) {
		s := NewNVFloat4Storage(linspace(-1, 1, 16), []int{16})
		newData := linspace(-2, 2, 16)
		s.Set(newData)
		out := s.Dequantize()
		if len(out) != 16 {
			t.Fatalf("after Set, Dequantize() len = %d, want 16", len(out))
		}
	})

	t.Run("empty", func(t *testing.T) {
		s := &NVFloat4Storage{}
		if err := s.Quantize(nil); err != nil {
			t.Fatal(err)
		}
		if s.Len() != 0 {
			t.Errorf("Len() = %d, want 0", s.Len())
		}
		out := s.Dequantize()
		if out != nil {
			t.Errorf("Dequantize() = %v, want nil", out)
		}
	})
}

func computeMSE(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := range n {
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return float32(sum / float64(n))
}

func absMax(a []float32) float32 {
	var m float32
	for _, v := range a {
		if av := float32(math.Abs(float64(v))); av > m {
			m = av
		}
	}
	return m
}

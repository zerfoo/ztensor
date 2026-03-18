package tensor

import (
	"math"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/device"
)

func TestAWQDequant(t *testing.T) {
	tests := []struct {
		name      string
		input     []float32
		groupSize int
		maxMSE    float64
	}{
		{
			name:      "128 values group128",
			input:     linspace(-1, 1, 128),
			groupSize: 128,
			maxMSE:    0.002,
		},
		{
			name:      "256 values group128 (2 groups)",
			input:     linspace(-2, 2, 256),
			groupSize: 128,
			maxMSE:    0.002,
		},
		{
			name:      "64 values group32",
			input:     linspace(-0.5, 0.5, 64),
			groupSize: 32,
			maxMSE:    0.001,
		},
		{
			name:      "zeros",
			input:     make([]float32, 128),
			groupSize: 128,
			maxMSE:    0.0,
		},
		{
			name:      "not multiple of group size",
			input:     linspace(-1, 1, 200),
			groupSize: 128,
			maxMSE:    0.001,
		},
		{
			name:      "positive only values",
			input:     linspace(0.5, 2.0, 128),
			groupSize: 128,
			maxMSE:    0.005,
		},
		{
			name:      "negative only values",
			input:     linspace(-2.0, -0.5, 128),
			groupSize: 128,
			maxMSE:    0.005,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := QuantizeAWQ(tt.input, tt.groupSize)
			if q.Len() != len(tt.input) {
				t.Fatalf("Len() = %d, want %d", q.Len(), len(tt.input))
			}

			dst := make([]float32, q.Len())
			q.Dequantize(dst)

			// Compute MSE relative to FP16 reference.
			var mse float64
			for i, v := range dst {
				ref := float16.FromFloat32(tt.input[i]).ToFloat32()
				diff := float64(v - ref)
				mse += diff * diff
			}
			mse /= float64(len(dst))

			if mse > tt.maxMSE {
				t.Errorf("MSE = %v, want <= %v", mse, tt.maxMSE)
			}
		})
	}
}

func TestAWQDequant_KnownVectors(t *testing.T) {
	// Hand-computed test vectors.
	// Group of 8 values, groupSize=8 for easy manual verification.
	// Values: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
	// min=0, max=7, scale=7/15≈0.4667, zero=0/scale=0
	// Quantized: [0, round(1/0.4667)=round(2.14)=2, round(4.28)=4, round(6.43)=6,
	//              round(8.57)=9, round(10.72)=11, round(12.86)=13, round(15.0)=15]
	// Dequantized: [0*0.4667, 2*0.4667, 4*0.4667, 6*0.4667, 9*0.4667, 11*0.4667, 13*0.4667, 15*0.4667]
	// ≈ [0, 0.933, 1.867, 2.800, 4.200, 5.133, 6.067, 7.000]
	input := []float32{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}
	q := QuantizeAWQ(input, 8)
	dst := make([]float32, 8)
	q.Dequantize(dst)

	// Check overall accuracy (allow ~15% error from 4-bit quantization).
	maxErr := float32(1.1) // allow up to 1.1 error for 4-bit on range [0,7]
	for i, v := range dst {
		diff := float32(math.Abs(float64(v - input[i])))
		if diff > maxErr {
			t.Errorf("index %d: got %v, want ~%v (diff=%v)", i, v, input[i], diff)
		}
	}

	// First and last values should be exact (min and max map to 0 and 15).
	if math.Abs(float64(dst[0])) > 0.1 {
		t.Errorf("first value (should be ~0): got %v", dst[0])
	}
	if math.Abs(float64(dst[7]-7.0)) > 0.1 {
		t.Errorf("last value (should be ~7): got %v", dst[7])
	}
}

func TestAWQDequant_ZeroScale(t *testing.T) {
	// All identical values — range is 0, scale is 0.
	// Dequantized output should be all zeros (0 - 0) * 0 = 0.
	input := []float32{3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14}
	q := QuantizeAWQ(input, 8)
	dst := make([]float32, 8)
	q.Dequantize(dst)

	for i, v := range dst {
		if v != 0.0 {
			t.Errorf("index %d: zero-scale group got %v, want 0", i, v)
		}
	}
}

func TestAWQDequant_MaxMinInt4(t *testing.T) {
	// Test that extreme INT4 values (0 and 15) are handled.
	// Input with exactly min and max in the same group.
	input := make([]float32, 16)
	input[0] = -10.0  // min → quantized to 0
	input[15] = 10.0  // max → quantized to 15
	for i := 1; i < 15; i++ {
		input[i] = float32(i-7) * 0.1 // intermediate values
	}

	q := QuantizeAWQ(input, 16)
	dst := make([]float32, 16)
	q.Dequantize(dst)

	// Min and max should reconstruct accurately.
	if math.Abs(float64(dst[0]-(-10.0))) > 0.2 {
		t.Errorf("min value: got %v, want ~-10.0", dst[0])
	}
	if math.Abs(float64(dst[15]-10.0)) > 0.2 {
		t.Errorf("max value: got %v, want ~10.0", dst[15])
	}
}

func TestAWQStorage_RoundTrip(t *testing.T) {
	// Round-trip accuracy: quantize then dequantize, check max per-element error.
	input := linspace(-1, 1, 512)
	q := QuantizeAWQ(input, 128)
	dst := make([]float32, len(input))
	q.Dequantize(dst)

	// 4-bit quantization: max error ≤ scale/2 ≈ range/(15*2).
	// For range 2.0: max error ≈ 0.067. Allow some FP16 rounding headroom.
	maxErr := float32(0.15)
	for i, v := range dst {
		diff := float32(math.Abs(float64(v - input[i])))
		if diff > maxErr {
			t.Errorf("index %d: got %v, want ~%v (diff=%v > %v)", i, v, input[i], diff, maxErr)
		}
	}
}

func TestAWQStorage_Slice(t *testing.T) {
	input := linspace(-1, 1, 128)
	q := QuantizeAWQ(input, 128)
	s := q.Slice()
	if len(s) != 128 {
		t.Fatalf("Slice() len = %d, want 128", len(s))
	}
	for i, v := range s {
		diff := float32(math.Abs(float64(v - input[i])))
		if diff > 0.2 {
			t.Errorf("Slice[%d] = %v, want ~%v (diff=%v)", i, v, input[i], diff)
		}
	}
}

func TestAWQStorage_DeviceType(t *testing.T) {
	q := QuantizeAWQ(make([]float32, 128), 128)
	if q.DeviceType() != device.CPU {
		t.Errorf("DeviceType() = %v, want CPU", q.DeviceType())
	}
}

func TestAWQStorage_Immutable(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on Set()")
		}
	}()
	q := QuantizeAWQ(make([]float32, 128), 128)
	q.Set(make([]float32, 128))
}

func TestAWQStorage_NumGroups(t *testing.T) {
	tests := []struct {
		n         int
		groupSize int
		want      int
	}{
		{128, 128, 1},
		{256, 128, 2},
		{200, 128, 2},
		{64, 32, 2},
	}
	for _, tt := range tests {
		q := QuantizeAWQ(make([]float32, tt.n), tt.groupSize)
		if q.NumGroups() != tt.want {
			t.Errorf("n=%d groupSize=%d: NumGroups() = %d, want %d",
				tt.n, tt.groupSize, q.NumGroups(), tt.want)
		}
	}
}

func TestNewAWQStorageFromRaw(t *testing.T) {
	input := linspace(-1, 1, 256)
	orig := QuantizeAWQ(input, 128)

	scales := make([]float16.Float16, orig.NumGroups())
	zeros := make([]float16.Float16, orig.NumGroups())
	for i := range orig.NumGroups() {
		scales[i] = orig.groups[i].scale
		zeros[i] = orig.groups[i].zero
	}

	// Concatenate packed data from all groups.
	totalWords := 0
	for _, g := range orig.groups {
		totalWords += len(g.data)
	}
	packedData := make([]uint32, totalWords)
	off := 0
	for _, g := range orig.groups {
		copy(packedData[off:], g.data)
		off += len(g.data)
	}

	got, err := NewAWQStorageFromRaw(packedData, scales, zeros, 256, 128)
	if err != nil {
		t.Fatalf("NewAWQStorageFromRaw: %v", err)
	}
	if got.Len() != 256 {
		t.Fatalf("Len() = %d, want 256", got.Len())
	}

	origSlice := orig.Slice()
	gotSlice := got.Slice()
	for i := range origSlice {
		if origSlice[i] != gotSlice[i] {
			t.Errorf("index %d: got %v, want %v", i, gotSlice[i], origSlice[i])
		}
	}
}

func TestNewAWQStorageFromRaw_Errors(t *testing.T) {
	tests := []struct {
		name      string
		data      []uint32
		scales    []float16.Float16
		zeros     []float16.Float16
		n         int
		groupSize int
	}{
		{"zero elements", nil, nil, nil, 0, 128},
		{"negative elements", nil, nil, nil, -1, 128},
		{"zero groupSize", nil, nil, nil, 128, 0},
		{"scale count mismatch", make([]uint32, 16), makeFloat16s(2), makeFloat16s(1), 128, 128},
		{"data too short", make([]uint32, 2), makeFloat16s(1), makeFloat16s(1), 128, 128},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewAWQStorageFromRaw(tt.data, tt.scales, tt.zeros, tt.n, tt.groupSize)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestAWQQuantRegistry(t *testing.T) {
	d, ok := GetQuantType("AWQ_4")
	if !ok {
		t.Fatal("AWQ_4 not registered")
	}
	if d.BitsPerWeight() != 4 {
		t.Errorf("BitsPerWeight() = %d, want 4", d.BitsPerWeight())
	}
	if d.BlockSize() != 128 {
		t.Errorf("BlockSize() = %d, want 128", d.BlockSize())
	}
}

func BenchmarkDequantizeAWQ(b *testing.B) {
	input := linspace(-1, 1, 4096)
	q := QuantizeAWQ(input, 128)
	dst := make([]float32, 4096)

	b.ResetTimer()
	b.SetBytes(int64(4096 * 4))
	for range b.N {
		q.Dequantize(dst)
	}
}

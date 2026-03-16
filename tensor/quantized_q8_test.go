package tensor

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/device"
)

func TestQuantizeQ8_RoundTrip(t *testing.T) { //nolint:dupl // Q8 and Q4 round-trip tests are structurally similar but differ in tolerance and quantizer.
	tests := []struct {
		name    string
		input   []float32
		maxErr  float32
		wantLen int
	}{
		{
			name:    "32 values in [-1,1]",
			input:   linspace(-1, 1, 32),
			maxErr:  0.01,
			wantLen: 32,
		},
		{
			name:    "64 values (2 blocks)",
			input:   linspace(-0.5, 0.5, 64),
			maxErr:  0.01,
			wantLen: 64,
		},
		{
			name:    "zeros",
			input:   make([]float32, 32),
			maxErr:  0.001,
			wantLen: 32,
		},
		{
			name:    "not multiple of 32 (48 values, padded to 64)",
			input:   linspace(-1, 1, 48),
			maxErr:  0.01,
			wantLen: 48,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := QuantizeQ8(tt.input)
			if q.Len() != tt.wantLen {
				t.Fatalf("Len() = %d, want %d", q.Len(), tt.wantLen)
			}

			dst := make([]float32, tt.wantLen)
			q.Dequantize(dst)

			for i, v := range dst {
				diff := float32(math.Abs(float64(v - tt.input[i])))
				if diff > tt.maxErr {
					t.Errorf("index %d: got %v, want %v (err=%v > %v)", i, v, tt.input[i], diff, tt.maxErr)
				}
			}
		})
	}
}

func TestQ8Storage_DeviceType(t *testing.T) {
	q := QuantizeQ8(make([]float32, 32))
	if q.DeviceType() != device.CPU {
		t.Errorf("DeviceType() = %v, want CPU", q.DeviceType())
	}
}

func TestQ8Storage_Slice(t *testing.T) {
	input := linspace(-1, 1, 32)
	q := QuantizeQ8(input)
	s := q.Slice()
	if len(s) != 32 {
		t.Fatalf("Slice() len = %d, want 32", len(s))
	}
	for i, v := range s {
		diff := float32(math.Abs(float64(v - input[i])))
		if diff > 0.01 {
			t.Errorf("Slice[%d] = %v, want ~%v", i, v, input[i])
		}
	}
}

func TestQ8Storage_CompressionRatio(t *testing.T) {
	n := 1024
	input := linspace(-1, 1, n)
	q := QuantizeQ8(input)

	fp32Bytes := n * 4
	q8Bytes := q.ByteSize()

	ratio := float64(fp32Bytes) / float64(q8Bytes)
	// Q8_0: 36 bytes per 32 values, FP32: 128 bytes per 32 values. Ratio ~3.6x
	if ratio < 3.0 {
		t.Errorf("compression ratio = %.1fx, want >= 3x", ratio)
	}
}

func TestQuantizeQ8_ExtremeValues(t *testing.T) {
	input := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100}
	q := QuantizeQ8(input)
	dst := make([]float32, 32)
	q.Dequantize(dst)

	if dst[31] < 90 {
		t.Errorf("extreme value: got %v, expected something close to 100", dst[31])
	}
}

func TestQuantizeQ8_NegativeValues(t *testing.T) {
	input := make([]float32, 32)
	for i := range input {
		input[i] = -float32(i) / 31.0
	}
	q := QuantizeQ8(input)
	dst := make([]float32, 32)
	q.Dequantize(dst)

	for i, v := range dst {
		diff := float32(math.Abs(float64(v - input[i])))
		if diff > 0.01 {
			t.Errorf("index %d: got %v, want %v (err=%v)", i, v, input[i], diff)
		}
	}
}

func TestQ8Storage_BlockCount(t *testing.T) {
	tests := []struct {
		n          int
		wantBlocks int
	}{
		{32, 1},
		{64, 2},
		{48, 2}, // padded to 64
		{1, 1},  // padded to 32
	}
	for _, tt := range tests {
		q := QuantizeQ8(make([]float32, tt.n))
		if q.NumBlocks() != tt.wantBlocks {
			t.Errorf("n=%d: NumBlocks() = %d, want %d", tt.n, q.NumBlocks(), tt.wantBlocks)
		}
	}
}

func TestNewQ8StorageFromBlocks(t *testing.T) {
	// Quantize known data, extract scales and quants, reconstruct, and verify.
	input := linspace(-1, 1, 64) // 2 blocks
	orig := QuantizeQ8(input)

	// Extract block data from the original.
	nBlocks := orig.NumBlocks()
	scales := make([]float32, nBlocks)
	quants := make([]int8, nBlocks*32)
	for i := range nBlocks {
		scales[i] = orig.BlockScale(i)
		copy(quants[i*32:(i+1)*32], orig.BlockQuants(i))
	}

	got, err := NewQ8StorageFromBlocks(scales, quants, 64)
	if err != nil {
		t.Fatalf("NewQ8StorageFromBlocks: %v", err)
	}
	if got.Len() != 64 {
		t.Fatalf("Len() = %d, want 64", got.Len())
	}
	if got.NumBlocks() != 2 {
		t.Fatalf("NumBlocks() = %d, want 2", got.NumBlocks())
	}

	// Dequantize and compare with original.
	origSlice := orig.Slice()
	gotSlice := got.Slice()
	for i := range origSlice {
		if origSlice[i] != gotSlice[i] {
			t.Errorf("index %d: got %v, want %v", i, gotSlice[i], origSlice[i])
		}
	}
}

func TestNewQ8StorageFromBlocks_Errors(t *testing.T) {
	tests := []struct {
		name   string
		scales []float32
		quants []int8
		n      int
	}{
		{"mismatched lengths", []float32{1.0}, make([]int8, 64), 32},
		{"zero elements", nil, nil, 0},
		{"negative elements", nil, nil, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewQ8StorageFromBlocks(tt.scales, tt.quants, tt.n)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

func BenchmarkDequantizeQ8(b *testing.B) {
	input := linspace(-1, 1, 4096)
	q := QuantizeQ8(input)
	dst := make([]float32, 4096)

	b.ResetTimer()
	b.SetBytes(int64(4096 * 4))
	for range b.N {
		q.Dequantize(dst)
	}
}

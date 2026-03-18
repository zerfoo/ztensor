package tensor

import (
	"math"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/device"
)

func TestGPTQDequant(t *testing.T) {
	tests := []struct {
		name      string
		input     []float32
		groupSize int
		bits      int
		maxMSE    float64 // max allowed MSE relative to FP16 range
	}{
		{
			name:      "128 values 4-bit group128",
			input:     linspace(-1, 1, 128),
			groupSize: 128,
			bits:      4,
			maxMSE:    0.002,
		},
		{
			name:      "256 values 4-bit group128 (2 groups)",
			input:     linspace(-2, 2, 256),
			groupSize: 128,
			bits:      4,
			maxMSE:    0.002,
		},
		{
			name:      "128 values 8-bit group128",
			input:     linspace(-1, 1, 128),
			groupSize: 128,
			bits:      8,
			maxMSE:    0.001,
		},
		{
			name:      "64 values 4-bit group32",
			input:     linspace(-0.5, 0.5, 64),
			groupSize: 32,
			bits:      4,
			maxMSE:    0.001,
		},
		{
			name:      "zeros",
			input:     make([]float32, 128),
			groupSize: 128,
			bits:      4,
			maxMSE:    0.0,
		},
		{
			name:      "not multiple of group size",
			input:     linspace(-1, 1, 200),
			groupSize: 128,
			bits:      4,
			maxMSE:    0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := QuantizeGPTQ(tt.input, tt.groupSize, tt.bits)
			if q.Len() != len(tt.input) {
				t.Fatalf("Len() = %d, want %d", q.Len(), len(tt.input))
			}

			dst := make([]float32, q.Len())
			q.Dequantize(dst)

			// Compute MSE relative to FP16 reference.
			var mse float64
			for i, v := range dst {
				// Reference is the FP16-rounded input.
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

func TestGPTQStorage_Slice(t *testing.T) {
	input := linspace(-1, 1, 128)
	q := QuantizeGPTQ(input, 128, 4)
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

func TestGPTQStorage_DeviceType(t *testing.T) {
	q := QuantizeGPTQ(make([]float32, 128), 128, 4)
	if q.DeviceType() != device.CPU {
		t.Errorf("DeviceType() = %v, want CPU", q.DeviceType())
	}
}

func TestGPTQStorage_Immutable(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on Set()")
		}
	}()
	q := QuantizeGPTQ(make([]float32, 128), 128, 4)
	q.Set(make([]float32, 128))
}

func TestGPTQStorage_NumGroups(t *testing.T) {
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
		q := QuantizeGPTQ(make([]float32, tt.n), tt.groupSize, 4)
		if q.NumGroups() != tt.want {
			t.Errorf("n=%d groupSize=%d: NumGroups() = %d, want %d",
				tt.n, tt.groupSize, q.NumGroups(), tt.want)
		}
	}
}

func TestGPTQStorage_ByteSize(t *testing.T) {
	// 256 elements, group_size=128, 4-bit: 2 groups
	// Each group: 128 values * 4 bits / 8 = 64 bytes packed + 2 bytes scale + 2 bytes zero = 68 bytes
	// Total: 2 * 68 = 136 bytes
	q := QuantizeGPTQ(linspace(-1, 1, 256), 128, 4)
	got := q.ByteSize()
	// 2 groups * (64 packed + 2 scale + 2 zero) = 136
	if got != 136 {
		t.Errorf("ByteSize() = %d, want 136", got)
	}
}

func TestGPTQDequant_Symmetric(t *testing.T) {
	// Test symmetric quantization (zeros = 0).
	// When all values are symmetric around 0, zeros should be ~0.
	input := linspace(-1, 1, 128)
	q := QuantizeGPTQ(input, 128, 4)
	dst := make([]float32, 128)
	q.Dequantize(dst)

	// Verify sign preservation.
	for i := range dst {
		if input[i] > 0.15 && dst[i] <= 0 {
			t.Errorf("index %d: sign flip, input=%v dequant=%v", i, input[i], dst[i])
		}
		if input[i] < -0.15 && dst[i] >= 0 {
			t.Errorf("index %d: sign flip, input=%v dequant=%v", i, input[i], dst[i])
		}
	}
}

func TestNewGPTQStorageFromRaw(t *testing.T) {
	input := linspace(-1, 1, 256)
	orig := QuantizeGPTQ(input, 128, 4)

	scales := make([]float16.Float16, orig.NumGroups())
	zeros := make([]float16.Float16, orig.NumGroups())
	for i := range orig.NumGroups() {
		scales[i] = orig.groups[i].scale
		zeros[i] = orig.groups[i].zero
	}
	packedData := make([]byte, len(orig.groups[0].data)*orig.NumGroups())
	for i, g := range orig.groups {
		copy(packedData[i*len(g.data):], g.data)
	}

	got, err := NewGPTQStorageFromRaw(packedData, scales, zeros, 256, 128, 4)
	if err != nil {
		t.Fatalf("NewGPTQStorageFromRaw: %v", err)
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

func TestNewGPTQStorageFromRaw_Errors(t *testing.T) {
	tests := []struct {
		name      string
		data      []byte
		scales    []float16.Float16
		zeros     []float16.Float16
		n         int
		groupSize int
		bits      int
	}{
		{"zero elements", nil, nil, nil, 0, 128, 4},
		{"negative elements", nil, nil, nil, -1, 128, 4},
		{"invalid bits", make([]byte, 64), makeFloat16s(1), makeFloat16s(1), 128, 128, 3},
		{"scale count mismatch", make([]byte, 64), makeFloat16s(2), makeFloat16s(1), 128, 128, 4},
		{"data too short", make([]byte, 10), makeFloat16s(1), makeFloat16s(1), 128, 128, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewGPTQStorageFromRaw(tt.data, tt.scales, tt.zeros, tt.n, tt.groupSize, tt.bits)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestGPTQDequant_8Bit(t *testing.T) {
	input := linspace(-2, 2, 256)
	q := QuantizeGPTQ(input, 128, 8)

	dst := make([]float32, 256)
	q.Dequantize(dst)

	var mse float64
	for i, v := range dst {
		ref := float16.FromFloat32(input[i]).ToFloat32()
		diff := float64(v - ref)
		mse += diff * diff
	}
	mse /= float64(len(dst))

	if mse > 0.001 {
		t.Errorf("8-bit MSE = %v, want <= 0.001", mse)
	}
}

func makeFloat16s(n int) []float16.Float16 {
	s := make([]float16.Float16, n)
	for i := range s {
		s[i] = float16.FromFloat32(0)
	}
	return s
}

func BenchmarkDequantizeGPTQ(b *testing.B) {
	input := linspace(-1, 1, 4096)
	q := QuantizeGPTQ(input, 128, 4)
	dst := make([]float32, 4096)

	b.ResetTimer()
	b.SetBytes(int64(4096 * 4))
	for range b.N {
		q.Dequantize(dst)
	}
}

package tensor

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/device"
)

func TestW8A8Dequant(t *testing.T) {
	tests := []struct {
		name   string
		input  []float32
		maxMSE float64
	}{
		{
			name:   "32 values (1 group)",
			input:  linspace(-1, 1, 32),
			maxMSE: 0.0005,
		},
		{
			name:   "64 values (2 groups)",
			input:  linspace(-2, 2, 64),
			maxMSE: 0.001,
		},
		{
			name:   "256 values",
			input:  linspace(-1, 1, 256),
			maxMSE: 0.0005,
		},
		{
			name:   "zeros",
			input:  make([]float32, 64),
			maxMSE: 0.0,
		},
		{
			name:   "not multiple of group size",
			input:  linspace(-1, 1, 50),
			maxMSE: 0.001,
		},
		{
			name:   "positive only",
			input:  linspace(0.5, 2.0, 64),
			maxMSE: 0.001,
		},
		{
			name:   "negative only",
			input:  linspace(-2.0, -0.5, 64),
			maxMSE: 0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := QuantizeW8A8(tt.input)
			if q.Len() != len(tt.input) {
				t.Fatalf("Len() = %d, want %d", q.Len(), len(tt.input))
			}

			dst := make([]float32, q.Len())
			q.Dequantize(dst)

			var mse float64
			for i, v := range dst {
				diff := float64(v - tt.input[i])
				mse += diff * diff
			}
			mse /= float64(len(dst))

			if mse > tt.maxMSE {
				t.Errorf("MSE = %v, want <= %v", mse, tt.maxMSE)
			}
		})
	}
}

func TestW8A8Dequant_SymmetricRange(t *testing.T) {
	input := []float32{-1.0, 0.0, 1.0}
	padded := make([]float32, 32)
	copy(padded, input)

	q := QuantizeW8A8(padded)
	dst := make([]float32, 32)
	q.Dequantize(dst)

	if math.Abs(float64(dst[0]-(-1.0))) > 0.01 {
		t.Errorf("min value: got %v, want ~-1.0", dst[0])
	}
	if math.Abs(float64(dst[1])) > 0.01 {
		t.Errorf("zero value: got %v, want ~0.0", dst[1])
	}
	if math.Abs(float64(dst[2]-1.0)) > 0.01 {
		t.Errorf("max value: got %v, want ~1.0", dst[2])
	}
}

func TestW8A8Storage_Slice(t *testing.T) {
	input := linspace(-1, 1, 64)
	q := QuantizeW8A8(input)
	s := q.Slice()
	if len(s) != 64 {
		t.Fatalf("Slice() len = %d, want 64", len(s))
	}
	for i, v := range s {
		diff := float32(math.Abs(float64(v - input[i])))
		if diff > 0.05 {
			t.Errorf("Slice[%d] = %v, want ~%v (diff=%v)", i, v, input[i], diff)
		}
	}
}

func TestW8A8Storage_DeviceType(t *testing.T) {
	q := QuantizeW8A8(make([]float32, 32))
	if q.DeviceType() != device.CPU {
		t.Errorf("DeviceType() = %v, want CPU", q.DeviceType())
	}
}

func TestW8A8Storage_Immutable(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on Set()")
		}
	}()
	q := QuantizeW8A8(make([]float32, 32))
	q.Set(make([]float32, 32))
}

func TestW8A8Storage_NumGroups(t *testing.T) {
	tests := []struct {
		n    int
		want int
	}{
		{32, 1},
		{64, 2},
		{50, 2},
		{256, 8},
	}
	for _, tt := range tests {
		q := QuantizeW8A8(make([]float32, tt.n))
		if q.NumGroups() != tt.want {
			t.Errorf("n=%d: NumGroups() = %d, want %d", tt.n, q.NumGroups(), tt.want)
		}
	}
}

func TestW8A8Storage_GroupSize(t *testing.T) {
	q := QuantizeW8A8(make([]float32, 32))
	if q.GroupSize() != 32 {
		t.Errorf("GroupSize() = %d, want 32", q.GroupSize())
	}
}

func TestW8A8Storage_ByteSize(t *testing.T) {
	q := QuantizeW8A8(make([]float32, 64))
	if q.ByteSize() != 72 {
		t.Errorf("ByteSize() = %d, want 72", q.ByteSize())
	}
}

func TestW8A8Storage_RawBytes(t *testing.T) {
	input := linspace(-1, 1, 32)
	q := QuantizeW8A8(input)
	raw := q.RawBytes()
	if len(raw) != 36 {
		t.Fatalf("RawBytes() len = %d, want 36", len(raw))
	}

	dst := make([]float32, 32)
	d := w8a8Dequantizer{}
	if err := d.Dequantize(raw, dst); err != nil {
		t.Fatalf("Dequantize from raw: %v", err)
	}

	orig := q.Slice()
	for i := range orig {
		if orig[i] != dst[i] {
			t.Errorf("index %d: raw round-trip got %v, want %v", i, dst[i], orig[i])
		}
	}
}

func TestNewW8A8StorageFromBlocks(t *testing.T) {
	input := linspace(-1, 1, 64)
	orig := QuantizeW8A8(input)

	scales := make([]float32, orig.NumGroups())
	quants := make([]int8, orig.NumGroups()*32)
	for i := range orig.NumGroups() {
		scales[i] = orig.BlockScale(i)
		copy(quants[i*32:(i+1)*32], orig.BlockQuants(i))
	}

	got, err := NewW8A8StorageFromBlocks(scales, quants, 64)
	if err != nil {
		t.Fatalf("NewW8A8StorageFromBlocks: %v", err)
	}
	if got.Len() != 64 {
		t.Fatalf("Len() = %d, want 64", got.Len())
	}

	origSlice := orig.Slice()
	gotSlice := got.Slice()
	for i := range origSlice {
		if origSlice[i] != gotSlice[i] {
			t.Errorf("index %d: got %v, want %v", i, gotSlice[i], origSlice[i])
		}
	}
}

func TestNewW8A8StorageFromBlocks_Errors(t *testing.T) {
	tests := []struct {
		name   string
		scales []float32
		quants []int8
		n      int
	}{
		{"zero elements", nil, nil, 0},
		{"negative elements", nil, nil, -1},
		{"scale count mismatch", []float32{1, 2}, make([]int8, 32), 32},
		{"quant count mismatch", []float32{1}, make([]int8, 16), 32},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewW8A8StorageFromBlocks(tt.scales, tt.quants, tt.n)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestW8A8QuantRegistry(t *testing.T) {
	d, ok := GetQuantType("W8A8")
	if !ok {
		t.Fatal("W8A8 not registered")
	}
	if d.BitsPerWeight() != 8 {
		t.Errorf("BitsPerWeight() = %d, want 8", d.BitsPerWeight())
	}
	if d.BlockSize() != 32 {
		t.Errorf("BlockSize() = %d, want 32", d.BlockSize())
	}
}

func TestW8A8_DequantizeBlock(t *testing.T) {
	input := linspace(-2, 2, 64)
	q := QuantizeW8A8(input)

	fullDequant := make([]float32, 64)
	q.Dequantize(fullDequant)

	var buf [32]float32
	for gi := range q.NumGroups() {
		q.DequantizeBlock(gi, &buf)
		offset := gi * 32
		for j := range 32 {
			if offset+j < 64 && buf[j] != fullDequant[offset+j] {
				t.Errorf("group %d, index %d: DequantizeBlock=%v, Dequantize=%v",
					gi, j, buf[j], fullDequant[offset+j])
			}
		}
	}
}

func TestGemmW8A8(t *testing.T) {
	m, k, n := 2, 32, 3

	aF32 := make([]float32, m*k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	bF32 := make([]float32, k*n)
	for i := range bF32 {
		bF32[i] = float32(i%5-2) * 0.1
	}

	aQ := QuantizeW8A8(aF32)
	bQ := QuantizeW8A8(bF32)
	c := make([]float32, m*n)

	GemmW8A8(m, n, k, aQ, bQ, c)

	aDequant := aQ.Slice()
	bDequant := bQ.Slice()
	ref := make([]float32, m*n)
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += aDequant[i*k+p] * bDequant[p*n+j]
			}
			ref[i*n+j] = sum
		}
	}

	for i := range c {
		diff := float32(math.Abs(float64(c[i] - ref[i])))
		if diff > 0.01 {
			t.Errorf("GemmW8A8 index %d: got %v, want %v (diff=%v)", i, c[i], ref[i], diff)
		}
	}
}

func TestGemmW8A8NT(t *testing.T) {
	m, k, n := 2, 32, 3

	aF32 := make([]float32, m*k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	bF32 := make([]float32, n*k)
	for i := range bF32 {
		bF32[i] = float32(i%5-2) * 0.1
	}

	aQ := QuantizeW8A8(aF32)
	bQ := QuantizeW8A8(bF32)
	c := make([]float32, m*n)

	GemmW8A8NT(m, n, k, aQ, bQ, c)

	aDequant := aQ.Slice()
	bDequant := bQ.Slice()
	ref := make([]float32, m*n)
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += aDequant[i*k+p] * bDequant[j*k+p]
			}
			ref[i*n+j] = sum
		}
	}

	for i := range c {
		diff := float32(math.Abs(float64(c[i] - ref[i])))
		if diff > 0.01 {
			t.Errorf("GemmW8A8NT index %d: got %v, want %v (diff=%v)", i, c[i], ref[i], diff)
		}
	}
}

func TestGemmF32W8A8NT(t *testing.T) {
	m, k, n := 2, 32, 3

	aF32 := make([]float32, m*k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	bF32 := make([]float32, n*k)
	for i := range bF32 {
		bF32[i] = float32(i%5-2) * 0.1
	}

	bQ := QuantizeW8A8(bF32)
	c := make([]float32, m*n)

	GemmF32W8A8NT(m, n, k, aF32, bQ, c)

	bDequant := bQ.Slice()
	ref := make([]float32, m*n)
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += aF32[i*k+p] * bDequant[j*k+p]
			}
			ref[i*n+j] = sum
		}
	}

	for i := range c {
		diff := float32(math.Abs(float64(c[i] - ref[i])))
		if diff > 0.01 {
			t.Errorf("GemmF32W8A8NT index %d: got %v, want %v (diff=%v)", i, c[i], ref[i], diff)
		}
	}
}

func TestW8A8_ZeroScale(t *testing.T) {
	input := make([]float32, 32)
	q := QuantizeW8A8(input)
	dst := make([]float32, 32)
	q.Dequantize(dst)

	for i, v := range dst {
		if v != 0.0 {
			t.Errorf("index %d: zero input got %v, want 0", i, v)
		}
	}
}

func TestW8A8_GPUPtr(t *testing.T) {
	q := QuantizeW8A8(make([]float32, 32))

	ptr, size, devID := q.GPUPtr()
	if ptr != nil || size != 0 || devID != 0 {
		t.Error("expected nil GPU pointer initially")
	}

	var fakePtr [1]byte
	q.SetGPUPtr(unsafe.Pointer(&fakePtr[0]), 36, 1)
	ptr, size, devID = q.GPUPtr()
	if ptr == nil || size != 36 || devID != 1 {
		t.Errorf("GPU ptr: got (%v, %d, %d), want (non-nil, 36, 1)", ptr, size, devID)
	}
}

func BenchmarkQuantizeW8A8(b *testing.B) {
	input := linspace(-1, 1, 4096)
	b.ResetTimer()
	b.SetBytes(int64(4096 * 4))
	for range b.N {
		QuantizeW8A8(input)
	}
}

func BenchmarkDequantizeW8A8(b *testing.B) {
	input := linspace(-1, 1, 4096)
	q := QuantizeW8A8(input)
	dst := make([]float32, 4096)

	b.ResetTimer()
	b.SetBytes(int64(4096 * 4))
	for range b.N {
		q.Dequantize(dst)
	}
}

func BenchmarkGemmW8A8(b *testing.B) {
	m, k, n := 1, 4096, 4096
	aF32 := linspace(-1, 1, m*k)
	bF32 := linspace(-1, 1, k*n)
	aQ := QuantizeW8A8(aF32)
	bQ := QuantizeW8A8(bF32)
	c := make([]float32, m*n)

	b.ResetTimer()
	for b.Loop() {
		GemmW8A8(m, n, k, aQ, bQ, c)
	}
}

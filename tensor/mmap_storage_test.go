package tensor

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/device"
)

// makeF32Raw encodes float32 values as little-endian bytes.
func makeF32Raw(values []float32) []byte {
	raw := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[i*4:i*4+4], math.Float32bits(v))
	}
	return raw
}

// makeF16Raw encodes float32 values as FP16 little-endian bytes.
func makeF16Raw(values []float32) []byte {
	raw := make([]byte, len(values)*2)
	for i, v := range values {
		fp16 := float16.FromFloat32(v)
		binary.LittleEndian.PutUint16(raw[i*2:i*2+2], fp16.Bits())
	}
	return raw
}

// makeBF16Raw encodes float32 values as BF16 little-endian bytes.
func makeBF16Raw(values []float32) []byte {
	raw := make([]byte, len(values)*2)
	for i, v := range values {
		bits := math.Float32bits(v)
		bf16 := uint16(bits >> 16)
		binary.LittleEndian.PutUint16(raw[i*2:i*2+2], bf16)
	}
	return raw
}

// makeQ4_0Raw creates Q4_0 format raw bytes from float32 values by
// quantizing them the same way the existing QuantizeQ4 does, then serializing.
func makeQ4_0Raw(values []float32) []byte {
	q := QuantizeQ4(values)
	return q.RawBytes()
}

// makeQ8_0Raw creates Q8_0 format raw bytes (GGUF format: 34 bytes/block with fp16 scale).
func makeQ8_0Raw(values []float32) []byte {
	const blockSize = 32
	const blockBytes = 34
	n := len(values)
	nBlocks := (n + blockSize - 1) / blockSize
	raw := make([]byte, nBlocks*blockBytes)

	for bi := range nBlocks {
		offset := bi * blockSize

		// Find absmax.
		var absMax float32
		for j := range blockSize {
			idx := offset + j
			var v float32
			if idx < n {
				v = values[idx]
			}
			if av := float32(math.Abs(float64(v))); av > absMax {
				absMax = av
			}
		}

		var scale float32
		if absMax > 0 {
			scale = absMax / 127.0
		}

		off := bi * blockBytes
		fp16Scale := float16.FromFloat32(scale)
		binary.LittleEndian.PutUint16(raw[off:off+2], fp16Scale.Bits())

		var invScale float32
		if scale > 0 {
			invScale = 1.0 / scale
		}
		for j := range blockSize {
			var v float32
			if offset+j < n {
				v = values[offset+j]
			}
			q := int(math.Round(float64(v * invScale)))
			if q < -128 {
				q = -128
			}
			if q > 127 {
				q = 127
			}
			raw[off+2+j] = byte(int8(q))
		}
	}
	return raw
}

func TestMmapStorage_F32(t *testing.T) {
	values := []float32{1.0, -2.5, 3.14, 0.0, 42.0, -1e-5, 100.0, -100.0}
	raw := makeF32Raw(values)

	s, err := NewMmapStorage(raw, len(values), GGMLTypeF32)
	if err != nil {
		t.Fatalf("NewMmapStorage: %v", err)
	}

	if s.Len() != len(values) {
		t.Errorf("Len() = %d, want %d", s.Len(), len(values))
	}
	if s.DeviceType() != device.CPU {
		t.Errorf("DeviceType() = %v, want CPU", s.DeviceType())
	}
	if s.QType() != GGMLTypeF32 {
		t.Errorf("QType() = %d, want %d", s.QType(), GGMLTypeF32)
	}

	got := s.Slice()
	for i, want := range values {
		if got[i] != want {
			t.Errorf("Slice()[%d] = %v, want %v", i, got[i], want)
		}
	}

	// Second call should return cached result.
	got2 := s.Slice()
	if &got[0] != &got2[0] {
		t.Error("Slice() did not return cached result on second call")
	}
}

func TestMmapStorage_F16(t *testing.T) {
	values := []float32{1.0, -2.5, 3.0, 0.0, 0.5, -0.25, 10.0, -10.0}
	raw := makeF16Raw(values)

	s, err := NewMmapStorage(raw, len(values), GGMLTypeF16)
	if err != nil {
		t.Fatalf("NewMmapStorage: %v", err)
	}

	got := s.Slice()
	for i, want := range values {
		// FP16 has limited precision; allow small error.
		if diff := math.Abs(float64(got[i] - want)); diff > 0.01 {
			t.Errorf("Slice()[%d] = %v, want ~%v (diff=%v)", i, got[i], want, diff)
		}
	}
}

func TestMmapStorage_BF16(t *testing.T) {
	values := []float32{1.0, -2.0, 0.5, 0.0, 100.0, -50.0, 0.125, -0.125}
	raw := makeBF16Raw(values)

	s, err := NewMmapStorage(raw, len(values), GGMLTypeBF16)
	if err != nil {
		t.Fatalf("NewMmapStorage: %v", err)
	}

	got := s.Slice()
	for i, want := range values {
		if diff := math.Abs(float64(got[i] - want)); diff > 0.1 {
			t.Errorf("Slice()[%d] = %v, want ~%v (diff=%v)", i, got[i], want, diff)
		}
	}
}

func TestMmapStorage_Q4_0(t *testing.T) {
	// Create 64 values (2 Q4 blocks).
	values := make([]float32, 64)
	for i := range values {
		values[i] = float32(i-32) * 0.1
	}

	// Create raw Q4 bytes the same way the existing loader does.
	raw := makeQ4_0Raw(values)

	s, err := NewMmapStorage(raw, len(values), GGMLTypeQ4_0)
	if err != nil {
		t.Fatalf("NewMmapStorage: %v", err)
	}

	if s.Len() != len(values) {
		t.Errorf("Len() = %d, want %d", s.Len(), len(values))
	}

	// Compare against existing Q4Storage dequantization.
	q4 := QuantizeQ4(values)
	expected := q4.Slice()
	got := s.Slice()

	for i := range expected {
		if diff := math.Abs(float64(got[i] - expected[i])); diff > 1e-5 {
			t.Errorf("Slice()[%d] = %v, want %v (diff=%v)", i, got[i], expected[i], diff)
		}
	}
}

func TestMmapStorage_Q8_0(t *testing.T) {
	// Create 64 values (2 Q8 blocks).
	values := make([]float32, 64)
	for i := range values {
		values[i] = float32(i-32) * 0.05
	}

	raw := makeQ8_0Raw(values)

	s, err := NewMmapStorage(raw, len(values), GGMLTypeQ8_0)
	if err != nil {
		t.Fatalf("NewMmapStorage: %v", err)
	}

	if s.Len() != len(values) {
		t.Errorf("Len() = %d, want %d", s.Len(), len(values))
	}

	// Quantization introduces error; allow tolerance.
	got := s.Slice()
	for i, v := range values {
		if diff := math.Abs(float64(got[i] - v)); diff > 0.02 {
			t.Errorf("Slice()[%d] = %v, want ~%v (diff=%v)", i, got[i], v, diff)
		}
	}
}

func TestMmapStorage_Errors(t *testing.T) {
	tests := []struct {
		name    string
		data    []byte
		length  int
		qtype   GGMLType
		wantErr string
	}{
		{
			name:    "zero length",
			data:    make([]byte, 4),
			length:  0,
			qtype:   GGMLTypeF32,
			wantErr: "length must be positive",
		},
		{
			name:    "empty data",
			data:    nil,
			length:  1,
			qtype:   GGMLTypeF32,
			wantErr: "data must not be empty",
		},
		{
			name:    "data too short for F32",
			data:    make([]byte, 3),
			length:  1,
			qtype:   GGMLTypeF32,
			wantErr: "data too short",
		},
		{
			name:    "data too short for Q4_0",
			data:    make([]byte, 10),
			length:  32,
			qtype:   GGMLTypeQ4_0,
			wantErr: "data too short",
		},
		{
			name:    "unsupported type",
			data:    make([]byte, 4),
			length:  1,
			qtype:   GGMLType(99),
			wantErr: "unsupported GGML type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewMmapStorage(tt.data, tt.length, tt.qtype)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if got := err.Error(); !contains(got, tt.wantErr) {
				t.Errorf("error = %q, want containing %q", got, tt.wantErr)
			}
		})
	}
}

func TestMmapStorage_SetPanics(t *testing.T) {
	raw := makeF32Raw([]float32{1.0})
	s, _ := NewMmapStorage(raw, 1, GGMLTypeF32)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Set() did not panic")
		}
	}()
	s.Set([]float32{2.0})
}

func TestMmapStorage_RawBytes(t *testing.T) {
	values := []float32{1.0, 2.0, 3.0, 4.0}
	raw := makeF32Raw(values)

	s, err := NewMmapStorage(raw, len(values), GGMLTypeF32)
	if err != nil {
		t.Fatalf("NewMmapStorage: %v", err)
	}

	rb := s.RawBytes()
	if len(rb) != len(raw) {
		t.Errorf("RawBytes() len = %d, want %d", len(rb), len(raw))
	}
	// RawBytes should return the same underlying data.
	if &rb[0] != &raw[0] {
		t.Error("RawBytes() returned a copy, expected same slice")
	}
}

func TestMmapStorage_WithTensor(t *testing.T) {
	values := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	raw := makeF32Raw(values)

	s, err := NewMmapStorage(raw, len(values), GGMLTypeF32)
	if err != nil {
		t.Fatalf("NewMmapStorage: %v", err)
	}

	// Verify it works with NewWithStorage.
	tensor, err := NewWithStorage[float32]([]int{2, 3}, s)
	if err != nil {
		t.Fatalf("NewWithStorage: %v", err)
	}

	got := tensor.Data()
	for i, want := range values {
		if got[i] != want {
			t.Errorf("Data()[%d] = %v, want %v", i, got[i], want)
		}
	}
}

func TestMmapStorage_ByteSize(t *testing.T) {
	tests := []struct {
		name     string
		length   int
		qtype    GGMLType
		wantSize int
	}{
		{"F32 4 elems", 4, GGMLTypeF32, 16},
		{"F16 4 elems", 4, GGMLTypeF16, 8},
		{"Q4_0 32 elems", 32, GGMLTypeQ4_0, 18},
		{"Q8_0 32 elems", 32, GGMLTypeQ8_0, 34},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := make([]byte, tt.wantSize)
			s, err := NewMmapStorage(raw, tt.length, tt.qtype)
			if err != nil {
				t.Fatalf("NewMmapStorage: %v", err)
			}
			if s.ByteSize() != tt.wantSize {
				t.Errorf("ByteSize() = %d, want %d", s.ByteSize(), tt.wantSize)
			}
		})
	}
}

func TestMmapStorage_ThreadSafety(t *testing.T) {
	values := make([]float32, 128)
	for i := range values {
		values[i] = float32(i) * 0.1
	}
	raw := makeF32Raw(values)

	s, err := NewMmapStorage(raw, len(values), GGMLTypeF32)
	if err != nil {
		t.Fatalf("NewMmapStorage: %v", err)
	}

	// Call Slice() concurrently to verify sync.Once safety.
	done := make(chan []float32, 10)
	for range 10 {
		go func() {
			done <- s.Slice()
		}()
	}

	var first []float32
	for range 10 {
		result := <-done
		if first == nil {
			first = result
		}
		// All goroutines should get the same cached slice.
		if &result[0] != &first[0] {
			t.Error("concurrent Slice() calls returned different slices")
		}
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

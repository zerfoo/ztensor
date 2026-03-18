package tensor

import (
	"math"
	"sync"
	"testing"
)

func TestQuantRegistryInitRegistrations(t *testing.T) {
	// Verify all built-in types are registered via init().
	want := []string{
		"AWQ_4",
		"FP8_E4M3", "FP8_E5M2",
		"Q4_0", "Q4_K", "Q5_0", "Q5_K", "Q6_K", "Q8_0",
		"W8A8",
	}
	got := ListQuantTypes()
	if len(got) != len(want) {
		t.Fatalf("ListQuantTypes() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("ListQuantTypes()[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestGetQuantType(t *testing.T) {
	tests := []struct {
		name  string
		found bool
	}{
		{"Q4_0", true},
		{"Q8_0", true},
		{"Q4_K", true},
		{"Q5_K", true},
		{"Q6_K", true},
		{"Q5_0", true},
		{"FP8_E4M3", true},
		{"FP8_E5M2", true},
		{"NONEXISTENT", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, ok := GetQuantType(tt.name)
			if ok != tt.found {
				t.Fatalf("GetQuantType(%q) found=%v, want %v", tt.name, ok, tt.found)
			}
			if tt.found && d == nil {
				t.Fatal("got nil Dequantizer for registered type")
			}
		})
	}
}

func TestRegisterQuantTypePanics(t *testing.T) {
	t.Run("empty name", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for empty name")
			}
		}()
		RegisterQuantType("", q4Dequantizer{})
	})

	t.Run("nil dequantizer", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for nil dequantizer")
			}
		}()
		RegisterQuantType("test_nil", nil)
	})

	t.Run("duplicate registration", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for duplicate registration")
			}
		}()
		// Q4_0 is already registered via init().
		RegisterQuantType("Q4_0", q4Dequantizer{})
	})
}

func TestCustomQuantTypeRegistration(t *testing.T) {
	// Save and restore registry state.
	quantMu.Lock()
	saved := quantRegistry
	quantRegistry = make(map[string]Dequantizer)
	quantMu.Unlock()
	defer func() {
		quantMu.Lock()
		quantRegistry = saved
		quantMu.Unlock()
	}()

	d := &mockDequantizer{blockSize: 64, bitsPerWeight: 3}
	RegisterQuantType("CUSTOM_3BIT", d)

	got, ok := GetQuantType("CUSTOM_3BIT")
	if !ok {
		t.Fatal("CUSTOM_3BIT not found after registration")
	}
	if got.BlockSize() != 64 {
		t.Errorf("BlockSize() = %d, want 64", got.BlockSize())
	}
	if got.BitsPerWeight() != 3 {
		t.Errorf("BitsPerWeight() = %d, want 3", got.BitsPerWeight())
	}

	names := ListQuantTypes()
	if len(names) != 1 || names[0] != "CUSTOM_3BIT" {
		t.Errorf("ListQuantTypes() = %v, want [CUSTOM_3BIT]", names)
	}
}

func TestQuantRegistryConcurrency(t *testing.T) {
	// Save and restore registry state.
	quantMu.Lock()
	saved := quantRegistry
	quantRegistry = make(map[string]Dequantizer)
	quantMu.Unlock()
	defer func() {
		quantMu.Lock()
		quantRegistry = saved
		quantMu.Unlock()
	}()

	var wg sync.WaitGroup

	// Register types concurrently.
	for i := range 50 {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			name := "CONCURRENT_" + string(rune('A'+idx))
			RegisterQuantType(name, &mockDequantizer{blockSize: idx + 1, bitsPerWeight: 4})
		}(i)
	}
	wg.Wait()

	// Read concurrently.
	for i := range 50 {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			name := "CONCURRENT_" + string(rune('A'+idx))
			d, ok := GetQuantType(name)
			if !ok {
				t.Errorf("GetQuantType(%q) not found", name)
				return
			}
			if d.BlockSize() != idx+1 {
				t.Errorf("GetQuantType(%q).BlockSize() = %d, want %d", name, d.BlockSize(), idx+1)
			}
		}(i)
	}
	wg.Wait()

	names := ListQuantTypes()
	if len(names) != 50 {
		t.Errorf("ListQuantTypes() returned %d entries, want 50", len(names))
	}
}

func TestQ4DequantizerRoundTrip(t *testing.T) {
	d, ok := GetQuantType("Q4_0")
	if !ok {
		t.Fatal("Q4_0 not registered")
	}
	if d.BlockSize() != 32 {
		t.Errorf("Q4_0 BlockSize() = %d, want 32", d.BlockSize())
	}
	if d.BitsPerWeight() != 4 {
		t.Errorf("Q4_0 BitsPerWeight() = %d, want 4", d.BitsPerWeight())
	}

	// Quantize via existing code, then dequantize via registry.
	input := make([]float32, 64)
	for i := range input {
		input[i] = float32(i-32) / 32.0
	}
	q := QuantizeQ4(input)
	raw := q.RawBytes()

	dst := make([]float32, 64)
	if err := d.Dequantize(raw, dst); err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	// Compare with direct dequantization.
	ref := make([]float32, 64)
	q.Dequantize(ref)
	for i := range dst {
		if dst[i] != ref[i] {
			t.Errorf("index %d: registry=%v, direct=%v", i, dst[i], ref[i])
		}
	}
}

func TestQ8DequantizerRoundTrip(t *testing.T) {
	d, ok := GetQuantType("Q8_0")
	if !ok {
		t.Fatal("Q8_0 not registered")
	}

	input := make([]float32, 64)
	for i := range input {
		input[i] = float32(i-32) / 16.0
	}
	q := QuantizeQ8(input)
	raw := q.RawBytes()

	dst := make([]float32, 64)
	if err := d.Dequantize(raw, dst); err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	ref := make([]float32, 64)
	q.Dequantize(ref)
	for i := range dst {
		if dst[i] != ref[i] {
			t.Errorf("index %d: registry=%v, direct=%v", i, dst[i], ref[i])
		}
	}
}

func TestFP8E4M3DequantizerRoundTrip(t *testing.T) {
	d, ok := GetQuantType("FP8_E4M3")
	if !ok {
		t.Fatal("FP8_E4M3 not registered")
	}
	if d.BlockSize() != 1 {
		t.Errorf("FP8_E4M3 BlockSize() = %d, want 1", d.BlockSize())
	}
	if d.BitsPerWeight() != 8 {
		t.Errorf("FP8_E4M3 BitsPerWeight() = %d, want 8", d.BitsPerWeight())
	}

	input := []float32{0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -200.0, 0.001}
	s := NewFP8E4M3Storage(input)
	rawData := s.RawBytes()
	scale := s.Scale()

	// Build src: 4-byte LE scale + raw FP8 bytes.
	src := make([]byte, 4+len(rawData))
	putFloat32LE(src[:4], scale)
	copy(src[4:], rawData)

	dst := make([]float32, len(input))
	if err := d.Dequantize(src, dst); err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	ref := s.Slice()
	for i := range dst {
		if dst[i] != ref[i] {
			t.Errorf("index %d: registry=%v, direct=%v", i, dst[i], ref[i])
		}
	}
}

func TestFP8E5M2DequantizerRoundTrip(t *testing.T) {
	d, ok := GetQuantType("FP8_E5M2")
	if !ok {
		t.Fatal("FP8_E5M2 not registered")
	}

	input := []float32{0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -200.0}
	s := NewFP8E5M2Storage(input)
	rawData := s.Slice() // Get dequantized reference

	// Build src from raw storage.
	fp8Storage := NewFP8E5M2Storage(input)
	src := make([]byte, 4+fp8Storage.Len())
	putFloat32LE(src[:4], fp8Storage.Scale())
	copy(src[4:], fp8Storage.data)

	dst := make([]float32, len(input))
	if err := d.Dequantize(src, dst); err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	for i := range dst {
		if dst[i] != rawData[i] {
			t.Errorf("index %d: registry=%v, direct=%v", i, dst[i], rawData[i])
		}
	}
}

func TestDequantizerShortData(t *testing.T) {
	tests := []struct {
		name string
		n    int
	}{
		{"Q4_0", 32},
		{"Q8_0", 32},
		{"FP8_E4M3", 8},
		{"FP8_E5M2", 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, _ := GetQuantType(tt.name)
			dst := make([]float32, tt.n)
			err := d.Dequantize(nil, dst)
			if err == nil {
				t.Error("expected error for nil/short src")
			}
		})
	}
}

func TestDequantizerBlockSizeAndBits(t *testing.T) {
	tests := []struct {
		name      string
		blockSize int
		bits      int
	}{
		{"Q4_0", 32, 4},
		{"Q8_0", 32, 8},
		{"Q4_K", 256, 4},
		{"Q5_K", 256, 5},
		{"Q6_K", 256, 6},
		{"Q5_0", 32, 5},
		{"FP8_E4M3", 1, 8},
		{"FP8_E5M2", 1, 8},
		{"W8A8", 32, 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, ok := GetQuantType(tt.name)
			if !ok {
				t.Fatalf("type %q not registered", tt.name)
			}
			if d.BlockSize() != tt.blockSize {
				t.Errorf("BlockSize() = %d, want %d", d.BlockSize(), tt.blockSize)
			}
			if d.BitsPerWeight() != tt.bits {
				t.Errorf("BitsPerWeight() = %d, want %d", d.BitsPerWeight(), tt.bits)
			}
		})
	}
}

// mockDequantizer is a test helper.
type mockDequantizer struct {
	blockSize     int
	bitsPerWeight int
}

func (m *mockDequantizer) Dequantize(src []byte, dst []float32) error {
	for i := range dst {
		dst[i] = 0
	}
	return nil
}

func (m *mockDequantizer) BlockSize() int     { return m.blockSize }
func (m *mockDequantizer) BitsPerWeight() int { return m.bitsPerWeight }

// putFloat32LE encodes a float32 as little-endian bytes.
func putFloat32LE(b []byte, f float32) {
	bits := math.Float32bits(f)
	b[0] = byte(bits)
	b[1] = byte(bits >> 8)
	b[2] = byte(bits >> 16)
	b[3] = byte(bits >> 24)
}

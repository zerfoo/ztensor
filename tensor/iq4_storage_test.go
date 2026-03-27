package tensor

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/float16"
)

// encodeIQ4NLBlock builds an 18-byte IQ4_NL block from a scale and 32 nibble indices.
func encodeIQ4NLBlock(scale float32, nibbles [32]uint8) [18]byte {
	var block [18]byte
	binary.LittleEndian.PutUint16(block[0:2], float16.FromFloat32(scale).Bits())
	for i := range 16 {
		block[2+i] = (nibbles[2*i] & 0x0F) | (nibbles[2*i+1] << 4)
	}
	return block
}

func TestDequantizeIQ4NL(t *testing.T) {
	scale := float32(2.0)
	var nibbles [32]uint8
	for i := range 32 {
		nibbles[i] = uint8(i % 16)
	}
	block := encodeIQ4NLBlock(scale, nibbles)

	var dst [32]float32
	DequantizeIQ4NL(block[:], dst[:])

	// Verify: use fp16 roundtrip for scale (matches what DequantizeIQ4NL does)
	actualScale := float16.FromFloat32(scale).ToFloat32()

	for i := range 32 {
		idx := nibbles[i]
		want := actualScale * IQ4NLTable[idx]
		if math.Abs(float64(dst[i]-want)) > 1e-6 {
			t.Errorf("dst[%d] = %v, want %v (nibble=%d)", i, dst[i], want, idx)
		}
	}
}

func TestIQ4NLStorageRoundtrip(t *testing.T) {
	const numElements = 64 // 2 blocks
	scales := [2]float32{1.5, 0.75}

	var raw [2 * 18]byte
	var expected [numElements]float32

	for bi := range 2 {
		var nibbles [32]uint8
		for i := range 32 {
			nibbles[i] = uint8((bi*7 + i*3) % 16)
		}
		block := encodeIQ4NLBlock(scales[bi], nibbles)
		copy(raw[bi*18:], block[:])

		actualScale := float16.FromFloat32(scales[bi]).ToFloat32()
		for i := range 32 {
			expected[bi*32+i] = actualScale * IQ4NLTable[nibbles[i]]
		}
	}

	s, err := NewIQ4NLStorageFromRaw(raw[:], numElements)
	if err != nil {
		t.Fatalf("NewIQ4NLStorageFromRaw: %v", err)
	}

	if s.Len() != numElements {
		t.Fatalf("Len() = %d, want %d", s.Len(), numElements)
	}

	got := s.Slice()
	if len(got) != numElements {
		t.Fatalf("Slice() length = %d, want %d", len(got), numElements)
	}

	for i := range numElements {
		if math.Abs(float64(got[i]-expected[i])) > 1e-6 {
			t.Errorf("element[%d] = %v, want %v", i, got[i], expected[i])
		}
	}
}

func TestIQ4NLStoragePartialBlock(t *testing.T) {
	// 48 elements = 1 full block (32) + 1 partial block (16)
	const numElements = 48
	nBlocks := 2

	var raw [2 * 18]byte
	scale := float32(1.0)
	for bi := range nBlocks {
		var nibbles [32]uint8
		for i := range 32 {
			nibbles[i] = uint8(i % 16)
		}
		block := encodeIQ4NLBlock(scale, nibbles)
		copy(raw[bi*18:], block[:])
	}

	s, err := NewIQ4NLStorageFromRaw(raw[:], numElements)
	if err != nil {
		t.Fatalf("NewIQ4NLStorageFromRaw: %v", err)
	}

	got := s.Slice()
	if len(got) != numElements {
		t.Fatalf("Slice() length = %d, want %d", len(got), numElements)
	}

	actualScale := float16.FromFloat32(scale).ToFloat32()
	for i := range numElements {
		idx := uint8(i % 16)
		want := actualScale * IQ4NLTable[idx]
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Errorf("element[%d] = %v, want %v", i, got[i], want)
		}
	}
}

func TestIQ4NLStorageErrors(t *testing.T) {
	t.Run("zero elements", func(t *testing.T) {
		_, err := NewIQ4NLStorageFromRaw(nil, 0)
		if err == nil {
			t.Fatal("expected error for zero elements")
		}
	})

	t.Run("negative elements", func(t *testing.T) {
		_, err := NewIQ4NLStorageFromRaw(nil, -1)
		if err == nil {
			t.Fatal("expected error for negative elements")
		}
	})

	t.Run("short raw data", func(t *testing.T) {
		_, err := NewIQ4NLStorageFromRaw(make([]byte, 10), 32)
		if err == nil {
			t.Fatal("expected error for short raw data")
		}
	})
}

func TestIQ4NLStorageImmutable(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on Set()")
		}
	}()

	raw := make([]byte, 18)
	s, _ := NewIQ4NLStorageFromRaw(raw, 32)
	s.Set(nil)
}

func TestIQ4NLTableValues(t *testing.T) {
	// Table should have 16 entries
	if len(IQ4NLTable) != 16 {
		t.Fatalf("IQ4NLTable has %d entries, want 16", len(IQ4NLTable))
	}

	// Zero index 7 should be 0.0
	if IQ4NLTable[7] != 0.0 {
		t.Errorf("IQ4NLTable[7] = %v, want 0.0", IQ4NLTable[7])
	}

	// Table should be monotonically increasing
	for i := 1; i < 16; i++ {
		if IQ4NLTable[i] <= IQ4NLTable[i-1] {
			t.Errorf("IQ4NLTable not monotonically increasing at [%d]=%v <= [%d]=%v",
				i, IQ4NLTable[i], i-1, IQ4NLTable[i-1])
		}
	}
}

func TestMergeIQ4NLStorage(t *testing.T) {
	raw1 := make([]byte, 18)
	raw2 := make([]byte, 18)
	raw1[0] = 0x01 // distinguish blocks
	raw2[0] = 0x02

	s1, _ := NewIQ4NLStorageFromRaw(raw1, 32)
	s2, _ := NewIQ4NLStorageFromRaw(raw2, 32)

	merged := MergeIQ4NLStorage(s1, s2)
	if merged.Len() != 64 {
		t.Fatalf("merged.Len() = %d, want 64", merged.Len())
	}
	if merged.NumBlocks() != 2 {
		t.Fatalf("merged.NumBlocks() = %d, want 2", merged.NumBlocks())
	}
}

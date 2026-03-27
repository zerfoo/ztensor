package tensor

import (
	"math"
	"testing"
)

func TestIQ2XXSGrid_Init(t *testing.T) {
	// Byte 0x00 = all 2-bit indices are 0 -> all -1.0
	for i := 0; i < 4; i++ {
		if iq2xxsGrid[0x00][i] != -1.0 {
			t.Errorf("grid[0x00][%d] = %g, want -1.0", i, iq2xxsGrid[0x00][i])
		}
	}
	// Byte 0xFF = all 2-bit indices are 3 -> all 1.0
	for i := 0; i < 4; i++ {
		if iq2xxsGrid[0xFF][i] != 1.0 {
			t.Errorf("grid[0xFF][%d] = %g, want 1.0", i, iq2xxsGrid[0xFF][i])
		}
	}
	// Byte 0xE4 = 0b11_10_01_00 -> indices [0,1,2,3] -> [-1, -1/3, 1/3, 1]
	want := [4]float32{-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0}
	for i := 0; i < 4; i++ {
		if math.Abs(float64(iq2xxsGrid[0xE4][i]-want[i])) > 1e-7 {
			t.Errorf("grid[0xE4][%d] = %g, want %g", i, iq2xxsGrid[0xE4][i], want[i])
		}
	}
}

func TestIQ2XXSStorage_Dequantize_SingleBlock(t *testing.T) {
	s := NewIQ2XXSStorage(256)

	// Fill block with 0xE4 bytes: each byte -> [-1, -1/3, 1/3, 1]
	data := make([]byte, 64)
	for i := range data {
		data[i] = 0xE4
	}
	s.SetBlock(0, 2.0, data)

	out := s.Dequantize()
	if len(out) != 256 {
		t.Fatalf("Dequantize() len = %d, want 256", len(out))
	}

	gridVals := [4]float32{-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0}
	scale := float32(2.0)
	for i := 0; i < 256; i++ {
		want := gridVals[i%4] * scale
		if math.Abs(float64(out[i]-want)) > 1e-6 {
			t.Errorf("out[%d] = %g, want %g", i, out[i], want)
		}
	}
}

func TestIQ2XXSStorage_Dequantize_AllZeroBits(t *testing.T) {
	s := NewIQ2XXSStorage(256)
	data := make([]byte, 64) // all 0x00 -> all indices 0 -> -1.0
	s.SetBlock(0, 3.0, data)

	out := s.Dequantize()
	for i, v := range out {
		if v != -3.0 {
			t.Errorf("out[%d] = %g, want -3.0", i, v)
		}
	}
}

func TestIQ2XXSStorage_Dequantize_AllOneBits(t *testing.T) {
	s := NewIQ2XXSStorage(256)
	data := make([]byte, 64)
	for i := range data {
		data[i] = 0xFF // all indices 3 -> 1.0
	}
	s.SetBlock(0, 5.0, data)

	out := s.Dequantize()
	for i, v := range out {
		if v != 5.0 {
			t.Errorf("out[%d] = %g, want 5.0", i, v)
		}
	}
}

func TestIQ2XXSStorage_Dequantize_MultipleBlocks(t *testing.T) {
	s := NewIQ2XXSStorage(512)

	data0 := make([]byte, 64)
	for i := range data0 {
		data0[i] = 0x00 // all -1.0
	}
	s.SetBlock(0, 1.0, data0)

	data1 := make([]byte, 64)
	for i := range data1 {
		data1[i] = 0xFF // all 1.0
	}
	s.SetBlock(1, 2.0, data1)

	out := s.Dequantize()
	if len(out) != 512 {
		t.Fatalf("len = %d, want 512", len(out))
	}

	for i := 0; i < 256; i++ {
		if out[i] != -1.0 {
			t.Errorf("block0 out[%d] = %g, want -1.0", i, out[i])
			break
		}
	}
	for i := 256; i < 512; i++ {
		if out[i] != 2.0 {
			t.Errorf("block1 out[%d] = %g, want 2.0", i, out[i])
			break
		}
	}
}

func TestIQ2XXSStorage_Dequantize_PartialBlock(t *testing.T) {
	// 100 elements = 1 block, but only 100 values returned
	s := NewIQ2XXSStorage(100)
	data := make([]byte, 64)
	for i := range data {
		data[i] = 0xFF // all 1.0
	}
	s.SetBlock(0, 4.0, data)

	out := s.Dequantize()
	if len(out) != 100 {
		t.Fatalf("len = %d, want 100", len(out))
	}
	for i, v := range out {
		if v != 4.0 {
			t.Errorf("out[%d] = %g, want 4.0", i, v)
		}
	}
}

func TestIQ2XXSStorage_Len(t *testing.T) {
	tests := []int{0, 1, 100, 256, 257, 512, 1024}
	for _, n := range tests {
		s := NewIQ2XXSStorage(n)
		if s.Len() != n {
			t.Errorf("NewIQ2XXSStorage(%d).Len() = %d", n, s.Len())
		}
	}
}

func TestIQ2XXSStorage_Empty(t *testing.T) {
	s := NewIQ2XXSStorage(0)
	if s.Len() != 0 {
		t.Errorf("Len() = %d, want 0", s.Len())
	}
	out := s.Dequantize()
	if len(out) != 0 {
		t.Errorf("Dequantize() len = %d, want 0", len(out))
	}
}

func TestIQ2XXSStorage_DeviceType(t *testing.T) {
	s := NewIQ2XXSStorage(256)
	if dt := s.DeviceType(); dt != 0 { // device.CPU = 0
		t.Errorf("DeviceType() = %v, want CPU (0)", dt)
	}
}

func TestIQ2XXSStorage_SetBlock_PanicOutOfRange(t *testing.T) {
	s := NewIQ2XXSStorage(256) // 1 block
	defer func() {
		if r := recover(); r == nil {
			t.Error("SetBlock(1, ...) did not panic for out of range")
		}
	}()
	s.SetBlock(1, 1.0, make([]byte, 64))
}

func TestIQ2XXSStorage_SetBlock_PanicWrongSize(t *testing.T) {
	s := NewIQ2XXSStorage(256)
	defer func() {
		if r := recover(); r == nil {
			t.Error("SetBlock with wrong data size did not panic")
		}
	}()
	s.SetBlock(0, 1.0, make([]byte, 32))
}

func TestIQ2XXSStorage_MarshalRoundTrip(t *testing.T) {
	s := NewIQ2XXSStorage(256)
	data := make([]byte, 64)
	for i := range data {
		data[i] = byte(i)
	}
	s.SetBlock(0, 42.5, data)

	buf, err := s.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	s2 := &IQ2XXSStorage{}
	if err := s2.UnmarshalBinary(buf); err != nil {
		t.Fatal(err)
	}

	if s2.Len() != s.Len() {
		t.Fatalf("Len() = %d, want %d", s2.Len(), s.Len())
	}

	out1 := s.Dequantize()
	out2 := s2.Dequantize()
	for i := range out1 {
		if math.Abs(float64(out1[i]-out2[i])) > 1e-7 {
			t.Errorf("mismatch at %d: %g vs %g", i, out1[i], out2[i])
		}
	}
}

func TestIQ2XXSStorage_NegativeSize(t *testing.T) {
	s := NewIQ2XXSStorage(-5)
	if s.Len() != 0 {
		t.Errorf("Len() = %d, want 0 for negative input", s.Len())
	}
}

func TestIQ2XXSStorage_ZeroScale(t *testing.T) {
	s := NewIQ2XXSStorage(256)
	data := make([]byte, 64)
	for i := range data {
		data[i] = 0xE4
	}
	s.SetBlock(0, 0.0, data)
	out := s.Dequantize()
	for i, v := range out {
		if v != 0.0 {
			t.Errorf("out[%d] = %g, want 0.0 (zero scale)", i, v)
		}
	}
}

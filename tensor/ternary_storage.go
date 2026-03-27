package tensor

import "github.com/zerfoo/ztensor/device"

// TernaryStorage packs ternary weights {-1, 0, 1} into 2 bits per value.
// Each byte holds 4 values. Encoding: 00=-1, 01=0, 10=1.
type TernaryStorage struct {
	data []byte
	len  int
}

// NewTernaryStorage creates a TernaryStorage that can hold size ternary values.
// All values are initialized to zero.
func NewTernaryStorage(size int) *TernaryStorage {
	if size < 0 {
		size = 0
	}
	nBytes := (size + 3) / 4
	return &TernaryStorage{
		data: make([]byte, nBytes),
		len:  size,
	}
}

// NewTernaryStorageFrom creates a TernaryStorage from a slice of int8 values.
// Each value must be -1, 0, or 1; otherwise the function panics.
func NewTernaryStorageFrom(values []int8) *TernaryStorage {
	s := NewTernaryStorage(len(values))
	for i, v := range values {
		s.Set(i, v)
	}
	return s
}

// Len returns the number of ternary values stored.
func (s *TernaryStorage) Len() int { return s.len }

// Get returns the ternary value at index i as -1, 0, or 1.
func (s *TernaryStorage) Get(i int) int8 {
	if i < 0 || i >= s.len {
		panic("tensor: TernaryStorage index out of range")
	}
	byteIdx := i / 4
	bitOffset := uint(i%4) * 2
	bits := (s.data[byteIdx] >> bitOffset) & 0x03
	return int8(bits) - 1
}

// Set stores a ternary value (-1, 0, or 1) at index i.
// Panics if val is not in {-1, 0, 1} or i is out of range.
func (s *TernaryStorage) Set(i int, val int8) {
	if i < 0 || i >= s.len {
		panic("tensor: TernaryStorage index out of range")
	}
	if val < -1 || val > 1 {
		panic("tensor: TernaryStorage value must be -1, 0, or 1")
	}
	encoded := byte(val + 1) // -1->0, 0->1, 1->2
	byteIdx := i / 4
	bitOffset := uint(i%4) * 2
	s.data[byteIdx] = (s.data[byteIdx] & ^(0x03 << bitOffset)) | (encoded << bitOffset)
}

// Slice dequantizes all ternary values to a float32 slice.
func (s *TernaryStorage) Slice() []float32 {
	out := make([]float32, s.len)
	for i := range out {
		out[i] = float32(s.Get(i))
	}
	return out
}

// RawBytes returns the underlying packed byte slice.
func (s *TernaryStorage) RawBytes() []byte { return s.data }

// DeviceType returns device.CPU.
func (s *TernaryStorage) DeviceType() device.Type { return device.CPU }

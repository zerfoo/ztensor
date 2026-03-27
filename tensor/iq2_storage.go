package tensor

import (
	"encoding/binary"
	"math"

	"github.com/zerfoo/ztensor/device"
)

// IQ2XXS block size: each super-block contains 256 elements.
const iq2xxsBlockSize = 256

// iq2xxsGrid is the precomputed codebook for IQ2_XXS.
// Each entry maps a byte (packing 4 two-bit indices) to 4 dequantized float32 values.
// The grid values are based on the E8 lattice codebook used in llama.cpp's IQ2_XXS.
// Two-bit values map to: 0 -> -1.0, 1 -> -1/3, 2 -> 1/3, 3 -> 1.0
var iq2xxsGrid [256][4]float32

func init() {
	gridValues := [4]float32{-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0}
	for b := 0; b < 256; b++ {
		iq2xxsGrid[b] = [4]float32{
			gridValues[(b>>0)&0x03],
			gridValues[(b>>2)&0x03],
			gridValues[(b>>4)&0x03],
			gridValues[(b>>6)&0x03],
		}
	}
}

// IQ2XXSStorage packs 2-bit quantized values with per-block scale factors.
// Each super-block holds 256 elements: 64 data bytes + 1 float32 scale.
// Dequantization: value = gridLookup(data_byte) * scale.
type IQ2XXSStorage struct {
	data        []byte
	scales      []float32
	numElements int
}

// NewIQ2XXSStorage creates an IQ2XXSStorage that can hold numElements values.
func NewIQ2XXSStorage(numElements int) *IQ2XXSStorage {
	if numElements < 0 {
		numElements = 0
	}
	numBlocks := (numElements + iq2xxsBlockSize - 1) / iq2xxsBlockSize
	bytesPerBlock := iq2xxsBlockSize / 4 // 4 values per byte at 2 bits each = 64 bytes
	return &IQ2XXSStorage{
		data:        make([]byte, numBlocks*bytesPerBlock),
		scales:      make([]float32, numBlocks),
		numElements: numElements,
	}
}

// Len returns the number of quantized elements.
func (s *IQ2XXSStorage) Len() int { return s.numElements }

// SetBlock sets the scale and packed data for the block at blockIdx.
// data must contain exactly 64 bytes (256 elements / 4 per byte).
func (s *IQ2XXSStorage) SetBlock(blockIdx int, scale float32, data []byte) {
	bytesPerBlock := iq2xxsBlockSize / 4
	if blockIdx < 0 || blockIdx >= len(s.scales) {
		panic("tensor: IQ2XXSStorage block index out of range")
	}
	if len(data) != bytesPerBlock {
		panic("tensor: IQ2XXSStorage SetBlock data must be 64 bytes")
	}
	s.scales[blockIdx] = scale
	copy(s.data[blockIdx*bytesPerBlock:], data)
}

// Dequantize converts all quantized values to float32.
func (s *IQ2XXSStorage) Dequantize() []float32 {
	out := make([]float32, s.numElements)
	bytesPerBlock := iq2xxsBlockSize / 4
	numBlocks := len(s.scales)
	idx := 0
	for block := 0; block < numBlocks; block++ {
		scale := s.scales[block]
		base := block * bytesPerBlock
		for b := 0; b < bytesPerBlock && idx < s.numElements; b++ {
			grid := iq2xxsGrid[s.data[base+b]]
			for v := 0; v < 4 && idx < s.numElements; v++ {
				out[idx] = grid[v] * scale
				idx++
			}
		}
	}
	return out
}

// RawBytes returns the underlying packed byte slice.
func (s *IQ2XXSStorage) RawBytes() []byte { return s.data }

// RawScales returns the block scale factors.
func (s *IQ2XXSStorage) RawScales() []float32 { return s.scales }

// DeviceType returns device.CPU.
func (s *IQ2XXSStorage) DeviceType() device.Type { return device.CPU }

// MarshalBinary encodes the storage as [numElements(4)] [scales...] [data...].
func (s *IQ2XXSStorage) MarshalBinary() ([]byte, error) {
	buf := make([]byte, 4+len(s.scales)*4+len(s.data))
	binary.LittleEndian.PutUint32(buf[0:4], uint32(s.numElements))
	for i, sc := range s.scales {
		binary.LittleEndian.PutUint32(buf[4+i*4:], math.Float32bits(sc))
	}
	copy(buf[4+len(s.scales)*4:], s.data)
	return buf, nil
}

// UnmarshalBinary decodes the storage from bytes produced by MarshalBinary.
func (s *IQ2XXSStorage) UnmarshalBinary(buf []byte) error {
	if len(buf) < 4 {
		panic("tensor: IQ2XXSStorage UnmarshalBinary: buffer too short")
	}
	s.numElements = int(binary.LittleEndian.Uint32(buf[0:4]))
	numBlocks := (s.numElements + iq2xxsBlockSize - 1) / iq2xxsBlockSize
	bytesPerBlock := iq2xxsBlockSize / 4
	s.scales = make([]float32, numBlocks)
	for i := range s.scales {
		s.scales[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4+i*4:]))
	}
	s.data = make([]byte, numBlocks*bytesPerBlock)
	copy(s.data, buf[4+numBlocks*4:])
	return nil
}

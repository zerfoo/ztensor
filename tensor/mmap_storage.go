package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/device"
)

// GGMLType identifies the quantization format of mmap'd tensor data.
// These values match the GGML type IDs used in GGUF files.
type GGMLType int

const (
	GGMLTypeF32  GGMLType = 0
	GGMLTypeF16  GGMLType = 1
	GGMLTypeQ4_0 GGMLType = 2
	GGMLTypeQ4_1 GGMLType = 3
	GGMLTypeQ5_0 GGMLType = 6
	GGMLTypeQ5_1 GGMLType = 7
	GGMLTypeQ8_0 GGMLType = 8
	GGMLTypeQ4_K GGMLType = 12
	GGMLTypeQ5_K GGMLType = 13
	GGMLTypeQ6_K GGMLType = 14
	GGMLTypeBF16 GGMLType = 30
)

// MmapStorage wraps a byte slice from an mmap'd GGUF file region.
// It implements Storage[float32] by lazily dequantizing the raw bytes
// on first access. The underlying byte slice is NOT copied -- it points
// directly into the mmap'd region.
type MmapStorage struct {
	data     []byte   // slice into mmap'd region (not a copy)
	offset   int      // byte offset into the original mmap for reference
	length   int      // number of logical float32 elements
	byteSize int      // raw byte size of the mmap'd region
	qtype    GGMLType // quantization type

	decoded  []float32 // lazily populated on first Slice() call
	decodeMu sync.Once

	gpuPtr      unsafe.Pointer // cached GPU device pointer for raw bytes
	gpuByteSize int
	gpuDeviceID int
}

// NewMmapStorage creates an MmapStorage that wraps a slice of mmap'd bytes.
// The data slice must remain valid for the lifetime of this storage (i.e.,
// the mmap must not be unmapped while this storage is in use).
//
// Parameters:
//   - data: raw bytes from the mmap'd region for this tensor
//   - length: number of logical float32 elements
//   - qtype: the GGML quantization type of the raw data
func NewMmapStorage(data []byte, length int, qtype GGMLType) (*MmapStorage, error) {
	if length <= 0 {
		return nil, fmt.Errorf("mmap storage: length must be positive, got %d", length)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("mmap storage: data must not be empty")
	}

	expectedSize, err := mmapByteSize(qtype, length)
	if err != nil {
		return nil, fmt.Errorf("mmap storage: %w", err)
	}
	if len(data) < expectedSize {
		return nil, fmt.Errorf("mmap storage: data too short for %d elements of type %d: need %d bytes, got %d",
			length, qtype, expectedSize, len(data))
	}

	return &MmapStorage{
		data:     data[:expectedSize],
		length:   length,
		byteSize: expectedSize,
		qtype:    qtype,
	}, nil
}

// mmapByteSize returns the expected byte size for a given GGML type and element count.
func mmapByteSize(qtype GGMLType, numElements int) (int, error) {
	switch qtype {
	case GGMLTypeF32:
		return numElements * 4, nil
	case GGMLTypeF16, GGMLTypeBF16:
		return numElements * 2, nil
	case GGMLTypeQ4_0:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 18, nil // 2 bytes scale + 16 bytes data
	case GGMLTypeQ8_0:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 34, nil // 2 bytes fp16 scale + 32 bytes int8
	case GGMLTypeQ4_1:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 20, nil // 2 bytes scale + 2 bytes min + 16 bytes data
	case GGMLTypeQ5_0:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 22, nil // 2 bytes scale + 4 bytes high bits + 16 bytes data
	case GGMLTypeQ5_1:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 24, nil // 2 bytes scale + 2 bytes min + 4 bytes high bits + 16 bytes data
	case GGMLTypeQ4_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 144, nil
	case GGMLTypeQ5_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 176, nil
	case GGMLTypeQ6_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 210, nil
	default:
		return 0, fmt.Errorf("unsupported GGML type %d", qtype)
	}
}

// Len returns the number of logical float32 elements.
func (s *MmapStorage) Len() int { return s.length }

// Slice returns a dequantized float32 view of the mmap'd data.
// The first call triggers dequantization; subsequent calls return the cached result.
// For F32 data, this reinterprets the bytes directly (zero-copy via unsafe would be
// ideal but we copy for safety since mmap pages may be read-only).
func (s *MmapStorage) Slice() []float32 {
	s.decodeMu.Do(func() {
		s.decoded = make([]float32, s.length)
		s.dequantize(s.decoded)
	})
	return s.decoded
}

// Set is not supported on mmap'd storage (weights are immutable).
func (s *MmapStorage) Set(_ []float32) {
	panic("MmapStorage is immutable: backed by mmap'd file data")
}

// DeviceType returns device.CPU.
func (s *MmapStorage) DeviceType() device.Type { return device.CPU }

// RawBytes returns the raw mmap'd byte slice for direct GPU DMA upload.
func (s *MmapStorage) RawBytes() []byte { return s.data }

// QType returns the GGML quantization type of the stored data.
func (s *MmapStorage) QType() GGMLType { return s.qtype }

// ByteSize returns the raw byte size of the mmap'd data.
func (s *MmapStorage) ByteSize() int { return s.byteSize }

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw quantized bytes.
// After calling this, GPUPtr() returns the cached pointer and the GPU engine can
// skip per-operation H2D copies.
func (s *MmapStorage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	s.gpuPtr = ptr
	s.gpuByteSize = byteSize
	s.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
// Returns (nil, 0, 0) if the data has not been uploaded to GPU yet.
func (s *MmapStorage) GPUPtr() (unsafe.Pointer, int, int) {
	return s.gpuPtr, s.gpuByteSize, s.gpuDeviceID
}

// RawBytesGPU returns the raw bytes in GPU-optimized layout. For Q4_0, this
// repacks the interleaved blocks into separated scales+data format matching
// Q4Storage.RawBytesGPU. For all other types, returns the raw bytes as-is.
func (s *MmapStorage) RawBytesGPU() []byte {
	if s.qtype != GGMLTypeQ4_0 {
		return s.data
	}
	// Q4_0: repack from interleaved [scale(2) data(16)]... to
	// [all_scales(N*2) padding all_data(N*16)].
	const blockBytes = 18
	totalBlocks := len(s.data) / blockBytes
	scaleBytes := totalBlocks * 2
	paddedScaleBytes := (scaleBytes + 15) &^ 15
	dataBytes := totalBlocks * 16
	out := make([]byte, paddedScaleBytes+dataBytes)
	for i := range totalBlocks {
		off := i * blockBytes
		copy(out[i*2:i*2+2], s.data[off:off+2])               // scale
		copy(out[paddedScaleBytes+i*16:], s.data[off+2:off+18]) // data
	}
	return out
}

// dequantize decodes the raw mmap'd bytes into float32 based on the quantization type.
func (s *MmapStorage) dequantize(dst []float32) {
	switch s.qtype {
	case GGMLTypeF32:
		s.dequantizeF32(dst)
	case GGMLTypeF16:
		s.dequantizeF16(dst)
	case GGMLTypeBF16:
		s.dequantizeBF16(dst)
	case GGMLTypeQ4_0:
		s.dequantizeQ4_0(dst)
	case GGMLTypeQ8_0:
		s.dequantizeQ8_0(dst)
	case GGMLTypeQ4_1:
		s.dequantizeQ4_1(dst)
	case GGMLTypeQ5_0:
		s.dequantizeQ5_0(dst)
	case GGMLTypeQ5_1:
		s.dequantizeQ5_1(dst)
	case GGMLTypeQ4_K:
		s.dequantizeQ4K(dst)
	case GGMLTypeQ5_K:
		s.dequantizeQ5K(dst)
	case GGMLTypeQ6_K:
		s.dequantizeQ6K(dst)
	}
}

func (s *MmapStorage) dequantizeF32(dst []float32) {
	for i := range s.length {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(s.data[i*4 : i*4+4]))
	}
}

func (s *MmapStorage) dequantizeF16(dst []float32) {
	for i := range s.length {
		bits := binary.LittleEndian.Uint16(s.data[i*2 : i*2+2])
		dst[i] = float16.FromBits(bits).ToFloat32()
	}
}

func (s *MmapStorage) dequantizeBF16(dst []float32) {
	for i := range s.length {
		bits := binary.LittleEndian.Uint16(s.data[i*2 : i*2+2])
		// BF16: top 16 bits of float32. Shift left 16 to reconstruct.
		dst[i] = math.Float32frombits(uint32(bits) << 16)
	}
}

func (s *MmapStorage) dequantizeQ4_0(dst []float32) {
	const blockSize = 32
	const halfBlock = blockSize / 2
	const blockBytes = 18
	nBlocks := (s.length + blockSize - 1) / blockSize

	for bi := range nBlocks {
		off := bi * blockBytes
		scale := float16.FromBits(binary.LittleEndian.Uint16(s.data[off : off+2])).ToFloat32()
		baseIdx := bi * blockSize

		for j := range halfBlock {
			packed := s.data[off+2+j]
			q0 := int(packed&0x0F) - 8
			q1 := int(packed>>4) - 8

			if idx := baseIdx + j; idx < s.length {
				dst[idx] = float32(q0) * scale
			}
			if idx := baseIdx + j + halfBlock; idx < s.length {
				dst[idx] = float32(q1) * scale
			}
		}
	}
}

func (s *MmapStorage) dequantizeQ8_0(dst []float32) {
	const blockSize = 32
	const blockBytes = 34 // 2 bytes fp16 scale + 32 bytes int8
	nBlocks := (s.length + blockSize - 1) / blockSize

	for bi := range nBlocks {
		off := bi * blockBytes
		scale := float16.FromBits(binary.LittleEndian.Uint16(s.data[off : off+2])).ToFloat32()
		baseIdx := bi * blockSize

		for j := range blockSize {
			idx := baseIdx + j
			if idx >= s.length {
				break
			}
			dst[idx] = float32(int8(s.data[off+2+j])) * scale
		}
	}
}

// dequantizeQ4K decodes Q4_K super-blocks (256 elements, 144 bytes each).
// Layout per super-block:
//
//	bytes 0-1:   fp16 d (super-block scale)
//	bytes 2-3:   fp16 dmin (super-block min)
//	bytes 4-15:  12 bytes packed sub-block scales/mins (6-bit each, 8 sub-blocks)
//	bytes 16-143: 128 bytes packed 4-bit quantized values (256 values, 2 per byte)
func (s *MmapStorage) dequantizeQ4K(dst []float32) {
	const superBlockSize = 256
	const superBlockBytes = 144
	nBlocks := (s.length + superBlockSize - 1) / superBlockSize

	for bi := range nBlocks {
		off := bi * superBlockBytes
		d := float16.FromBits(binary.LittleEndian.Uint16(s.data[off : off+2])).ToFloat32()
		dmin := float16.FromBits(binary.LittleEndian.Uint16(s.data[off+2 : off+4])).ToFloat32()

		// Decode 8 sub-block scales and mins from 12 bytes.
		scales := s.data[off+4 : off+16]
		var sc [8]float32
		var mn [8]float32
		for i := range 8 {
			var scBits, mnBits uint8
			if i < 4 {
				scBits = scales[i] & 0x3F
				mnBits = scales[i+4] & 0x3F
			} else {
				scBits = (scales[i+4] & 0x0F) | ((scales[i-4] >> 6) << 4)
				mnBits = (scales[i+4] >> 4) | ((scales[i] >> 6) << 4)
			}
			sc[i] = d * float32(scBits)
			mn[i] = dmin * float32(mnBits)
		}

		// Decode 256 values from 128 packed bytes.
		qdata := s.data[off+16 : off+superBlockBytes]
		baseIdx := bi * superBlockSize
		for j := range 128 {
			packed := qdata[j]
			subBlock0 := (j * 2) / superBlockSize * 4           // sub-block for low pair
			subIdx0 := (j * 2) % superBlockSize / (superBlockSize / 8) // sub-block index
			_ = subBlock0
			_ = subIdx0

			// Simpler indexing: each sub-block covers 32 elements
			elemIdx0 := baseIdx + j*2
			elemIdx1 := baseIdx + j*2 + 1
			sb := j / 16 // sub-block index (0-7): 16 bytes per sub-block, 32 values

			q0 := float32(packed & 0x0F)
			q1 := float32(packed >> 4)

			if elemIdx0 < s.length {
				dst[elemIdx0] = q0*sc[sb] - mn[sb]
			}
			if elemIdx1 < s.length {
				dst[elemIdx1] = q1*sc[sb] - mn[sb]
			}
		}
	}
}

// dequantizeQ5K decodes Q5_K super-blocks (256 elements, 176 bytes each).
func (s *MmapStorage) dequantizeQ5K(dst []float32) {
	const superBlockSize = 256
	const superBlockBytes = 176
	nBlocks := (s.length + superBlockSize - 1) / superBlockSize

	for bi := range nBlocks {
		off := bi * superBlockBytes
		d := float16.FromBits(binary.LittleEndian.Uint16(s.data[off : off+2])).ToFloat32()
		dmin := float16.FromBits(binary.LittleEndian.Uint16(s.data[off+2 : off+4])).ToFloat32()

		// Decode sub-block scales/mins from 12 bytes (same layout as Q4_K).
		scales := s.data[off+4 : off+16]
		var sc [8]float32
		var mn [8]float32
		for i := range 8 {
			var scBits, mnBits uint8
			if i < 4 {
				scBits = scales[i] & 0x3F
				mnBits = scales[i+4] & 0x3F
			} else {
				scBits = (scales[i+4] & 0x0F) | ((scales[i-4] >> 6) << 4)
				mnBits = (scales[i+4] >> 4) | ((scales[i] >> 6) << 4)
			}
			sc[i] = d * float32(scBits)
			mn[i] = dmin * float32(mnBits)
		}

		// Q5_K: 128 bytes low nibbles (ql) + 32 bytes high bits (qh)
		ql := s.data[off+16 : off+144]
		qh := s.data[off+144 : off+176]
		baseIdx := bi * superBlockSize

		for j := range 128 {
			packed := ql[j]
			sb := j / 16

			q0Low := packed & 0x0F
			q1Low := packed >> 4

			// High bit from qh
			hByte := qh[j/4]
			hBit0 := (hByte >> (uint(j%4) * 2)) & 1
			hBit1 := (hByte >> (uint(j%4)*2 + 1)) & 1

			q0 := float32(q0Low | (hBit0 << 4))
			q1 := float32(q1Low | (hBit1 << 4))

			elemIdx0 := baseIdx + j*2
			elemIdx1 := baseIdx + j*2 + 1

			if elemIdx0 < s.length {
				dst[elemIdx0] = q0*sc[sb] - mn[sb]
			}
			if elemIdx1 < s.length {
				dst[elemIdx1] = q1*sc[sb] - mn[sb]
			}
		}
	}
}

// dequantizeQ6K decodes Q6_K super-blocks (256 elements, 210 bytes each).
// Layout: 128 bytes ql + 64 bytes qh + 16 bytes scales + 2 bytes d
func (s *MmapStorage) dequantizeQ6K(dst []float32) {
	const superBlockSize = 256
	const superBlockBytes = 210
	nBlocks := (s.length + superBlockSize - 1) / superBlockSize

	for bi := range nBlocks {
		off := bi * superBlockBytes
		ql := s.data[off : off+128]
		qh := s.data[off+128 : off+192]
		scaleBytes := s.data[off+192 : off+208]
		d := float16.FromBits(binary.LittleEndian.Uint16(s.data[off+208 : off+210])).ToFloat32()

		// 16 int8 sub-block scales
		var sc [16]float32
		for i := range 16 {
			sc[i] = d * float32(int8(scaleBytes[i]))
		}

		baseIdx := bi * superBlockSize

		// Decode 256 values. Q6_K packs 6 bits per value:
		// - ql: lower 4 bits (128 bytes for 256 values, 2 per byte)
		// - qh: upper 2 bits (64 bytes for 256 values, 4 per byte)
		for j := range 128 {
			packedLow := ql[j]
			q0Low := packedLow & 0x0F
			q1Low := packedLow >> 4

			// Each qh byte holds 2-bit high parts for 4 values
			qhByte := qh[j/2]
			var q0High, q1High byte
			if j%2 == 0 {
				q0High = (qhByte >> 0) & 0x03
				q1High = (qhByte >> 2) & 0x03
			} else {
				q0High = (qhByte >> 4) & 0x03
				q1High = (qhByte >> 6) & 0x03
			}

			q0 := int8(q0Low|(q0High<<4)) - 32
			q1 := int8(q1Low|(q1High<<4)) - 32

			// Sub-block: 16 values each, so 16 sub-blocks
			sb0 := (j * 2) / 16
			sb1 := (j*2 + 1) / 16

			elemIdx0 := baseIdx + j*2
			elemIdx1 := baseIdx + j*2 + 1

			if elemIdx0 < s.length {
				dst[elemIdx0] = float32(q0) * sc[sb0]
			}
			if elemIdx1 < s.length {
				dst[elemIdx1] = float32(q1) * sc[sb1]
			}
		}
	}
}

// dequantizeQ4_1 decodes Q4_1 blocks: 32 elements per block, 20 bytes each.
// Layout: 2 bytes fp16 scale + 2 bytes fp16 min + 16 bytes packed nibbles.
func (s *MmapStorage) dequantizeQ4_1(dst []float32) {
	const blockSize = 32
	const blockBytes = 20
	nBlocks := (s.length + blockSize - 1) / blockSize
	for bi := range nBlocks {
		off := bi * blockBytes
		d := float16.FromBits(binary.LittleEndian.Uint16(s.data[off : off+2])).ToFloat32()
		m := float16.FromBits(binary.LittleEndian.Uint16(s.data[off+2 : off+4])).ToFloat32()
		for j := range blockSize / 2 {
			b := s.data[off+4+j]
			idx0 := bi*blockSize + j
			idx1 := bi*blockSize + j + blockSize/2
			if idx0 < s.length {
				dst[idx0] = float32(b&0x0F)*d + m
			}
			if idx1 < s.length {
				dst[idx1] = float32(b>>4)*d + m
			}
		}
	}
}

// dequantizeQ5_0 decodes Q5_0 blocks: 32 elements per block, 22 bytes each.
// Layout: 2 bytes fp16 scale + 4 bytes high bits + 16 bytes low nibbles.
func (s *MmapStorage) dequantizeQ5_0(dst []float32) {
	const blockSize = 32
	const halfBlock = blockSize / 2
	const blockBytes = 22
	nBlocks := (s.length + blockSize - 1) / blockSize
	for bi := range nBlocks {
		off := bi * blockBytes
		d := float16.FromBits(binary.LittleEndian.Uint16(s.data[off : off+2])).ToFloat32()
		qh := binary.LittleEndian.Uint32(s.data[off+2 : off+6])
		for j := range halfBlock {
			b := s.data[off+6+j]
			lo := int(b & 0x0F)
			hi := int(b >> 4)
			// Add high bit from qh
			if qh&(1<<uint(j)) != 0 {
				lo |= 16
			}
			if qh&(1<<uint(j+halfBlock)) != 0 {
				hi |= 16
			}
			idx0 := bi*blockSize + j
			idx1 := bi*blockSize + j + halfBlock
			if idx0 < s.length {
				dst[idx0] = float32(lo-16) * d
			}
			if idx1 < s.length {
				dst[idx1] = float32(hi-16) * d
			}
		}
	}
}

// dequantizeQ5_1 decodes Q5_1 blocks: 32 elements per block, 24 bytes each.
// Layout: 2 bytes fp16 scale + 2 bytes fp16 min + 4 bytes high bits + 16 bytes low nibbles.
func (s *MmapStorage) dequantizeQ5_1(dst []float32) {
	const blockSize = 32
	const halfBlock = blockSize / 2
	const blockBytes = 24
	nBlocks := (s.length + blockSize - 1) / blockSize
	for bi := range nBlocks {
		off := bi * blockBytes
		d := float16.FromBits(binary.LittleEndian.Uint16(s.data[off : off+2])).ToFloat32()
		m := float16.FromBits(binary.LittleEndian.Uint16(s.data[off+2 : off+4])).ToFloat32()
		qh := binary.LittleEndian.Uint32(s.data[off+4 : off+8])
		for j := range halfBlock {
			b := s.data[off+8+j]
			lo := int(b & 0x0F)
			hi := int(b >> 4)
			if qh&(1<<uint(j)) != 0 {
				lo |= 16
			}
			if qh&(1<<uint(j+halfBlock)) != 0 {
				hi |= 16
			}
			idx0 := bi*blockSize + j
			idx1 := bi*blockSize + j + halfBlock
			if idx0 < s.length {
				dst[idx0] = float32(lo)*d + m
			}
			if idx1 < s.length {
				dst[idx1] = float32(hi)*d + m
			}
		}
	}
}

// Ensure MmapStorage implements Storage[float32].
var _ Storage[float32] = (*MmapStorage)(nil)

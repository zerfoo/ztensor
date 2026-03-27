package tensor

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/device"
)

// IQ3_S format: importance-weighted 3-bit quantization with super-blocks of 256 values.
// Each super-block is 110 bytes:
//   - 2 bytes: fp16 d (super-block scale)
//   - 64 bytes: qs (low 8 bits of grid indices, 4 groups of 8 values each mapped by 2 bytes)
//   - 8 bytes: qh (high 1 bit of grid index per group of 4)
//   - 32 bytes: signs (1 sign bit per value, 256/8 = 32)
//   - 4 bytes: scales (4-bit sub-block scales for 8 sub-blocks of 32, packed into 4 bytes)
//
// Each grid index (9 bits) maps to 4 values from the set {1, 3, 5, 7} via iq3sGrid.
// Signs are applied per-value.
//
// Reference: llama.cpp ggml-quants.c dequantize_row_iq3_s / block_iq3_s
const (
	iq3SSuperBlockSize = 256
	iq3SBlockBytes     = 110
)

// DequantizeIQ3S dequantizes one IQ3_S super-block (110 bytes) into 256 float32 values.
// This matches llama.cpp's dequantize_row_iq3_s.
func DequantizeIQ3S(raw []byte, dst []float32) {
	d := float16.FromBits(binary.LittleEndian.Uint16(raw[0:2])).ToFloat32()
	qs := raw[2:66]    // 64 bytes: packed grid indices (low 8 bits)
	qh := raw[66:74]   // 8 bytes: high bit of grid indices
	signs := raw[74:106] // 32 bytes: sign bits
	scales := raw[106:110] // 4 bytes: 4-bit sub-block scales

	qsIdx := 0
	signIdx := 0

	for i := range 8 { // 8 sub-blocks of 32 values
		// Extract 4-bit scale for this sub-block.
		var sc uint8
		if i%2 == 0 {
			sc = scales[i/2] & 0x0F
		} else {
			sc = scales[i/2] >> 4
		}
		subScale := d * float32(1+2*int(sc))

		outOff := i * 32
		qhByte := qh[i]

		for j := range 4 { // 4 groups of 8 values per sub-block
			// Build 9-bit grid index: 8 low bits from qs, 1 high bit from qh.
			gridIdx := uint16(qs[qsIdx]) | (uint16((qhByte>>(2*uint(j)))&1) << 8) //nolint:gosimple
			qsIdx++

			grid := iq3sGrid[gridIdx]
			signByte := signs[signIdx]
			signIdx++

			for k := range 4 {
				val := subScale * float32(grid[k])
				if signByte&(1<<uint(k)) != 0 {
					val = -val
				}
				dst[outOff+j*8+k] = val
			}

			// Second half of the group: next qs byte.
			gridIdx2 := uint16(qs[qsIdx]) | (uint16((qhByte>>(2*uint(j)+1))&1) << 8) //nolint:gosimple
			qsIdx++

			grid2 := iq3sGrid[gridIdx2]

			for k := range 4 {
				val := subScale * float32(grid2[k])
				if signByte&(1<<uint(k+4)) != 0 {
					val = -val
				}
				dst[outOff+j*8+4+k] = val
			}
		}
	}
}

// IQ3SStorage holds IQ3_S quantized tensor data on CPU.
type IQ3SStorage struct {
	raw []byte // raw super-block data
	len int    // number of logical float32 elements

	gpuPtr      unsafe.Pointer
	gpuByteSize int
	gpuDeviceID int
}

// NewIQ3SStorageFromRaw creates IQ3SStorage from raw super-block data.
func NewIQ3SStorageFromRaw(raw []byte, numElements int) (*IQ3SStorage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nBlocks := (numElements + iq3SSuperBlockSize - 1) / iq3SSuperBlockSize
	need := nBlocks * iq3SBlockBytes
	if len(raw) < need {
		return nil, fmt.Errorf("IQ3_S raw data too short: need %d bytes for %d blocks, got %d", need, nBlocks, len(raw))
	}
	data := make([]byte, need)
	copy(data, raw[:need])
	return &IQ3SStorage{raw: data, len: numElements}, nil
}

// Dequantize unpacks all IQ3_S super-blocks into dst.
func (q *IQ3SStorage) Dequantize(dst []float32) {
	nBlocks := (q.len + iq3SSuperBlockSize - 1) / iq3SSuperBlockSize
	for bi := range nBlocks {
		blockRaw := q.raw[bi*iq3SBlockBytes : (bi+1)*iq3SBlockBytes]
		off := bi * iq3SSuperBlockSize
		remaining := q.len - off
		if remaining >= iq3SSuperBlockSize {
			DequantizeIQ3S(blockRaw, dst[off:off+iq3SSuperBlockSize])
		} else {
			var tmp [iq3SSuperBlockSize]float32
			DequantizeIQ3S(blockRaw, tmp[:])
			copy(dst[off:], tmp[:remaining])
		}
	}
}

// Len returns the number of logical float32 elements.
func (q *IQ3SStorage) Len() int { return q.len }

// Slice dequantizes and returns all elements as a float32 slice.
func (q *IQ3SStorage) Slice() []float32 {
	dst := make([]float32, q.len)
	q.Dequantize(dst)
	return dst
}

// Set panics because IQ3SStorage is immutable.
func (q *IQ3SStorage) Set(_ []float32) { panic("IQ3SStorage is immutable") }

// DeviceType returns device.CPU.
func (q *IQ3SStorage) DeviceType() device.Type { return device.CPU }

// RawBytes returns the raw IQ3_S super-block data for GPU upload.
func (q *IQ3SStorage) RawBytes() []byte { return q.raw }

// NumBlocks returns the number of IQ3_S super-blocks.
func (q *IQ3SStorage) NumBlocks() int {
	return (q.len + iq3SSuperBlockSize - 1) / iq3SSuperBlockSize
}

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw bytes.
func (q *IQ3SStorage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	q.gpuPtr = ptr
	q.gpuByteSize = byteSize
	q.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
func (q *IQ3SStorage) GPUPtr() (unsafe.Pointer, int, int) {
	return q.gpuPtr, q.gpuByteSize, q.gpuDeviceID
}

var _ Storage[float32] = (*IQ3SStorage)(nil)

// iq3sDequantizer wraps IQ3_S dequantization for the quant registry.
type iq3sDequantizer struct{}

func (iq3sDequantizer) Dequantize(src []byte, dst []float32) error {
	n := len(dst)
	s, err := NewIQ3SStorageFromRaw(src, n)
	if err != nil {
		return err
	}
	s.Dequantize(dst)
	return nil
}

func (iq3sDequantizer) BlockSize() int     { return iq3SSuperBlockSize }
func (iq3sDequantizer) BitsPerWeight() int { return 3 }

func init() {
	RegisterQuantType("IQ3_S", iq3sDequantizer{})
}

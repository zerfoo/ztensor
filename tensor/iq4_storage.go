package tensor

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/device"
)

// IQ4_NL format: non-linear 4-bit quantization with a 16-entry lookup table.
// Each block contains 32 values:
//   - 2 bytes: fp16 d (block scale)
//   - 16 bytes: 32 x 4-bit nibbles packed into pairs
//
// Dequantization: output[i] = table[nibble[i]] * scale
//
// Reference: llama.cpp ggml-quants.c dequantize_row_iq4_nl
const (
	iq4NLBlockSize = 32
	iq4NLBlockBytes = 18 // 2 (fp16 scale) + 16 (packed nibbles)
)

// IQ4NLTable is the non-linear 4-bit quantization lookup table.
// These 16 values are the reconstruction points for the IQ4_NL format.
// Reference: llama.cpp kvalues_iq4nl in ggml-quants.c
var IQ4NLTable = [16]float32{
	-1.0, -0.6961928, -0.5250730, -0.3947555,
	-0.2831612, -0.1790887, -0.0805542, 0.0,
	0.0805542, 0.1790887, 0.2831612, 0.3947555,
	0.5250730, 0.6961928, 1.0, 1.3312578,
}

// DequantizeIQ4NL dequantizes one IQ4_NL block (18 bytes) into 32 float32 values.
func DequantizeIQ4NL(raw []byte, dst []float32) {
	d := float16.FromBits(binary.LittleEndian.Uint16(raw[0:2])).ToFloat32()
	qs := raw[2:18] // 16 bytes of packed nibbles

	for i := range 16 {
		lo := qs[i] & 0x0F
		hi := qs[i] >> 4
		dst[2*i] = d * IQ4NLTable[lo]
		dst[2*i+1] = d * IQ4NLTable[hi]
	}
}

// IQ4NLStorage holds IQ4_NL quantized tensor data on CPU.
type IQ4NLStorage struct {
	raw []byte // raw block data
	len int    // number of logical float32 elements

	// GPU-resident copy of the raw bytes (optional).
	gpuPtr      unsafe.Pointer
	gpuByteSize int
	gpuDeviceID int
}

// NewIQ4NLStorageFromRaw creates IQ4NLStorage from raw block data.
func NewIQ4NLStorageFromRaw(raw []byte, numElements int) (*IQ4NLStorage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nBlocks := (numElements + iq4NLBlockSize - 1) / iq4NLBlockSize
	need := nBlocks * iq4NLBlockBytes
	if len(raw) < need {
		return nil, fmt.Errorf("IQ4_NL raw data too short: need %d bytes for %d blocks, got %d", need, nBlocks, len(raw))
	}
	data := make([]byte, need)
	copy(data, raw[:need])
	return &IQ4NLStorage{raw: data, len: numElements}, nil
}

// Dequantize unpacks all IQ4_NL blocks into dst.
func (q *IQ4NLStorage) Dequantize(dst []float32) {
	nBlocks := (q.len + iq4NLBlockSize - 1) / iq4NLBlockSize
	for bi := range nBlocks {
		blockRaw := q.raw[bi*iq4NLBlockBytes : (bi+1)*iq4NLBlockBytes]
		off := bi * iq4NLBlockSize
		remaining := q.len - off
		if remaining >= iq4NLBlockSize {
			DequantizeIQ4NL(blockRaw, dst[off:off+iq4NLBlockSize])
		} else {
			var tmp [iq4NLBlockSize]float32
			DequantizeIQ4NL(blockRaw, tmp[:])
			copy(dst[off:], tmp[:remaining])
		}
	}
}

// Len returns the number of logical float32 elements.
func (q *IQ4NLStorage) Len() int { return q.len }

// Slice dequantizes and returns all elements as a float32 slice.
func (q *IQ4NLStorage) Slice() []float32 { dst := make([]float32, q.len); q.Dequantize(dst); return dst }

// Set panics because IQ4NLStorage is immutable.
func (q *IQ4NLStorage) Set(_ []float32) { panic("IQ4NLStorage is immutable") }

// DeviceType returns device.CPU.
func (q *IQ4NLStorage) DeviceType() device.Type { return device.CPU }

// RawBytes returns the raw IQ4_NL block data for GPU upload.
func (q *IQ4NLStorage) RawBytes() []byte { return q.raw }

// NumBlocks returns the number of IQ4_NL blocks.
func (q *IQ4NLStorage) NumBlocks() int {
	return (q.len + iq4NLBlockSize - 1) / iq4NLBlockSize
}

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw bytes.
func (q *IQ4NLStorage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	q.gpuPtr = ptr
	q.gpuByteSize = byteSize
	q.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
// Returns nil if no GPU copy exists.
func (q *IQ4NLStorage) GPUPtr() (unsafe.Pointer, int, int) {
	return q.gpuPtr, q.gpuByteSize, q.gpuDeviceID
}

// MergeIQ4NLStorage concatenates multiple IQ4NLStorage objects into one.
func MergeIQ4NLStorage(storages ...*IQ4NLStorage) *IQ4NLStorage {
	totalBytes := 0
	totalLen := 0
	for _, s := range storages {
		totalBytes += len(s.raw)
		totalLen += s.len
	}
	raw := make([]byte, 0, totalBytes)
	for _, s := range storages {
		raw = append(raw, s.raw...)
	}
	return &IQ4NLStorage{
		raw: raw,
		len: totalLen,
	}
}

var _ Storage[float32] = (*IQ4NLStorage)(nil)

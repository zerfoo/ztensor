package tensor

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/device"
)

// Q5_0 format: blocks of 32 values.
// Each block is 22 bytes:
//   - 2 bytes: fp16 d (block scale)
//   - 4 bytes: qh (32 high bits, one per element)
//   - 16 bytes: qs (packed nibbles, two 4-bit values per byte)
//
// Reference: llama.cpp ggml-quants.c dequantize_row_q5_0
const (
	q5_0BlockSize  = 32
	q5_0BlockBytes = 22
)

// DequantizeQ5_0 dequantizes one Q5_0 block (22 bytes) into 32 float32 values.
// This matches llama.cpp's dequantize_row_q5_0.
func DequantizeQ5_0(raw []byte, dst []float32) {
	d := float16.FromBits(binary.LittleEndian.Uint16(raw[0:2])).ToFloat32()
	qh := binary.LittleEndian.Uint32(raw[2:6])
	qs := raw[6:22]

	for j := range 16 {
		packed := qs[j]
		low4 := packed & 0x0F
		high4 := packed >> 4

		xh0 := uint8(((qh >> j) & 1) << 4)
		x0 := int(low4|xh0) - 16

		xh1 := uint8(((qh >> (j + 16)) & 1) << 4)
		x1 := int(high4|xh1) - 16

		dst[j] = d * float32(x0)
		dst[j+16] = d * float32(x1)
	}
}

// Q5_0Storage holds Q5_0 quantized tensor data on CPU.
type Q5_0Storage struct {
	raw []byte
	len int

	// GPU-resident copy of the raw bytes (optional).
	// Set by GPUEngine.UploadWeights to avoid per-op H2D copies.
	gpuPtr      unsafe.Pointer
	gpuByteSize int
	gpuDeviceID int
}

// NewQ5_0StorageFromRaw creates Q5_0Storage from raw block data.
func NewQ5_0StorageFromRaw(raw []byte, numElements int) (*Q5_0Storage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nBlocks := (numElements + q5_0BlockSize - 1) / q5_0BlockSize
	need := nBlocks * q5_0BlockBytes
	if len(raw) < need {
		return nil, fmt.Errorf("Q5_0 raw data too short: need %d bytes for %d blocks, got %d", need, nBlocks, len(raw))
	}
	data := make([]byte, need)
	copy(data, raw[:need])
	return &Q5_0Storage{raw: data, len: numElements}, nil
}

// Dequantize unpacks all Q5_0 blocks into dst.
func (q *Q5_0Storage) Dequantize(dst []float32) {
	nBlocks := (q.len + q5_0BlockSize - 1) / q5_0BlockSize
	for bi := range nBlocks {
		blockRaw := q.raw[bi*q5_0BlockBytes : (bi+1)*q5_0BlockBytes]
		off := bi * q5_0BlockSize
		remaining := q.len - off
		if remaining >= q5_0BlockSize {
			DequantizeQ5_0(blockRaw, dst[off:off+q5_0BlockSize])
		} else {
			var tmp [q5_0BlockSize]float32
			DequantizeQ5_0(blockRaw, tmp[:])
			copy(dst[off:], tmp[:remaining])
		}
	}
}

// Len returns the number of logical float32 elements.
func (q *Q5_0Storage) Len() int { return q.len }

// Slice dequantizes and returns all elements as a float32 slice.
func (q *Q5_0Storage) Slice() []float32 { dst := make([]float32, q.len); q.Dequantize(dst); return dst }

// Set panics because Q5_0Storage is immutable.
func (q *Q5_0Storage) Set(_ []float32) { panic("Q5_0Storage is immutable") }

// DeviceType returns device.CPU.
func (q *Q5_0Storage) DeviceType() device.Type { return device.CPU }

// RawBytes returns the raw Q5_0 block data for GPU upload.
// The layout is contiguous blocks, each 22 bytes.
func (q *Q5_0Storage) RawBytes() []byte { return q.raw }

// RawBytesGPU returns Q5_0 data in a separated GPU-optimized layout.
// Instead of interleaved 22-byte blocks [d(2) | qh(4) | qs(16)], the data
// is separated into three contiguous regions:
//
//	[all scales (fp16, 2B each)] pad to 16B
//	[all qh values (uint32, 4B each)] pad to 16B
//	[all qs values (16B each)]
//
// This ensures natural alignment: fp16 scales at 2-byte boundaries,
// uint32 qh at 4-byte boundaries. Eliminates the byte-wise __ldg loads
// required for the interleaved layout on ARM64 Grace Hopper.
func (q *Q5_0Storage) RawBytesGPU() []byte {
	nBlocks := (q.len + q5_0BlockSize - 1) / q5_0BlockSize
	scaleBytes := nBlocks * 2
	paddedScaleBytes := (scaleBytes + 15) &^ 15 // align to 16B
	qhBytes := nBlocks * 4
	paddedQhBytes := (qhBytes + 15) &^ 15 // align to 16B
	qsBytes := nBlocks * 16
	totalSize := paddedScaleBytes + paddedQhBytes + qsBytes

	out := make([]byte, totalSize)

	// Region 1: all fp16 scales contiguously.
	for i := range nBlocks {
		blk := q.raw[i*q5_0BlockBytes : (i+1)*q5_0BlockBytes]
		copy(out[i*2:i*2+2], blk[0:2]) // fp16 d
	}
	// Region 2: all uint32 qh values contiguously.
	qhOff := paddedScaleBytes
	for i := range nBlocks {
		blk := q.raw[i*q5_0BlockBytes : (i+1)*q5_0BlockBytes]
		copy(out[qhOff+i*4:qhOff+i*4+4], blk[2:6]) // uint32 qh
	}
	// Region 3: all packed nibbles contiguously.
	qsOff := paddedScaleBytes + paddedQhBytes
	for i := range nBlocks {
		blk := q.raw[i*q5_0BlockBytes : (i+1)*q5_0BlockBytes]
		copy(out[qsOff+i*16:qsOff+i*16+16], blk[6:22]) // 16 bytes qs
	}
	return out
}

// Q5_0GPUQhOffset returns the byte offset where the qh region starts
// in the RawBytesGPU layout, given the total number of blocks.
func Q5_0GPUQhOffset(totalBlocks int) int {
	scaleBytes := totalBlocks * 2
	return (scaleBytes + 15) &^ 15
}

// Q5_0GPUQsOffset returns the byte offset where the qs region starts
// in the RawBytesGPU layout, given the total number of blocks.
func Q5_0GPUQsOffset(totalBlocks int) int {
	scaleBytes := totalBlocks * 2
	paddedScaleBytes := (scaleBytes + 15) &^ 15
	qhBytes := totalBlocks * 4
	paddedQhBytes := (qhBytes + 15) &^ 15
	return paddedScaleBytes + paddedQhBytes
}

// NumBlocks returns the number of Q5_0 blocks.
func (q *Q5_0Storage) NumBlocks() int {
	return (q.len + q5_0BlockSize - 1) / q5_0BlockSize
}

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw bytes.
func (q *Q5_0Storage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	q.gpuPtr = ptr
	q.gpuByteSize = byteSize
	q.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
// Returns nil if no GPU copy exists.
func (q *Q5_0Storage) GPUPtr() (unsafe.Pointer, int, int) {
	return q.gpuPtr, q.gpuByteSize, q.gpuDeviceID
}

var _ Storage[float32] = (*Q5_0Storage)(nil)

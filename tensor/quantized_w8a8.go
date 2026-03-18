package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/ztensor/device"
)

// W8A8 (Weight 8-bit, Activation 8-bit) mixed-precision format.
//
// Both weights and activations are quantized to symmetric INT8 with per-group
// scaling. Accumulation is performed in FP32 for numerical stability.
//
// Each group of w8a8GroupSize elements stores:
//   - scale (float32): maps [-127, 127] back to the original float range
//   - data  ([]int8):  symmetric quantized values
//
// Quantization: q = clamp(round(x / scale), -127, 127)
// Dequantization: x = q * scale
//
// This is the standard W8A8 scheme used by SmoothQuant and similar methods.

const w8a8GroupSize = 32

// w8a8Group holds the metadata and quantized data for one group.
type w8a8Group struct {
	scale float32
	data  [w8a8GroupSize]int8
}

// W8A8Storage holds W8A8 symmetric INT8 quantized tensor data on CPU.
type W8A8Storage struct {
	groups []w8a8Group
	len    int // number of logical float32 elements

	gpuPtr      unsafe.Pointer
	gpuByteSize int
	gpuDeviceID int
}

// QuantizeW8A8 quantizes a float32 slice into W8A8 symmetric INT8 format.
// groupSize is fixed at 32 elements per group.
func QuantizeW8A8(src []float32) *W8A8Storage {
	n := len(src)
	nGroups := (n + w8a8GroupSize - 1) / w8a8GroupSize
	groups := make([]w8a8Group, nGroups)

	for gi := range nGroups {
		offset := gi * w8a8GroupSize
		end := offset + w8a8GroupSize
		if end > n {
			end = n
		}

		// Find absmax for symmetric quantization.
		var absMax float32
		for i := offset; i < end; i++ {
			v := src[i]
			if v < 0 {
				v = -v
			}
			if v > absMax {
				absMax = v
			}
		}

		// Symmetric scale: maps [-absMax, absMax] to [-127, 127].
		var scale float32
		if absMax > 0 {
			scale = absMax / 127.0
		}
		groups[gi].scale = scale

		var invScale float32
		if scale > 0 {
			invScale = 1.0 / scale
		}

		for i := offset; i < end; i++ {
			q := int(math.Round(float64(src[i] * invScale)))
			groups[gi].data[i-offset] = int8(clampInt(q, -127, 127))
		}
	}

	return &W8A8Storage{groups: groups, len: n}
}

// Dequantize unpacks W8A8 groups into dst. len(dst) must be >= s.Len().
func (s *W8A8Storage) Dequantize(dst []float32) {
	for gi, g := range s.groups {
		offset := gi * w8a8GroupSize
		end := offset + w8a8GroupSize
		if end > s.len {
			end = s.len
		}
		for i := offset; i < end; i++ {
			dst[i] = float32(g.data[i-offset]) * g.scale
		}
	}
}

// DequantizeBlock unpacks a single W8A8 group into a 32-element buffer.
func (s *W8A8Storage) DequantizeBlock(blockIdx int, dst *[w8a8GroupSize]float32) {
	g := s.groups[blockIdx]
	for j := range w8a8GroupSize {
		dst[j] = float32(g.data[j]) * g.scale
	}
}

// Len returns the number of logical float32 elements.
func (s *W8A8Storage) Len() int { return s.len }

// NumGroups returns the number of quantization groups.
func (s *W8A8Storage) NumGroups() int { return len(s.groups) }

// GroupSize returns the number of elements per group (always 32).
func (s *W8A8Storage) GroupSize() int { return w8a8GroupSize }

// ByteSize returns the raw byte size of the quantized data.
// Each group: 4 bytes float32 scale + 32 bytes int8 data = 36 bytes.
func (s *W8A8Storage) ByteSize() int { return len(s.groups) * 36 }

// Slice returns a dequantized float32 view of the data.
func (s *W8A8Storage) Slice() []float32 {
	dst := make([]float32, s.len)
	s.Dequantize(dst)
	return dst
}

// Set is not supported on quantized storage (weights are immutable).
func (s *W8A8Storage) Set(_ []float32) { panic("W8A8Storage is immutable") }

// DeviceType returns device.CPU.
func (s *W8A8Storage) DeviceType() device.Type { return device.CPU }

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw bytes.
func (s *W8A8Storage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	s.gpuPtr = ptr
	s.gpuByteSize = byteSize
	s.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
func (s *W8A8Storage) GPUPtr() (unsafe.Pointer, int, int) {
	return s.gpuPtr, s.gpuByteSize, s.gpuDeviceID
}

// BlockScale returns the float32 scale for group i.
func (s *W8A8Storage) BlockScale(i int) float32 {
	return s.groups[i].scale
}

// BlockQuants returns the int8 quantized values for group i.
func (s *W8A8Storage) BlockQuants(i int) []int8 {
	return s.groups[i].data[:]
}

// RawBytes serializes W8A8 groups as contiguous bytes for GPU upload.
// Each group is 36 bytes: 4 bytes little-endian float32 scale + 32 bytes int8 data.
func (s *W8A8Storage) RawBytes() []byte {
	const blockBytes = 36
	out := make([]byte, len(s.groups)*blockBytes)
	for i, g := range s.groups {
		off := i * blockBytes
		binary.LittleEndian.PutUint32(out[off:off+4], math.Float32bits(g.scale))
		for j, v := range g.data {
			out[off+4+j] = byte(v)
		}
	}
	return out
}

// NewW8A8StorageFromBlocks creates W8A8Storage from pre-decoded block data.
// scales has one entry per group. quants has w8a8GroupSize int8 values per group (flattened).
// numElements is the number of logical float32 elements.
func NewW8A8StorageFromBlocks(scales []float32, quants []int8, numElements int) (*W8A8Storage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nGroups := (numElements + w8a8GroupSize - 1) / w8a8GroupSize
	if len(scales) != nGroups {
		return nil, fmt.Errorf("expected %d scales for %d elements, got %d", nGroups, numElements, len(scales))
	}
	if len(quants) != nGroups*w8a8GroupSize {
		return nil, fmt.Errorf("expected %d quants for %d groups, got %d", nGroups*w8a8GroupSize, nGroups, len(quants))
	}

	groups := make([]w8a8Group, nGroups)
	for i := range nGroups {
		groups[i].scale = scales[i]
		copy(groups[i].data[:], quants[i*w8a8GroupSize:(i+1)*w8a8GroupSize])
	}
	return &W8A8Storage{groups: groups, len: numElements}, nil
}

// GemmW8A8 computes C = A * B where both A and B are W8A8 quantized.
// A has logical shape [M, K], B has shape [K, N], C is float32 [M, N].
// Dequantizes both operands and accumulates in FP32 for numerical stability.
// For the optimized INT8xINT8->FP32 path, use GemmW8A8NT with [N,K] layout.
func GemmW8A8(m, n, k int, a, b *W8A8Storage, c []float32) {
	af32 := a.Slice()
	bf32 := b.Slice()

	for i := range c {
		c[i] = 0
	}

	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += af32[i*k+p] * bf32[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// GemmW8A8NT computes C = A * B^T where A is W8A8 [M,K] and B is W8A8 [N,K].
// B is stored in row-major [N,K] layout; the transpose is implicit.
// Uses INT8xINT8 dot product with FP32 accumulation for numerical stability.
// K must be a multiple of w8a8GroupSize (32).
func GemmW8A8NT(m, n, k int, a, b *W8A8Storage, c []float32) {
	blocksPerRow := k / w8a8GroupSize

	for i := range c {
		c[i] = 0
	}

	for i := range m {
		for j := range n {
			var acc float32
			for bi := range blocksPerRow {
				aGroup := a.groups[i*blocksPerRow+bi]
				bGroup := b.groups[j*blocksPerRow+bi]
				combinedScale := aGroup.scale * bGroup.scale

				if combinedScale == 0 {
					continue
				}

				var dot int32
				for p := range w8a8GroupSize {
					dot += int32(aGroup.data[p]) * int32(bGroup.data[p])
				}
				acc += float32(dot) * combinedScale
			}
			c[i*n+j] = acc
		}
	}
}

// GemmF32W8A8NT computes C = A * B^T where A is float32 [M,K] and B is W8A8 [N,K].
// B is stored in row-major W8A8 format. The "NT" suffix means B is not transposed
// in memory — the caller passes B in its original [N,K] layout.
// K must be a multiple of w8a8GroupSize (32). Falls back to dequant for unaligned K.
func GemmF32W8A8NT(m, n, k int, a []float32, b *W8A8Storage, c []float32) {
	blocksPerRow := k / w8a8GroupSize

	for i := range c {
		c[i] = 0
	}

	var buf [w8a8GroupSize]float32
	for i := range m {
		aRow := a[i*k:]
		for j := range n {
			var sum float32
			for bi := range blocksPerRow {
				blkIdx := j*blocksPerRow + bi
				b.DequantizeBlock(blkIdx, &buf)
				kBase := bi * w8a8GroupSize
				for p := range w8a8GroupSize {
					sum += aRow[kBase+p] * buf[p]
				}
			}
			c[i*n+j] = sum
		}
	}
}

// w8a8Dequantizer wraps W8A8 dequantization for the registry.
// Wire format: 36 bytes per block (4 bytes float32 scale LE + 32 bytes int8 data).
type w8a8Dequantizer struct{}

func (w8a8Dequantizer) Dequantize(src []byte, dst []float32) error {
	n := len(dst)
	if n == 0 {
		return nil
	}
	nGroups := (n + w8a8GroupSize - 1) / w8a8GroupSize
	const blockBytes = 36
	if len(src) < nGroups*blockBytes {
		return errShortData("W8A8", nGroups*blockBytes, len(src))
	}

	scales := make([]float32, nGroups)
	quants := make([]int8, nGroups*w8a8GroupSize)
	for i := range nGroups {
		off := i * blockBytes
		scales[i] = decodeFloat32LE(src[off : off+4])
		for j := range w8a8GroupSize {
			quants[i*w8a8GroupSize+j] = int8(src[off+4+j])
		}
	}
	s, err := NewW8A8StorageFromBlocks(scales, quants, n)
	if err != nil {
		return err
	}
	s.Dequantize(dst)
	return nil
}

func (w8a8Dequantizer) BlockSize() int     { return w8a8GroupSize }
func (w8a8Dequantizer) BitsPerWeight() int { return 8 }

func init() {
	RegisterQuantType("W8A8", w8a8Dequantizer{})
}

// Ensure W8A8Storage implements Storage[float32].
var _ Storage[float32] = (*W8A8Storage)(nil)

package tensor

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/device"
)

// AWQ (Activation-aware Weight Quantization) format: group-wise asymmetric quantization.
//
// Weights are quantized to 4-bit integers in groups (typically 128).
// Each group stores:
//   - scale (FP16): maps quantized range back to float
//   - zero  (FP16): zero-point offset (integer, stored as FP16)
//
// Dequantization: weight_fp = (quant_int4 - zero) * scale
//
// Packed format: 8 INT4 values packed little-endian into one uint32.
// This follows the AutoAWQ / MIT-HAN-Lab convention.

// awqGroup holds the metadata and packed data for one quantization group.
type awqGroup struct {
	scale float16.Float16
	zero  float16.Float16
	data  []uint32 // packed INT4 values: 8 per uint32, little-endian nibbles
}

// AWQStorage holds AWQ group-quantized tensor data on CPU.
type AWQStorage struct {
	groups    []awqGroup
	len       int // number of logical float32 elements
	groupSize int // elements per group

	gpuPtr      unsafe.Pointer
	gpuByteSize int
	gpuDeviceID int
}

// QuantizeAWQ quantizes a float32 slice into AWQ format.
// groupSize is the number of elements per group (typically 128).
// Weights are quantized to 4-bit (INT4, unsigned 0-15).
func QuantizeAWQ(src []float32, groupSize int) *AWQStorage {
	n := len(src)
	nGroups := (n + groupSize - 1) / groupSize
	groups := make([]awqGroup, nGroups)

	const maxVal = 15 // 4-bit unsigned max

	for gi := range nGroups {
		offset := gi * groupSize
		end := offset + groupSize
		if end > n {
			end = n
		}
		groupData := src[offset:end]

		// Find min/max for asymmetric quantization.
		minVal := groupData[0]
		maxV := groupData[0]
		for _, v := range groupData[1:] {
			if v < minVal {
				minVal = v
			}
			if v > maxV {
				maxV = v
			}
		}

		// Compute scale and zero point.
		var scale float32
		rangeV := maxV - minVal
		if rangeV > 0 {
			scale = rangeV / float32(maxVal)
		}
		var zero float32
		if scale > 0 {
			zero = -minVal / scale
		}

		groups[gi].scale = float16.FromFloat32(scale)
		groups[gi].zero = float16.FromFloat32(zero)

		// Quantize and pack: 8 INT4 values per uint32, little-endian nibbles.
		count := end - offset
		packedLen := (count + 7) / 8
		packed := make([]uint32, packedLen)

		scaleF := groups[gi].scale.ToFloat32()
		zeroF := groups[gi].zero.ToFloat32()
		var invScale float32
		if scaleF > 0 {
			invScale = 1.0 / scaleF
		}

		for j := range count {
			v := groupData[j]
			q := int(v*invScale+zeroF+0.5) // round to nearest
			q = clampInt(q, 0, maxVal)

			wordIdx := j / 8
			nibbleIdx := uint(j % 8)
			packed[wordIdx] |= uint32(q&0xF) << (nibbleIdx * 4)
		}
		groups[gi].data = packed
	}

	return &AWQStorage{
		groups:    groups,
		len:       n,
		groupSize: groupSize,
	}
}

// Dequantize unpacks AWQ groups into dst. len(dst) must be >= s.Len().
// Formula: weight_fp = (quant_int4 - zero) * scale
func (s *AWQStorage) Dequantize(dst []float32) {
	for gi, g := range s.groups {
		scale := g.scale.ToFloat32()
		zero := g.zero.ToFloat32()
		offset := gi * s.groupSize
		end := offset + s.groupSize
		if end > s.len {
			end = s.len
		}
		count := end - offset

		for j := range count {
			wordIdx := j / 8
			nibbleIdx := uint(j % 8)
			q := int((g.data[wordIdx] >> (nibbleIdx * 4)) & 0xF)
			dst[offset+j] = (float32(q) - zero) * scale
		}
	}
}

// Len returns the number of logical float32 elements.
func (s *AWQStorage) Len() int { return s.len }

// NumGroups returns the number of quantization groups.
func (s *AWQStorage) NumGroups() int { return len(s.groups) }

// GroupSize returns the number of elements per group.
func (s *AWQStorage) GroupSize() int { return s.groupSize }

// ByteSize returns the raw byte size of the quantized data.
// Each group: packed uint32 data bytes + 2 bytes scale + 2 bytes zero.
func (s *AWQStorage) ByteSize() int {
	total := 0
	for _, g := range s.groups {
		total += len(g.data)*4 + 4 // uint32s * 4 bytes + 2 scale + 2 zero
	}
	return total
}

// Slice returns a dequantized float32 view of the data.
func (s *AWQStorage) Slice() []float32 {
	dst := make([]float32, s.len)
	s.Dequantize(dst)
	return dst
}

// Set is not supported on quantized storage (weights are immutable).
func (s *AWQStorage) Set(_ []float32) { panic("AWQStorage is immutable") }

// DeviceType returns device.CPU.
func (s *AWQStorage) DeviceType() device.Type { return device.CPU }

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw bytes.
func (s *AWQStorage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	s.gpuPtr = ptr
	s.gpuByteSize = byteSize
	s.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
func (s *AWQStorage) GPUPtr() (unsafe.Pointer, int, int) {
	return s.gpuPtr, s.gpuByteSize, s.gpuDeviceID
}

// NewAWQStorageFromRaw creates AWQStorage from pre-extracted components.
// data is the packed INT4 values (all groups concatenated, 8 nibbles per uint32).
// scales and zeros are FP16 values, one per group.
// numElements is the logical element count.
// groupSize defines the quantization group size.
func NewAWQStorageFromRaw(data []uint32, scales, zeros []float16.Float16, numElements, groupSize int) (*AWQStorage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	if groupSize <= 0 {
		return nil, fmt.Errorf("groupSize must be positive, got %d", groupSize)
	}
	nGroups := (numElements + groupSize - 1) / groupSize
	if len(scales) != nGroups {
		return nil, fmt.Errorf("expected %d scales for %d groups, got %d", nGroups, nGroups, len(scales))
	}
	if len(zeros) != nGroups {
		return nil, fmt.Errorf("expected %d zeros for %d groups, got %d", nGroups, nGroups, len(zeros))
	}

	// Validate total packed data size.
	totalPacked := 0
	for gi := range nGroups {
		count := groupSize
		remaining := numElements - gi*groupSize
		if remaining < groupSize {
			count = remaining
		}
		totalPacked += (count + 7) / 8
	}
	if len(data) < totalPacked {
		return nil, fmt.Errorf("AWQ data too short: need %d uint32s, got %d", totalPacked, len(data))
	}

	groups := make([]awqGroup, nGroups)
	dataOff := 0
	for gi := range nGroups {
		count := groupSize
		remaining := numElements - gi*groupSize
		if remaining < groupSize {
			count = remaining
		}
		packedLen := (count + 7) / 8
		groups[gi].scale = scales[gi]
		groups[gi].zero = zeros[gi]
		groups[gi].data = make([]uint32, packedLen)
		copy(groups[gi].data, data[dataOff:dataOff+packedLen])
		dataOff += packedLen
	}

	return &AWQStorage{
		groups:    groups,
		len:       numElements,
		groupSize: groupSize,
	}, nil
}

// awqDequantizer wraps AWQ dequantization for the registry.
// Wire format: header (4 bytes numElements LE, 4 bytes groupSize LE, 4 bytes nGroups LE),
// then nGroups * (2 bytes scale FP16 + 2 bytes zero FP16), then packed uint32 data.
type awqDequantizer struct{}

func (awqDequantizer) Dequantize(src []byte, dst []float32) error {
	n := len(dst)
	if n == 0 {
		return nil
	}
	// Header: numElements (4), groupSize (4), nGroups (4) = 12 bytes.
	if len(src) < 12 {
		return errShortData("AWQ", 12, len(src))
	}
	numElements := int(decodeUint32LE(src[0:4]))
	groupSize := int(decodeUint32LE(src[4:8]))
	nGroups := int(decodeUint32LE(src[8:12]))
	if numElements != n {
		return fmt.Errorf("tensor: AWQ numElements mismatch: header=%d, dst=%d", numElements, n)
	}
	if groupSize <= 0 || nGroups <= 0 {
		return fmt.Errorf("tensor: AWQ invalid groupSize=%d or nGroups=%d", groupSize, nGroups)
	}

	// Scales and zeros: nGroups * 4 bytes.
	metaBytes := 12 + nGroups*4
	if len(src) < metaBytes {
		return errShortData("AWQ", metaBytes, len(src))
	}
	scales := make([]float16.Float16, nGroups)
	zeros := make([]float16.Float16, nGroups)
	for i := range nGroups {
		off := 12 + i*4
		scales[i] = float16.Float16(decodeUint16LE(src[off : off+2]))
		zeros[i] = float16.Float16(decodeUint16LE(src[off+2 : off+4]))
	}

	// Packed data: uint32 words.
	dataBytes := src[metaBytes:]
	totalPacked := 0
	for gi := range nGroups {
		count := groupSize
		remaining := numElements - gi*groupSize
		if remaining < groupSize {
			count = remaining
		}
		totalPacked += (count + 7) / 8
	}
	if len(dataBytes) < totalPacked*4 {
		return errShortData("AWQ data", totalPacked*4, len(dataBytes))
	}
	packed := make([]uint32, totalPacked)
	for i := range totalPacked {
		packed[i] = decodeUint32LE(dataBytes[i*4 : i*4+4])
	}

	s, err := NewAWQStorageFromRaw(packed, scales, zeros, numElements, groupSize)
	if err != nil {
		return err
	}
	s.Dequantize(dst)
	return nil
}

func (awqDequantizer) BlockSize() int     { return 128 } // typical AWQ group size
func (awqDequantizer) BitsPerWeight() int { return 4 }

func init() {
	RegisterQuantType("AWQ_4", awqDequantizer{})
}

// decodeUint16LE decodes a little-endian uint16 from 2 bytes.
func decodeUint16LE(b []byte) uint16 {
	return uint16(b[0]) | uint16(b[1])<<8
}

// decodeUint32LE decodes a little-endian uint32 from 4 bytes.
func decodeUint32LE(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

// Ensure AWQStorage implements Storage[float32].
var _ Storage[float32] = (*AWQStorage)(nil)

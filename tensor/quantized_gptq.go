package tensor

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/device"
)

// GPTQ format: group-wise asymmetric quantization.
//
// Weights are quantized to 4-bit or 8-bit integers in groups (typically 128).
// Each group stores:
//   - scale (FP16): maps quantized range back to float
//   - zero  (FP16): zero-point offset
//
// Dequantization: weight_fp = (quant_int - zero) * scale
//
// This follows the AutoGPTQ / HuggingFace convention.

// gptqGroup holds the metadata and packed data for one quantization group.
type gptqGroup struct {
	scale float16.Float16
	zero  float16.Float16
	data  []byte // packed quantized values
}

// GPTQStorage holds GPTQ group-quantized tensor data on CPU.
type GPTQStorage struct {
	groups    []gptqGroup
	len       int // number of logical float32 elements
	groupSize int // elements per group
	bits      int // quantization bit width (4 or 8)

	gpuPtr      unsafe.Pointer
	gpuByteSize int
	gpuDeviceID int
}

// QuantizeGPTQ quantizes a float32 slice into GPTQ format.
// groupSize is the number of elements per group (typically 128).
// bits must be 4 or 8.
func QuantizeGPTQ(src []float32, groupSize, bits int) *GPTQStorage {
	n := len(src)
	nGroups := (n + groupSize - 1) / groupSize
	groups := make([]gptqGroup, nGroups)

	maxVal := (1 << bits) - 1 // 15 for 4-bit, 255 for 8-bit

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
			zero = math.Float32frombits(math.Float32bits(minVal/scale)) * -1
			// zero = -minVal / scale, but avoid -0.
			zero = -minVal / scale
		}

		groups[gi].scale = float16.FromFloat32(scale)
		groups[gi].zero = float16.FromFloat32(zero)

		// Quantize and pack.
		count := end - offset
		var packedLen int
		if bits == 4 {
			packedLen = (count + 1) / 2
		} else {
			packedLen = count
		}
		packed := make([]byte, packedLen)

		scaleF := groups[gi].scale.ToFloat32()
		zeroF := groups[gi].zero.ToFloat32()
		var invScale float32
		if scaleF > 0 {
			invScale = 1.0 / scaleF
		}

		for j := range count {
			v := groupData[j]
			q := int(math.Round(float64(v*invScale + zeroF)))
			q = clampInt(q, 0, maxVal)

			if bits == 4 {
				byteIdx := j / 2
				if j%2 == 0 {
					packed[byteIdx] = byte(q & 0xF)
				} else {
					packed[byteIdx] |= byte((q & 0xF) << 4)
				}
			} else {
				packed[j] = byte(q)
			}
		}
		groups[gi].data = packed
	}

	return &GPTQStorage{
		groups:    groups,
		len:       n,
		groupSize: groupSize,
		bits:      bits,
	}
}

// Dequantize unpacks GPTQ groups into dst. len(dst) must be >= s.Len().
// Formula: weight_fp = (quant_int - zero) * scale
func (s *GPTQStorage) Dequantize(dst []float32) {
	for gi, g := range s.groups {
		scale := g.scale.ToFloat32()
		zero := g.zero.ToFloat32()
		offset := gi * s.groupSize
		end := offset + s.groupSize
		if end > s.len {
			end = s.len
		}
		count := end - offset

		if s.bits == 4 {
			for j := range count {
				byteIdx := j / 2
				var q int
				if j%2 == 0 {
					q = int(g.data[byteIdx] & 0xF)
				} else {
					q = int(g.data[byteIdx] >> 4)
				}
				dst[offset+j] = (float32(q) - zero) * scale
			}
		} else {
			for j := range count {
				q := int(g.data[j])
				dst[offset+j] = (float32(q) - zero) * scale
			}
		}
	}
}

// Len returns the number of logical float32 elements.
func (s *GPTQStorage) Len() int { return s.len }

// NumGroups returns the number of quantization groups.
func (s *GPTQStorage) NumGroups() int { return len(s.groups) }

// GroupSize returns the number of elements per group.
func (s *GPTQStorage) GroupSize() int { return s.groupSize }

// Bits returns the quantization bit width (4 or 8).
func (s *GPTQStorage) Bits() int { return s.bits }

// ByteSize returns the raw byte size of the quantized data.
// Each group: packed data bytes + 2 bytes scale + 2 bytes zero.
func (s *GPTQStorage) ByteSize() int {
	total := 0
	for _, g := range s.groups {
		total += len(g.data) + 4 // 2 bytes scale + 2 bytes zero
	}
	return total
}

// Slice returns a dequantized float32 view of the data.
func (s *GPTQStorage) Slice() []float32 {
	dst := make([]float32, s.len)
	s.Dequantize(dst)
	return dst
}

// Set is not supported on quantized storage (weights are immutable).
func (s *GPTQStorage) Set(_ []float32) { panic("GPTQStorage is immutable") }

// DeviceType returns device.CPU.
func (s *GPTQStorage) DeviceType() device.Type { return device.CPU }

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw bytes.
func (s *GPTQStorage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	s.gpuPtr = ptr
	s.gpuByteSize = byteSize
	s.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
func (s *GPTQStorage) GPUPtr() (unsafe.Pointer, int, int) {
	return s.gpuPtr, s.gpuByteSize, s.gpuDeviceID
}

// NewGPTQStorageFromRaw creates GPTQStorage from pre-extracted components.
// data is the packed quantized values (all groups concatenated).
// scales and zeros are FP16 values, one per group.
// numElements is the logical element count.
// groupSize and bits define the quantization parameters.
func NewGPTQStorageFromRaw(data []byte, scales, zeros []float16.Float16, numElements, groupSize, bits int) (*GPTQStorage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	if bits != 4 && bits != 8 {
		return nil, fmt.Errorf("bits must be 4 or 8, got %d", bits)
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
		if bits == 4 {
			totalPacked += (count + 1) / 2
		} else {
			totalPacked += count
		}
	}
	if len(data) < totalPacked {
		return nil, fmt.Errorf("GPTQ data too short: need %d bytes, got %d", totalPacked, len(data))
	}

	groups := make([]gptqGroup, nGroups)
	dataOff := 0
	for gi := range nGroups {
		count := groupSize
		remaining := numElements - gi*groupSize
		if remaining < groupSize {
			count = remaining
		}
		var packedLen int
		if bits == 4 {
			packedLen = (count + 1) / 2
		} else {
			packedLen = count
		}
		groups[gi].scale = scales[gi]
		groups[gi].zero = zeros[gi]
		groups[gi].data = make([]byte, packedLen)
		copy(groups[gi].data, data[dataOff:dataOff+packedLen])
		dataOff += packedLen
	}

	return &GPTQStorage{
		groups:    groups,
		len:       numElements,
		groupSize: groupSize,
		bits:      bits,
	}, nil
}

// Ensure GPTQStorage implements Storage[float32].
var _ Storage[float32] = (*GPTQStorage)(nil)

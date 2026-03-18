package tensor

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// Dequantizer decodes quantized data back to floating point.
type Dequantizer interface {
	// Dequantize decodes quantized bytes in src into float32 values in dst.
	// The caller must ensure dst has sufficient capacity for the decoded output.
	Dequantize(src []byte, dst []float32) error

	// BlockSize returns the number of elements per quantization block.
	BlockSize() int

	// BitsPerWeight returns the effective number of bits per weight element.
	BitsPerWeight() int
}

var (
	quantMu       sync.RWMutex
	quantRegistry = make(map[string]Dequantizer)
)

// RegisterQuantType registers a quantization format by name.
// It panics if name is empty or if a format with that name is already registered.
// This is intended to be called from init() functions.
func RegisterQuantType(name string, d Dequantizer) {
	if name == "" {
		panic("tensor: RegisterQuantType called with empty name")
	}
	if d == nil {
		panic("tensor: RegisterQuantType called with nil Dequantizer")
	}
	quantMu.Lock()
	defer quantMu.Unlock()
	if _, exists := quantRegistry[name]; exists {
		panic("tensor: RegisterQuantType called twice for " + name)
	}
	quantRegistry[name] = d
}

// GetQuantType returns the Dequantizer registered under name.
// The second return value is false if no format is registered under that name.
func GetQuantType(name string) (Dequantizer, bool) {
	quantMu.RLock()
	defer quantMu.RUnlock()
	d, ok := quantRegistry[name]
	return d, ok
}

// ListQuantTypes returns the names of all registered quantization formats
// in sorted order.
func ListQuantTypes() []string {
	quantMu.RLock()
	defer quantMu.RUnlock()
	names := make([]string, 0, len(quantRegistry))
	for name := range quantRegistry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// resetQuantRegistry clears the registry. Only for testing.
func resetQuantRegistry() {
	quantMu.Lock()
	defer quantMu.Unlock()
	quantRegistry = make(map[string]Dequantizer)
}

// ---------------------------------------------------------------------------
// Built-in dequantizers wrapping existing storage types.
// Registered via init() below.
// ---------------------------------------------------------------------------

// q4Dequantizer wraps Q4_0 dequantization.
type q4Dequantizer struct{}

func (q4Dequantizer) Dequantize(src []byte, dst []float32) error {
	n := len(dst)
	s, err := NewQ4StorageFromRaw(src, n)
	if err != nil {
		return err
	}
	s.Dequantize(dst)
	return nil
}

func (q4Dequantizer) BlockSize() int     { return q4BlockSize }
func (q4Dequantizer) BitsPerWeight() int { return 4 }

// q8Dequantizer wraps Q8_0 dequantization.
type q8Dequantizer struct{}

func (q8Dequantizer) Dequantize(src []byte, dst []float32) error {
	n := len(dst)
	nBlocks := (n + q8BlockSize - 1) / q8BlockSize
	const blockBytes = 36
	if len(src) < nBlocks*blockBytes {
		return errShortData("Q8_0", nBlocks*blockBytes, len(src))
	}
	// Decode blocks from raw bytes.
	scales := make([]float32, nBlocks)
	quants := make([]int8, nBlocks*q8BlockSize)
	for i := range nBlocks {
		off := i * blockBytes
		scales[i] = decodeFloat32LE(src[off : off+4])
		for j := range q8BlockSize {
			quants[i*q8BlockSize+j] = int8(src[off+4+j])
		}
	}
	s, err := NewQ8StorageFromBlocks(scales, quants, n)
	if err != nil {
		return err
	}
	s.Dequantize(dst)
	return nil
}

func (q8Dequantizer) BlockSize() int     { return q8BlockSize }
func (q8Dequantizer) BitsPerWeight() int { return 8 }

// q4KDequantizer wraps Q4_K dequantization.
type q4KDequantizer struct{}

func (q4KDequantizer) Dequantize(src []byte, dst []float32) error {
	n := len(dst)
	s, err := NewQ4KStorageFromRaw(src, n)
	if err != nil {
		return err
	}
	s.Dequantize(dst)
	return nil
}

func (q4KDequantizer) BlockSize() int     { return q4KSuperBlockSize }
func (q4KDequantizer) BitsPerWeight() int { return 4 }

// q5KDequantizer wraps Q5_K dequantization.
type q5KDequantizer struct{}

func (q5KDequantizer) Dequantize(src []byte, dst []float32) error {
	n := len(dst)
	s, err := NewQ5KStorageFromRaw(src, n)
	if err != nil {
		return err
	}
	s.Dequantize(dst)
	return nil
}

func (q5KDequantizer) BlockSize() int     { return q5KSuperBlockSize }
func (q5KDequantizer) BitsPerWeight() int { return 5 }

// q6KDequantizer wraps Q6_K dequantization.
type q6KDequantizer struct{}

func (q6KDequantizer) Dequantize(src []byte, dst []float32) error {
	n := len(dst)
	s, err := NewQ6KStorageFromRaw(src, n)
	if err != nil {
		return err
	}
	s.Dequantize(dst)
	return nil
}

func (q6KDequantizer) BlockSize() int     { return q6KSuperBlockSize }
func (q6KDequantizer) BitsPerWeight() int { return 6 }

// q5_0Dequantizer wraps Q5_0 dequantization.
type q5_0Dequantizer struct{}

func (q5_0Dequantizer) Dequantize(src []byte, dst []float32) error {
	n := len(dst)
	s, err := NewQ5_0StorageFromRaw(src, n)
	if err != nil {
		return err
	}
	s.Dequantize(dst)
	return nil
}

func (q5_0Dequantizer) BlockSize() int     { return q5_0BlockSize }
func (q5_0Dequantizer) BitsPerWeight() int { return 5 }

// fp8E4M3Dequantizer wraps FP8 E4M3 dequantization.
// Note: FP8 uses per-tensor scaling so this adapter assumes the scale is
// encoded as the first 4 bytes of src (little-endian float32), followed
// by one byte per element.
type fp8E4M3Dequantizer struct{}

func (fp8E4M3Dequantizer) Dequantize(src []byte, dst []float32) error {
	n := len(dst)
	if len(src) < 4+n {
		return errShortData("FP8_E4M3", 4+n, len(src))
	}
	scale := decodeFloat32LE(src[:4])
	for i := range n {
		dst[i] = decodeE4M3(src[4+i]) * scale
	}
	return nil
}

func (fp8E4M3Dequantizer) BlockSize() int     { return 1 }
func (fp8E4M3Dequantizer) BitsPerWeight() int { return 8 }

// fp8E5M2Dequantizer wraps FP8 E5M2 dequantization.
// Same layout as E4M3: 4-byte LE scale prefix + 1 byte per element.
type fp8E5M2Dequantizer struct{}

func (fp8E5M2Dequantizer) Dequantize(src []byte, dst []float32) error {
	n := len(dst)
	if len(src) < 4+n {
		return errShortData("FP8_E5M2", 4+n, len(src))
	}
	scale := decodeFloat32LE(src[:4])
	for i := range n {
		dst[i] = decodeE5M2(src[4+i]) * scale
	}
	return nil
}

func (fp8E5M2Dequantizer) BlockSize() int     { return 1 }
func (fp8E5M2Dequantizer) BitsPerWeight() int { return 8 }

func init() {
	RegisterQuantType("Q4_0", q4Dequantizer{})
	RegisterQuantType("Q8_0", q8Dequantizer{})
	RegisterQuantType("Q4_K", q4KDequantizer{})
	RegisterQuantType("Q5_K", q5KDequantizer{})
	RegisterQuantType("Q6_K", q6KDequantizer{})
	RegisterQuantType("Q5_0", q5_0Dequantizer{})
	RegisterQuantType("FP8_E4M3", fp8E4M3Dequantizer{})
	RegisterQuantType("FP8_E5M2", fp8E5M2Dequantizer{})
}

// errShortData returns a formatted error for insufficient input data.
func errShortData(format string, need, got int) error {
	return fmt.Errorf("tensor: %s data too short: need %d bytes, got %d", format, need, got)
}

// decodeFloat32LE decodes a little-endian float32 from 4 bytes.
func decodeFloat32LE(b []byte) float32 {
	bits := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	return math.Float32frombits(bits)
}

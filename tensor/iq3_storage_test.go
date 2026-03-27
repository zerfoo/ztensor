package tensor

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/float16"
)

// buildIQ3SBlock constructs a raw IQ3_S super-block (110 bytes) for testing.
// values must have exactly 256 elements.
func buildIQ3SBlock(values []float32) []byte {
	if len(values) != 256 {
		panic("buildIQ3SBlock requires exactly 256 values")
	}

	raw := make([]byte, 110)

	// Find overall absmax.
	var absMax float32
	for _, v := range values {
		if av := float32(math.Abs(float64(v))); av > absMax {
			absMax = av
		}
	}

	if absMax == 0 {
		return raw
	}

	// Compute per-sub-block scales (8 sub-blocks of 32 values).
	subAbsMax := make([]float32, 8)
	for sb := range 8 {
		off := sb * 32
		for j := range 32 {
			if av := float32(math.Abs(float64(values[off+j]))); av > subAbsMax[sb] {
				subAbsMax[sb] = av
			}
		}
	}

	// Determine d (super-block scale). The sub-block scale factor is (1 + 2*sc)
	// where sc is a 4-bit value (0-15), so max multiplier is 31.
	// The grid values are in {1,3,5,7}, max = 7.
	// So: value = d * (1+2*sc) * grid_val * sign
	// Max representable = d * 31 * 7
	d := absMax / (31.0 * 7.0)

	// Write fp16 d.
	binary.LittleEndian.PutUint16(raw[0:2], float16.FromFloat32(d).Bits())
	dActual := float16.FromFloat32(d).ToFloat32()

	// Compute quantized sub-block scales.
	scalesQ := make([]uint8, 8)
	for sb := range 8 {
		// subAbsMax[sb] = dActual * (1 + 2*sc) * 7  (worst case grid=7)
		if dActual > 0 {
			sc := (subAbsMax[sb]/(dActual*7.0) - 1.0) / 2.0
			scQ := int(math.Round(float64(sc)))
			if scQ < 0 {
				scQ = 0
			}
			if scQ > 15 {
				scQ = 15
			}
			scalesQ[sb] = uint8(scQ)
		}
	}

	// Pack 4-bit scales: pairs packed into bytes.
	for i := range 4 {
		raw[106+i] = scalesQ[2*i] | (scalesQ[2*i+1] << 4)
	}

	// Quantize each value.
	// For each group of 8 values, we choose grid indices and set sign bits.
	qsIdx := 0
	signIdx := 0

	for sb := range 8 {
		subScale := dActual * float32(1+2*int(scalesQ[sb]))
		qhByte := uint8(0)

		for g := range 4 { // 4 groups of 8 per sub-block
			baseIdx := sb*32 + g*8

			// First 4 values -> grid1
			bestGrid1 := uint16(0)
			bestErr1 := float32(math.MaxFloat32)
			for gi := uint16(0); gi < 256; gi++ {
				grid := iq3sGrid[gi]
				err := float32(0)
				for k := range 4 {
					target := float32(math.Abs(float64(values[baseIdx+k])))
					reconstructed := subScale * float32(grid[k])
					diff := target - reconstructed
					err += diff * diff
				}
				if err < bestErr1 {
					bestErr1 = err
					bestGrid1 = gi
				}
			}

			// Check indices 256-511 too (high bit set).
			for gi := uint16(256); gi < 512; gi++ {
				grid := iq3sGrid[gi]
				err := float32(0)
				for k := range 4 {
					target := float32(math.Abs(float64(values[baseIdx+k])))
					reconstructed := subScale * float32(grid[k])
					diff := target - reconstructed
					err += diff * diff
				}
				if err < bestErr1 {
					bestErr1 = err
					bestGrid1 = gi
				}
			}

			raw[2+qsIdx] = uint8(bestGrid1 & 0xFF)
			if bestGrid1&0x100 != 0 {
				qhByte |= 1 << (2 * uint(g))
			}

			// Sign bits for first 4 values.
			signByte := uint8(0)
			for k := range 4 {
				if values[baseIdx+k] < 0 {
					signByte |= 1 << uint(k)
				}
			}

			// Second 4 values -> grid2
			bestGrid2 := uint16(0)
			bestErr2 := float32(math.MaxFloat32)
			for gi := uint16(0); gi < 512; gi++ {
				grid := iq3sGrid[gi]
				err := float32(0)
				for k := range 4 {
					target := float32(math.Abs(float64(values[baseIdx+4+k])))
					reconstructed := subScale * float32(grid[k])
					diff := target - reconstructed
					err += diff * diff
				}
				if err < bestErr2 {
					bestErr2 = err
					bestGrid2 = gi
				}
			}

			raw[2+qsIdx+1] = uint8(bestGrid2 & 0xFF)
			if bestGrid2&0x100 != 0 {
				qhByte |= 1 << (2*uint(g) + 1)
			}

			// Sign bits for second 4 values.
			for k := range 4 {
				if values[baseIdx+4+k] < 0 {
					signByte |= 1 << uint(k+4)
				}
			}

			raw[74+signIdx] = signByte
			qsIdx += 2
			signIdx++
		}

		raw[66+sb] = qhByte
	}

	return raw
}

func TestDequantizeIQ3S_RoundTrip(t *testing.T) {
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(math.Sin(float64(i)*0.1)) * 2.0
	}

	raw := buildIQ3SBlock(values)
	dst := make([]float32, 256)
	DequantizeIQ3S(raw, dst)

	maxErr := float32(0.0)
	for i := range values {
		err := float32(math.Abs(float64(dst[i] - values[i])))
		if err > maxErr {
			maxErr = err
		}
	}
	// IQ3_S has 3-bit precision with grid quantization; allow generous tolerance.
	if maxErr > 1.5 {
		t.Errorf("max dequantization error %f exceeds 1.5", maxErr)
	}
}

func TestDequantizeIQ3S_Zeros(t *testing.T) {
	values := make([]float32, 256)
	raw := buildIQ3SBlock(values)
	dst := make([]float32, 256)
	DequantizeIQ3S(raw, dst)

	for i, v := range dst {
		if v != 0 {
			t.Errorf("dst[%d] = %f, want 0", i, v)
			break
		}
	}
}

func TestDequantizeIQ3S_KnownBlock(t *testing.T) {
	// Construct a block with known grid indices and verify exact output.
	raw := make([]byte, 110)

	// Set d = 0.5
	binary.LittleEndian.PutUint16(raw[0:2], float16.FromFloat32(0.5).Bits())
	dActual := float16.FromFloat32(0.5).ToFloat32()

	// Set scale for sub-block 0 = 0 (so multiplier = 1+2*0 = 1).
	raw[106] = 0x00

	// Set grid index 0 for first qs byte: index 0 maps to {1,1,1,1}.
	raw[2] = 0
	// Set grid index 0xFF (255) for second qs byte: maps to {7,7,7,7}.
	raw[3] = 0xFF
	// No high bits (qh = 0).
	raw[66] = 0

	// No sign bits for first group.
	raw[74] = 0x00

	// Fill remaining qs with 0 and signs with 0.
	// (Already zero from make.)

	dst := make([]float32, 256)
	DequantizeIQ3S(raw, dst)

	// First 4 values: d * 1 * {1,1,1,1} = {0.5, 0.5, 0.5, 0.5}
	for k := range 4 {
		want := dActual * 1.0
		if math.Abs(float64(dst[k]-want)) > 1e-4 {
			t.Errorf("dst[%d] = %f, want %f", k, dst[k], want)
		}
	}

	// Next 4 values: d * 1 * {7,7,7,7} = {3.5, 3.5, 3.5, 3.5}
	for k := range 4 {
		want := dActual * 7.0
		if math.Abs(float64(dst[4+k]-want)) > 1e-4 {
			t.Errorf("dst[%d] = %f, want %f", 4+k, dst[4+k], want)
		}
	}
}

func TestDequantizeIQ3S_Signs(t *testing.T) {
	raw := make([]byte, 110)

	// d = 1.0
	binary.LittleEndian.PutUint16(raw[0:2], float16.FromFloat32(1.0).Bits())
	dActual := float16.FromFloat32(1.0).ToFloat32()

	// Scale = 0 -> multiplier = 1.
	raw[106] = 0

	// Grid index 0 -> {1,1,1,1} for both halves.
	raw[2] = 0
	raw[3] = 0

	// Sign: negate values 0 and 6 (bit 0 of first half, bit 2 of second half).
	raw[74] = (1 << 0) | (1 << 6)

	dst := make([]float32, 256)
	DequantizeIQ3S(raw, dst)

	// dst[0] should be negative (sign bit 0 set).
	if dst[0] != -dActual {
		t.Errorf("dst[0] = %f, want %f", dst[0], -dActual)
	}
	// dst[1] should be positive.
	if dst[1] != dActual {
		t.Errorf("dst[1] = %f, want %f", dst[1], dActual)
	}
	// dst[6] should be negative (sign bit 6 = bit 2 of second half).
	if dst[6] != -dActual {
		t.Errorf("dst[6] = %f, want %f", dst[6], -dActual)
	}
}

func TestNewIQ3SStorageFromRaw(t *testing.T) {
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(i) * 0.01
	}

	raw := buildIQ3SBlock(values)
	storage, err := NewIQ3SStorageFromRaw(raw, 256)
	if err != nil {
		t.Fatalf("NewIQ3SStorageFromRaw: %v", err)
	}

	if storage.Len() != 256 {
		t.Errorf("Len() = %d, want 256", storage.Len())
	}

	slice := storage.Slice()
	if len(slice) != 256 {
		t.Fatalf("Slice() len = %d, want 256", len(slice))
	}
}

func TestNewIQ3SStorageFromRaw_InvalidSize(t *testing.T) {
	_, err := NewIQ3SStorageFromRaw(make([]byte, 10), 256)
	if err == nil {
		t.Fatal("expected error for short raw data")
	}
}

func TestNewIQ3SStorageFromRaw_InvalidElements(t *testing.T) {
	_, err := NewIQ3SStorageFromRaw(make([]byte, 110), 0)
	if err == nil {
		t.Fatal("expected error for zero elements")
	}

	_, err = NewIQ3SStorageFromRaw(make([]byte, 110), -1)
	if err == nil {
		t.Fatal("expected error for negative elements")
	}
}

func TestIQ3SStorage_DeviceType(t *testing.T) {
	raw := make([]byte, 110)
	s, err := NewIQ3SStorageFromRaw(raw, 256)
	if err != nil {
		t.Fatal(err)
	}
	if dt := s.DeviceType(); dt != 0 { // device.CPU = 0
		t.Errorf("DeviceType() = %v, want CPU (0)", dt)
	}
}

func TestIQ3SStorage_SetPanics(t *testing.T) {
	raw := make([]byte, 110)
	s, err := NewIQ3SStorageFromRaw(raw, 256)
	if err != nil {
		t.Fatal(err)
	}

	defer func() {
		if r := recover(); r == nil {
			t.Error("Set() did not panic")
		}
	}()
	s.Set([]float32{1, 2, 3})
}

func TestIQ3SStorage_NumBlocks(t *testing.T) {
	tests := []struct {
		numElements int
		wantBlocks  int
	}{
		{256, 1},
		{512, 2},
		{257, 2},
		{1, 1},
	}
	for _, tt := range tests {
		raw := make([]byte, tt.wantBlocks*iq3SBlockBytes)
		s, err := NewIQ3SStorageFromRaw(raw, tt.numElements)
		if err != nil {
			t.Fatalf("NewIQ3SStorageFromRaw(%d): %v", tt.numElements, err)
		}
		if got := s.NumBlocks(); got != tt.wantBlocks {
			t.Errorf("NumBlocks() for %d elements = %d, want %d", tt.numElements, got, tt.wantBlocks)
		}
	}
}

func TestIQ3SStorage_RawBytesRoundTrip(t *testing.T) {
	raw := make([]byte, 110)
	for i := range raw {
		raw[i] = byte(i)
	}
	s, err := NewIQ3SStorageFromRaw(raw, 256)
	if err != nil {
		t.Fatal(err)
	}
	got := s.RawBytes()
	if len(got) != 110 {
		t.Fatalf("RawBytes() len = %d, want 110", len(got))
	}
	for i := range raw {
		if got[i] != raw[i] {
			t.Errorf("RawBytes()[%d] = %d, want %d", i, got[i], raw[i])
		}
	}
}

func TestIQ3SDequantizer_Registry(t *testing.T) {
	d, ok := GetQuantType("IQ3_S")
	if !ok {
		t.Fatal("IQ3_S not registered")
	}
	if d.BlockSize() != 256 {
		t.Errorf("BlockSize() = %d, want 256", d.BlockSize())
	}
	if d.BitsPerWeight() != 3 {
		t.Errorf("BitsPerWeight() = %d, want 3", d.BitsPerWeight())
	}
}

func TestIQ3SDequantizer_DequantizeViaRegistry(t *testing.T) {
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(math.Sin(float64(i)*0.15)) * 1.5
	}

	raw := buildIQ3SBlock(values)
	d, _ := GetQuantType("IQ3_S")

	dst := make([]float32, 256)
	if err := d.Dequantize(raw, dst); err != nil {
		t.Fatalf("Dequantize via registry: %v", err)
	}

	maxErr := float32(0.0)
	for i := range values {
		err := float32(math.Abs(float64(dst[i] - values[i])))
		if err > maxErr {
			maxErr = err
		}
	}
	if maxErr > 1.5 {
		t.Errorf("max dequantization error %f exceeds 1.5", maxErr)
	}
}

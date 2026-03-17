package kernels

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// dequantizeQ5_0 dequantizes one Q5_0 block (22 bytes) into 32 float32 values.
// Inlined here to avoid an import cycle with the tensor package.
func dequantizeQ5_0(raw []byte, dst []float32) {
	d := float16BitsToFloat32(binary.LittleEndian.Uint16(raw[0:2]))
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

// buildQ5_0TestData constructs M rows of Q5_0 blocks and computes the
// reference GEMV output using dequantizeQ5_0 for the dequantization.
// K must be a multiple of 32.
func buildQ5_0TestData(M, K int) (raw []byte, x []float32, ref []float32) {
	const blockSize = 32
	const blockBytes = 22

	blocksPerRow := K / blockSize
	raw = make([]byte, M*blocksPerRow*blockBytes)
	x = make([]float32, K)
	ref = make([]float32, M)

	for i := range x {
		x[i] = float32(i%17-8) * 0.05
	}

	for row := range M {
		rowValues := make([]float32, K)
		for i := range K {
			rowValues[i] = float32(math.Sin(float64(row*K+i)*0.03)) * 1.5
		}

		for bi := range blocksPerRow {
			blkValues := rowValues[bi*blockSize : (bi+1)*blockSize]
			blkRaw := buildQ5_0BlockFromValues(blkValues)
			off := (row*blocksPerRow + bi) * blockBytes
			copy(raw[off:off+blockBytes], blkRaw)
		}

		dequant := make([]float32, K)
		for bi := range blocksPerRow {
			off := (row*blocksPerRow + bi) * blockBytes
			dequantizeQ5_0(raw[off:off+blockBytes], dequant[bi*blockSize:(bi+1)*blockSize])
		}
		var sum float32
		for k := range K {
			sum += dequant[k] * x[k]
		}
		ref[row] = sum
	}

	return raw, x, ref
}

// buildQ5_0BlockFromValues quantizes 32 float32 values into a Q5_0 block.
func buildQ5_0BlockFromValues(values []float32) []byte {
	// Find range for the block.
	minVal := values[0]
	maxVal := values[0]
	for _, v := range values[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Q5_0: values are mapped to -16..15 (5-bit signed).
	// d = (max - min) / 31
	// quantized = round((v - min) / d) - 16
	// so: v ≈ d * (q + 16) + min = d * q + d*16 + min
	// But the actual formula is: v = d * (q - 16) where q is in 0..31.
	// So d = max(abs(max), abs(min)) / 15... Actually:
	// dequant formula: dst[j] = d * (int(nibble | highbit) - 16)
	// Range: -16*d to 15*d. So d = max(abs(values)) / 15 if symmetric around 0.
	// Better: d = max(abs(min), abs(max)) / 15
	absMax := math.Max(math.Abs(float64(minVal)), math.Abs(float64(maxVal)))
	d := float32(absMax / 15.0)
	if d == 0 {
		d = 1.0 // avoid division by zero
	}

	// Round-trip d through fp16 to match GPU behavior.
	dBits := float32ToFloat16Bits(d)
	d = float16BitsToFloat32(dBits)

	raw := make([]byte, 22)
	binary.LittleEndian.PutUint16(raw[0:2], dBits)

	var qh uint32
	qs := raw[6:22]

	for j := range 16 {
		v0 := values[j]
		v1 := values[j+16]

		// Quantize: q = round(v/d) + 16, clamped to [0, 31]
		q0 := clampInt(int(math.Round(float64(v0/d)))+16, 0, 31)
		q1 := clampInt(int(math.Round(float64(v1/d)))+16, 0, 31)

		// Split into low 4 bits and high bit.
		low0 := uint8(q0 & 0xF)
		high0 := uint32((q0 >> 4) & 1)
		low1 := uint8(q1 & 0xF)
		high1 := uint32((q1 >> 4) & 1)

		qs[j] = low0 | (low1 << 4)
		qh |= high0 << j
		qh |= high1 << (j + 16)
	}

	binary.LittleEndian.PutUint32(raw[2:6], qh)

	return raw
}

func TestGemvQ5_0F32_Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	M, K := 64, 32
	raw, x, ref := buildQ5_0TestData(M, K)

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	devW, err := cuda.Malloc(len(raw))
	if err != nil {
		t.Fatalf("cuda.Malloc W: %v", err)
	}
	defer func() { _ = cuda.Free(devW) }()

	devX, err := cuda.Malloc(K * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc x: %v", err)
	}
	defer func() { _ = cuda.Free(devX) }()

	devY, err := cuda.Malloc(M * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc y: %v", err)
	}
	defer func() { _ = cuda.Free(devY) }()

	if err := cuda.Memcpy(devW, unsafe.Pointer(&raw[0]), len(raw), cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy W: %v", err)
	}
	if err := cuda.Memcpy(devX, unsafe.Pointer(&x[0]), K*4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy x: %v", err)
	}

	if err := GemvQ5_0F32(devW, devX, devY, M, K, stream.Ptr()); err != nil {
		t.Fatalf("GemvQ5_0F32: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, M)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, M*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy y: %v", err)
	}

	maxRelErr := 0.0
	for i := range got {
		absRef := math.Abs(float64(ref[i]))
		diff := math.Abs(float64(got[i] - ref[i]))
		var relErr float64
		if absRef > 1e-6 {
			relErr = diff / absRef
		} else {
			relErr = diff
		}
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		if relErr > 1e-4 {
			t.Errorf("y[%d] = %f, want %f (rel err %e)", i, got[i], ref[i], relErr)
			if t.Failed() {
				break
			}
		}
	}
	t.Logf("max relative error: %e", maxRelErr)
}

func TestGemvQ5_0F32_MultipleSizes(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	cases := []struct {
		name string
		M, K int
	}{
		{"small_32x32", 32, 32},
		{"medium_64x128", 64, 128},
		{"square_128x128", 128, 128},
		{"wide_128x1024", 128, 1024},
		{"tall_1024x32", 1024, 32},
		{"large_512x2048", 512, 2048},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, x, ref := buildQ5_0TestData(tc.M, tc.K)

			stream, err := cuda.CreateStream()
			if err != nil {
				t.Fatalf("CreateStream: %v", err)
			}
			defer func() { _ = stream.Destroy() }()

			devW, err := cuda.Malloc(len(raw))
			if err != nil {
				t.Fatalf("cuda.Malloc W: %v", err)
			}
			defer func() { _ = cuda.Free(devW) }()

			devX, err := cuda.Malloc(tc.K * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc x: %v", err)
			}
			defer func() { _ = cuda.Free(devX) }()

			devY, err := cuda.Malloc(tc.M * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc y: %v", err)
			}
			defer func() { _ = cuda.Free(devY) }()

			if err := cuda.Memcpy(devW, unsafe.Pointer(&raw[0]), len(raw), cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy W: %v", err)
			}
			if err := cuda.Memcpy(devX, unsafe.Pointer(&x[0]), tc.K*4, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy x: %v", err)
			}

			if err := GemvQ5_0F32(devW, devX, devY, tc.M, tc.K, stream.Ptr()); err != nil {
				t.Fatalf("GemvQ5_0F32: %v", err)
			}

			if err := stream.Synchronize(); err != nil {
				t.Fatalf("Synchronize: %v", err)
			}

			got := make([]float32, tc.M)
			if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, tc.M*4, cuda.MemcpyDeviceToHost); err != nil {
				t.Fatalf("Memcpy y: %v", err)
			}

			maxRelErr := 0.0
			for i := range got {
				absRef := math.Abs(float64(ref[i]))
				diff := math.Abs(float64(got[i] - ref[i]))
				var relErr float64
				if absRef > 1e-6 {
					relErr = diff / absRef
				} else {
					relErr = diff
				}
				if relErr > maxRelErr {
					maxRelErr = relErr
				}
				if relErr > 1e-4 {
					t.Errorf("y[%d] = %f, want %f (rel err %e)", i, got[i], ref[i], relErr)
					if t.Failed() {
						break
					}
				}
			}
			t.Logf("max relative error: %e", maxRelErr)
		})
	}
}

func BenchmarkGemvQ5_0F32_4096(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}

	M, K := 4096, 4096
	raw, x, _ := buildQ5_0TestData(M, K)

	stream, err := cuda.CreateStream()
	if err != nil {
		b.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	devW, _ := cuda.Malloc(len(raw))
	defer func() { _ = cuda.Free(devW) }()
	devX, _ := cuda.Malloc(K * 4)
	defer func() { _ = cuda.Free(devX) }()
	devY, _ := cuda.Malloc(M * 4)
	defer func() { _ = cuda.Free(devY) }()

	_ = cuda.Memcpy(devW, unsafe.Pointer(&raw[0]), len(raw), cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devX, unsafe.Pointer(&x[0]), K*4, cuda.MemcpyHostToDevice)

	b.ResetTimer()
	for b.Loop() {
		_ = GemvQ5_0F32(devW, devX, devY, M, K, stream.Ptr())
	}
	_ = stream.Synchronize()

	elapsed := b.Elapsed()
	flops := 2.0 * float64(M) * float64(K) * float64(b.N)
	gflops := flops / elapsed.Seconds() / 1e9
	b.ReportMetric(gflops, "GFLOPS")
}

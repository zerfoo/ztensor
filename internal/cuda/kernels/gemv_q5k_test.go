package kernels

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// dequantizeQ5K dequantizes one Q5_K super-block (176 bytes) into 256 float32 values.
// Inlined here to avoid an import cycle with the tensor package.
func dequantizeQ5K(raw []byte, dst []float32) {
	d := float16BitsToFloat32(binary.LittleEndian.Uint16(raw[0:2]))
	dmin := float16BitsToFloat32(binary.LittleEndian.Uint16(raw[2:4]))

	sc := raw[4:16]
	var scales, mins [8]uint8
	for i := range 4 {
		scales[i] = sc[i] & 63
		mins[i] = sc[4+i] & 63
	}
	for i := range 4 {
		scales[4+i] = (sc[8+i] & 0xF) | ((sc[i] >> 6) << 4)
		mins[4+i] = (sc[8+i] >> 4) | ((sc[4+i] >> 6) << 4)
	}

	ql := raw[16:144]  // 128 bytes: low 4 bits
	qh := raw[144:176] // 32 bytes: high 1 bit

	u1 := uint8(1)
	u2 := uint8(2)
	for group := range 4 {
		sb0 := group * 2
		sb1 := group*2 + 1
		sc0 := d * float32(scales[sb0])
		mn0 := dmin * float32(mins[sb0])
		sc1 := d * float32(scales[sb1])
		mn1 := dmin * float32(mins[sb1])

		baseOut := group * 64
		baseQ := group * 32
		for l := range 32 {
			q := ql[baseQ+l]
			hb := qh[l]

			var h0, h1 uint8
			if hb&u1 != 0 {
				h0 = 16
			}
			if hb&u2 != 0 {
				h1 = 16
			}

			dst[baseOut+l] = sc0*float32((q&0xF)|h0) - mn0
			dst[baseOut+l+32] = sc1*float32((q>>4)|h1) - mn1
		}
		u1 <<= 2
		u2 <<= 2
	}
}

// buildQ5KTestData constructs M rows of Q5_K super-blocks and computes the
// reference GEMV output using dequantizeQ5K for the dequantization.
// K must be a multiple of 256.
func buildQ5KTestData(M, K int) (raw []byte, x []float32, ref []float32) {
	const superBlockSize = 256
	const blockBytes = 176

	blocksPerRow := K / superBlockSize
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
			blkValues := rowValues[bi*superBlockSize : (bi+1)*superBlockSize]
			blkRaw := buildQ5KBlockFromValues(blkValues)
			off := (row*blocksPerRow + bi) * blockBytes
			copy(raw[off:off+blockBytes], blkRaw)
		}

		dequant := make([]float32, K)
		for bi := range blocksPerRow {
			off := (row*blocksPerRow + bi) * blockBytes
			dequantizeQ5K(raw[off:off+blockBytes], dequant[bi*superBlockSize:(bi+1)*superBlockSize])
		}
		var sum float32
		for k := range K {
			sum += dequant[k] * x[k]
		}
		ref[row] = sum
	}

	return raw, x, ref
}

// buildQ5KBlockFromValues quantizes 256 float32 values into a Q5_K super-block.
func buildQ5KBlockFromValues(values []float32) []byte {
	const numSubBlocks = 8
	const subBlockSize = 32

	subScales := make([]float32, numSubBlocks)
	subMins := make([]float32, numSubBlocks)

	for sb := range numSubBlocks {
		off := sb * subBlockSize
		minVal := values[off]
		maxVal := values[off]
		for j := 1; j < subBlockSize; j++ {
			v := values[off+j]
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
		if minVal > 0 {
			minVal = 0
		}
		subScales[sb] = (maxVal - minVal) / 31.0 // Q5 has 5-bit range 0..31
		subMins[sb] = -minVal
	}

	var maxScale, maxMin float32
	for sb := range numSubBlocks {
		if subScales[sb] > maxScale {
			maxScale = subScales[sb]
		}
		if subMins[sb] > maxMin {
			maxMin = subMins[sb]
		}
	}

	d := maxScale / 63.0
	dmin := maxMin / 63.0

	scalesQ := make([]uint8, numSubBlocks)
	minsQ := make([]uint8, numSubBlocks)
	for sb := range numSubBlocks {
		if d > 0 {
			scalesQ[sb] = uint8(math.Round(float64(subScales[sb] / d)))
			if scalesQ[sb] > 63 {
				scalesQ[sb] = 63
			}
		}
		if dmin > 0 {
			minsQ[sb] = uint8(math.Round(float64(subMins[sb] / dmin)))
			if minsQ[sb] > 63 {
				minsQ[sb] = 63
			}
		}
	}

	raw := make([]byte, 176)

	binary.LittleEndian.PutUint16(raw[0:2], float32ToFloat16Bits(d))
	binary.LittleEndian.PutUint16(raw[2:4], float32ToFloat16Bits(dmin))

	// Pack 6-bit scales and mins (same as Q4_K).
	for i := range 4 {
		raw[4+i] = (scalesQ[i] & 63) | ((scalesQ[4+i] >> 4) << 6)
		raw[8+i] = (minsQ[i] & 63) | ((minsQ[4+i] >> 4) << 6)
	}
	for i := range 4 {
		raw[12+i] = (scalesQ[4+i] & 0xF) | ((minsQ[4+i] & 0xF) << 4)
	}

	// Round-trip d/dmin through fp16 for accurate quantization.
	dRT := float16BitsToFloat32(float32ToFloat16Bits(d))
	dminRT := float16BitsToFloat32(float32ToFloat16Bits(dmin))

	// Quantize values to 5 bits using ql (low 4 bits) + qh (high 1 bit).
	for group := range 4 {
		sb0 := group * 2
		sb1 := group*2 + 1

		sc0 := dRT * float32(scalesQ[sb0])
		mn0 := dminRT * float32(minsQ[sb0])
		sc1 := dRT * float32(scalesQ[sb1])
		mn1 := dminRT * float32(minsQ[sb1])

		var invScale0, invScale1 float32
		if sc0 > 0 {
			invScale0 = 1.0 / sc0
		}
		if sc1 > 0 {
			invScale1 = 1.0 / sc1
		}

		u1 := uint8(1 << (2 * group))
		u2 := uint8(2 << (2 * group))

		baseOut := group * 64
		baseQ := group * 32
		for l := range 32 {
			v0 := values[baseOut+l]
			v1 := values[baseOut+l+32]
			q0 := clampInt(int(math.Round(float64((v0+mn0)*invScale0))), 0, 31)
			q1 := clampInt(int(math.Round(float64((v1+mn1)*invScale1))), 0, 31)

			// Low 4 bits go into ql.
			raw[16+baseQ+l] = byte(q0&0xF) | (byte(q1&0xF) << 4)

			// High bit (bit 4) goes into qh.
			if q0&16 != 0 {
				raw[144+l] |= u1
			}
			if q1&16 != 0 {
				raw[144+l] |= u2
			}
		}
	}

	return raw
}

func TestGemvQ5KF32_Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	M, K := 64, 256
	raw, x, ref := buildQ5KTestData(M, K)

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

	if err := GemvQ5KF32(devW, devX, devY, M, K, stream.Ptr()); err != nil {
		t.Fatalf("GemvQ5KF32: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, M)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, M*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy y: %v", err)
	}

	maxAbsErr := 0.0
	for i := range got {
		diff := math.Abs(float64(got[i] - ref[i]))
		if diff > maxAbsErr {
			maxAbsErr = diff
		}
		if diff > 1e-3 {
			t.Errorf("y[%d] = %f, want %f (abs err %e)", i, got[i], ref[i], diff)
			if t.Failed() {
				break
			}
		}
	}
	t.Logf("max absolute error: %e", maxAbsErr)
}

func TestGemvQ5KF32_LargerMatrix(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	M, K := 512, 1024
	raw, x, ref := buildQ5KTestData(M, K)

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

	if err := GemvQ5KF32(devW, devX, devY, M, K, stream.Ptr()); err != nil {
		t.Fatalf("GemvQ5KF32: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, M)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, M*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy y: %v", err)
	}

	maxAbsErr := 0.0
	for i := range got {
		diff := math.Abs(float64(got[i] - ref[i]))
		if diff > maxAbsErr {
			maxAbsErr = diff
		}
		if diff > 1e-3 {
			t.Errorf("y[%d] = %f, want %f (abs err %e)", i, got[i], ref[i], diff)
			if t.Failed() {
				break
			}
		}
	}
	t.Logf("max absolute error: %e", maxAbsErr)
}

func TestGemvQ5KF32_MultipleSizes(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	cases := []struct {
		name string
		M, K int
	}{
		{"small_32x256", 32, 256},
		{"medium_64x512", 64, 512},
		{"square_256x256", 256, 256},
		{"wide_128x1024", 128, 1024},
		{"tall_1024x256", 1024, 256},
		{"large_512x2048", 512, 2048},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, x, ref := buildQ5KTestData(tc.M, tc.K)

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

			if err := GemvQ5KF32(devW, devX, devY, tc.M, tc.K, stream.Ptr()); err != nil {
				t.Fatalf("GemvQ5KF32: %v", err)
			}

			if err := stream.Synchronize(); err != nil {
				t.Fatalf("Synchronize: %v", err)
			}

			got := make([]float32, tc.M)
			if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, tc.M*4, cuda.MemcpyDeviceToHost); err != nil {
				t.Fatalf("Memcpy y: %v", err)
			}

			maxAbsErr := 0.0
			for i := range got {
				diff := math.Abs(float64(got[i] - ref[i]))
				if diff > maxAbsErr {
					maxAbsErr = diff
				}
				if diff > 1e-3 {
					t.Errorf("y[%d] = %f, want %f (abs err %e)", i, got[i], ref[i], diff)
					if t.Failed() {
						break
					}
				}
			}
			t.Logf("max absolute error: %e", maxAbsErr)
		})
	}
}

func BenchmarkGemvQ5KF32_4096(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}

	M, K := 4096, 4096
	raw, x, _ := buildQ5KTestData(M, K)

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
		_ = GemvQ5KF32(devW, devX, devY, M, K, stream.Ptr())
	}
	_ = stream.Synchronize()

	elapsed := b.Elapsed()
	flops := 2.0 * float64(M) * float64(K) * float64(b.N)
	gflops := flops / elapsed.Seconds() / 1e9
	b.ReportMetric(gflops, "GFLOPS")
}

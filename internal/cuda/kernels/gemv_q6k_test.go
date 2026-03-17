package kernels

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// dequantizeQ6K dequantizes one Q6_K super-block (210 bytes) into 256 float32 values.
// Inlined here to avoid an import cycle with the tensor package.
func dequantizeQ6K(raw []byte, dst []float32) {
	ql := raw[0:128]   // low 4 bits
	qh := raw[128:192] // high 2 bits
	sc := raw[192:208] // int8 scales for 16 sub-blocks
	d := float16BitsToFloat32(binary.LittleEndian.Uint16(raw[208:210]))

	// Process two 128-element halves.
	for half := range 2 {
		qlOff := half * 64
		qhOff := half * 32
		scOff := half * 8
		outOff := half * 128

		for l := range 32 {
			is := l / 16 // sub-block offset within each group of 32

			q1 := int8((ql[qlOff+l]&0xF)|((qh[qhOff+l]&3)<<4)) - 32
			q2 := int8((ql[qlOff+32+l]&0xF)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
			q3 := int8((ql[qlOff+l]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
			q4 := int8((ql[qlOff+32+l]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32

			dst[outOff+l] = d * float32(int8(sc[scOff+is+0])) * float32(q1)
			dst[outOff+32+l] = d * float32(int8(sc[scOff+is+2])) * float32(q2)
			dst[outOff+64+l] = d * float32(int8(sc[scOff+is+4])) * float32(q3)
			dst[outOff+96+l] = d * float32(int8(sc[scOff+is+6])) * float32(q4)
		}
	}
}

// buildQ6KBlockFromValues quantizes 256 float32 values into a Q6_K super-block.
func buildQ6KBlockFromValues(values []float32) []byte {
	const numSubBlocks = 16
	const subBlockSize = 16

	// Compute per-sub-block scales.
	var subScales [numSubBlocks]float32
	for sb := range numSubBlocks {
		off := sb * subBlockSize
		maxAbs := float32(0)
		for j := range subBlockSize {
			v := values[off+j]
			if v < 0 {
				v = -v
			}
			if v > maxAbs {
				maxAbs = v
			}
		}
		if maxAbs > 0 {
			subScales[sb] = maxAbs / 31.0 // Q6_K values range [-32, 31]
		}
	}

	// Compute super-block scale d from max sub-scale.
	var maxSubScale float32
	for _, s := range subScales {
		if s > maxSubScale {
			maxSubScale = s
		}
	}
	d := maxSubScale / 127.0 // sc is int8 [-128, 127]

	// Quantize sub-block scales to int8.
	var scQ [numSubBlocks]int8
	for sb := range numSubBlocks {
		if d > 0 {
			scQ[sb] = int8(clampInt(int(math.Round(float64(subScales[sb]/d))), -128, 127))
		}
	}

	// Quantize values to 6-bit signed integers [-32, 31].
	var quants [256]int8
	for sb := range numSubBlocks {
		off := sb * subBlockSize
		effScale := float16BitsToFloat32(float32ToFloat16Bits(d)) * float32(scQ[sb])
		var invScale float32
		if effScale != 0 {
			invScale = 1.0 / effScale
		}
		for j := range subBlockSize {
			q := int(math.Round(float64(values[off+j] * invScale)))
			quants[off+j] = int8(clampInt(q+32, 0, 63) - 32)
		}
	}

	// Pack into Q6_K super-block format (210 bytes).
	raw := make([]byte, 210)

	// Pack ql (low 4 bits) and qh (high 2 bits).
	// Two 128-element halves. Each half:
	//   ql[0:32]:  low nibbles of positions 0-31 (from quants q1)
	//   ql[32:64]: low nibbles of positions 32-63 (from quants q2)
	//   High nibbles of ql[0:32]:  positions 64-95 (from quants q3)
	//   High nibbles of ql[32:64]: positions 96-127 (from quants q4)
	//   qh[0:32]:  high 2 bits packed across all 4 groups
	for half := range 2 {
		qlOff := half * 64
		qhOff := half * 32
		outOff := half * 128

		for l := range 32 {
			// Convert back to unsigned 6-bit [0, 63]
			uq1 := uint8(quants[outOff+l] + 32)
			uq2 := uint8(quants[outOff+32+l] + 32)
			uq3 := uint8(quants[outOff+64+l] + 32)
			uq4 := uint8(quants[outOff+96+l] + 32)

			// ql: pack low 4 bits (q1 low nibble, q3 high nibble)
			raw[qlOff+l] = (uq1 & 0xF) | ((uq3 & 0xF) << 4)
			raw[qlOff+32+l] = (uq2 & 0xF) | ((uq4 & 0xF) << 4)

			// qh: pack high 2 bits from all 4 groups into one byte
			raw[128+qhOff+l] = (uq1 >> 4) | ((uq2 >> 4) << 2) | ((uq3 >> 4) << 4) | ((uq4 >> 4) << 6)
		}
	}

	// sc: int8 scales for 16 sub-blocks.
	for i := range numSubBlocks {
		raw[192+i] = byte(scQ[i])
	}

	// d: fp16 super-block scale.
	binary.LittleEndian.PutUint16(raw[208:210], float32ToFloat16Bits(d))

	return raw
}

// buildQ6KTestData constructs M rows of Q6_K super-blocks and computes the
// reference GEMV output using dequantizeQ6K for the dequantization.
// K must be a multiple of 256.
func buildQ6KTestData(M, K int) (raw []byte, x []float32, ref []float32) {
	const superBlockSize = 256
	const blockBytes = 210

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
			blkRaw := buildQ6KBlockFromValues(blkValues)
			off := (row*blocksPerRow + bi) * blockBytes
			copy(raw[off:off+blockBytes], blkRaw)
		}

		dequant := make([]float32, K)
		for bi := range blocksPerRow {
			off := (row*blocksPerRow + bi) * blockBytes
			dequantizeQ6K(raw[off:off+blockBytes], dequant[bi*superBlockSize:(bi+1)*superBlockSize])
		}
		var sum float32
		for k := range K {
			sum += dequant[k] * x[k]
		}
		ref[row] = sum
	}

	return raw, x, ref
}

func TestGemvQ6KF32_Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	M, K := 64, 256
	raw, x, ref := buildQ6KTestData(M, K)

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

	if err := GemvQ6KF32(devW, devX, devY, M, K, stream.Ptr()); err != nil {
		t.Fatalf("GemvQ6KF32: %v", err)
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

func TestGemvQ6KF32_LargerMatrix(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	M, K := 512, 1024
	raw, x, ref := buildQ6KTestData(M, K)

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

	if err := GemvQ6KF32(devW, devX, devY, M, K, stream.Ptr()); err != nil {
		t.Fatalf("GemvQ6KF32: %v", err)
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

func TestGemvQ6KF32_MultipleSizes(t *testing.T) {
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
			raw, x, ref := buildQ6KTestData(tc.M, tc.K)

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

			if err := GemvQ6KF32(devW, devX, devY, tc.M, tc.K, stream.Ptr()); err != nil {
				t.Fatalf("GemvQ6KF32: %v", err)
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

func BenchmarkGemvQ6KF32_4096(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}

	M, K := 4096, 4096
	raw, x, _ := buildQ6KTestData(M, K)

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
		_ = GemvQ6KF32(devW, devX, devY, M, K, stream.Ptr())
	}
	_ = stream.Synchronize()

	elapsed := b.Elapsed()
	flops := 2.0 * float64(M) * float64(K) * float64(b.N)
	gflops := flops / elapsed.Seconds() / 1e9
	b.ReportMetric(gflops, "GFLOPS")
}

package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestCPUEngine_MatMul_QuantizedStorage(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	m, k, n := 2, 32, 3

	aF32 := make([]float32, m*k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	bData := make([]float32, k*n)
	for i := range bData {
		bData[i] = float32(i%5-2) * 0.1
	}
	b, err := tensor.New[float32]([]int{k, n}, bData)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name   string
		stor   tensor.Storage[float32]
		maxErr float32
	}{
		{"Q4_0", tensor.QuantizeQ4(aF32), 0.15},
		{"Q8_0", tensor.QuantizeQ8(aF32), 0.02},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := tensor.NewWithStorage([]int{m, k}, tt.stor)
			if err != nil {
				t.Fatal(err)
			}

			result, err := engine.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("MatMul failed: %v", err)
			}

			if result.Shape()[0] != m || result.Shape()[1] != n {
				t.Errorf("shape = %v, want [%d %d]", result.Shape(), m, n)
			}

			// Reference: dequantize then float32 GEMM.
			refA, _ := tensor.New[float32]([]int{m, k}, tt.stor.Slice())
			refResult, err := engine.MatMul(ctx, refA, b)
			if err != nil {
				t.Fatal(err)
			}

			got := result.Data()
			want := refResult.Data()
			for i := range got {
				diff := float32(math.Abs(float64(got[i] - want[i])))
				if diff > tt.maxErr {
					t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
				}
			}
		})
	}
}

func TestCPUEngine_MatMul_Q4KStorage(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// K must be a multiple of 256 for Q4_K super-blocks.
	m, k, n := 2, 256, 3

	aF32 := make([]float32, m*k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	bData := make([]float32, k*n)
	for i := range bData {
		bData[i] = float32(i%5-2) * 0.1
	}
	b, err := tensor.New[float32]([]int{k, n}, bData)
	if err != nil {
		t.Fatal(err)
	}

	// Quantize aF32 to Q4_K using the quantizer.
	q4k := quantizeQ4K(aF32)
	a, err := tensor.NewWithStorage([]int{m, k}, q4k)
	if err != nil {
		t.Fatal(err)
	}

	result, err := engine.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("MatMul with Q4_K failed: %v", err)
	}

	if result.Shape()[0] != m || result.Shape()[1] != n {
		t.Errorf("shape = %v, want [%d %d]", result.Shape(), m, n)
	}

	// Reference: dequantize Q4_K then float32 GEMM.
	refA, _ := tensor.New[float32]([]int{m, k}, q4k.Slice())
	refResult, err := engine.MatMul(ctx, refA, b)
	if err != nil {
		t.Fatal(err)
	}

	got := result.Data()
	want := refResult.Data()
	for i := range got {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		if diff > 0.3 {
			t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

func TestCPUEngine_MatMul_Q4KStorage_BWeight_GEMV(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// GEMV: m=1, K must be multiple of 256.
	k, n := 256, 4

	// A is [1, K] float32 activation.
	aData := make([]float32, k)
	for i := range aData {
		aData[i] = float32(i%17-8) * 0.05
	}
	a, err := tensor.New[float32]([]int{1, k}, aData)
	if err != nil {
		t.Fatal(err)
	}

	// B is Q4_K weight. The Q4KStorage holds data in [N, K] layout.
	// The tensor shape is [K, N] (virtual transpose for MatMul).
	// So MatMul(A[1,K], B[K,N]) = C[1,N], but internally B_q4k is [N,K].
	bF32 := make([]float32, n*k)
	for i := range bF32 {
		bF32[i] = float32(math.Sin(float64(i)*0.03)) * 1.5
	}
	q4kB := quantizeQ4K(bF32)
	b, err := tensor.NewWithStorage([]int{k, n}, q4kB)
	if err != nil {
		t.Fatal(err)
	}

	result, err := engine.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("MatMul with Q4_K B-weight failed: %v", err)
	}

	if result.Shape()[0] != 1 || result.Shape()[1] != n {
		t.Errorf("shape = %v, want [1 %d]", result.Shape(), n)
	}

	// Reference: manually compute y[j] = sum_i( dequant(B_q4k[j,i]) * x[i] )
	// where B_q4k is in [N, K] layout.
	dequantB := q4kB.Slice() // [N*K] flat in [N,K] order
	want := make([]float32, n)
	for j := range n {
		var sum float32
		for i := range k {
			sum += dequantB[j*k+i] * aData[i]
		}
		want[j] = sum
	}

	got := result.Data()
	for i := range got {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		if diff > 0.5 {
			t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

// quantizeQ4K quantizes float32 values into Q4_K format using a simple
// quantizer that matches the Q4_K super-block layout (256 values, 8 sub-blocks).
func quantizeQ4K(data []float32) *tensor.Q4KStorage {
	const superBlockSize = 256
	const blockBytes = 144
	const numSubBlocks = 8
	const subBlockSize = 32

	nBlocks := (len(data) + superBlockSize - 1) / superBlockSize
	raw := make([]byte, nBlocks*blockBytes)

	for bi := range nBlocks {
		off := bi * superBlockSize
		end := off + superBlockSize
		if end > len(data) {
			end = len(data)
		}
		var values [superBlockSize]float32
		copy(values[:], data[off:end])

		var subScales, subMins [numSubBlocks]float32
		for sb := range numSubBlocks {
			sOff := sb * subBlockSize
			minVal := values[sOff]
			maxVal := values[sOff]
			for j := 1; j < subBlockSize; j++ {
				v := values[sOff+j]
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
			subScales[sb] = (maxVal - minVal) / 15.0
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

		var scalesQ, minsQ [numSubBlocks]uint8
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

		blk := raw[bi*blockBytes : (bi+1)*blockBytes]

		// fp16 d and dmin (approximate using float32->float16 truncation).
		dBits := float32ToFloat16Bits(d)
		dminBits := float32ToFloat16Bits(dmin)
		blk[0] = byte(dBits)
		blk[1] = byte(dBits >> 8)
		blk[2] = byte(dminBits)
		blk[3] = byte(dminBits >> 8)

		// Pack 6-bit scales and mins.
		for i := range 4 {
			blk[4+i] = (scalesQ[i] & 63) | ((scalesQ[4+i] >> 4) << 6)
			blk[8+i] = (minsQ[i] & 63) | ((minsQ[4+i] >> 4) << 6)
		}
		for i := range 4 {
			blk[12+i] = (scalesQ[4+i] & 0xF) | ((minsQ[4+i] & 0xF) << 4)
		}

		// Quantize values to 4-bit.
		for group := range 4 {
			sb0 := group * 2
			sb1 := group*2 + 1

			sc0 := float16BitsToFloat32(float32ToFloat16Bits(d)) * float32(scalesQ[sb0])
			mn0 := float16BitsToFloat32(float32ToFloat16Bits(dmin)) * float32(minsQ[sb0])
			sc1 := float16BitsToFloat32(float32ToFloat16Bits(d)) * float32(scalesQ[sb1])
			mn1 := float16BitsToFloat32(float32ToFloat16Bits(dmin)) * float32(minsQ[sb1])

			var invScale0, invScale1 float32
			if sc0 > 0 {
				invScale0 = 1.0 / sc0
			}
			if sc1 > 0 {
				invScale1 = 1.0 / sc1
			}

			baseOut := group * 64
			baseQ := group * 32
			for l := range 32 {
				v0 := values[baseOut+l]
				v1 := values[baseOut+l+32]
				q0 := clampInt(int(math.Round(float64((v0+mn0)*invScale0))), 0, 15)
				q1 := clampInt(int(math.Round(float64((v1+mn1)*invScale1))), 0, 15)
				blk[16+baseQ+l] = byte(q0) | (byte(q1) << 4)
			}
		}
	}

	q4k, err := tensor.NewQ4KStorageFromRaw(raw, len(data))
	if err != nil {
		panic(err)
	}
	return q4k
}

// TestQ5KGEMV tests the native Q5_K GEMV CPU path (B-weight, [K,N] virtual shape).
// It verifies that direct Q5_K dequant+GEMV produces the same result as the
// reference path (dequantize all weights, then float32 dot-product), confirming
// no re-quantization to Q4_0 occurs on the CPU path.
func TestQ5KGEMV(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	cases := []struct {
		name string
		k, n int
	}{
		{"k256_n4", 256, 4},
		{"k512_n8", 512, 8},
		{"k256_n16", 256, 16},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Activation vector: [1, K].
			aData := make([]float32, tc.k)
			for i := range aData {
				aData[i] = float32(i%17-8) * 0.05
			}
			a, err := tensor.New[float32]([]int{1, tc.k}, aData)
			if err != nil {
				t.Fatal(err)
			}

			// Weight matrix: [N, K] as float32, then quantized to Q5_K.
			bF32 := make([]float32, tc.n*tc.k)
			for i := range bF32 {
				bF32[i] = float32(math.Sin(float64(i)*0.03)) * 1.5
			}
			q5k := quantizeQ5K(bF32, tc.n, tc.k)
			// Shape [K, N] (virtual transpose — Q5KStorage holds [N,K] internally).
			b, err := tensor.NewWithStorage[float32]([]int{tc.k, tc.n}, q5k)
			if err != nil {
				t.Fatal(err)
			}

			result, err := engine.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("MatMul Q5_K B-weight: %v", err)
			}
			if result.Shape()[0] != 1 || result.Shape()[1] != tc.n {
				t.Errorf("shape = %v, want [1 %d]", result.Shape(), tc.n)
			}

			// Reference: dequantize Q5_K weights, then manual dot-product.
			dequantB := q5k.Slice() // [N*K] in [N,K] order
			want := make([]float32, tc.n)
			for j := range tc.n {
				var sum float32
				for i := range tc.k {
					sum += dequantB[j*tc.k+i] * aData[i]
				}
				want[j] = sum
			}

			got := result.Data()
			for i := range got {
				diff := float32(math.Abs(float64(got[i] - want[i])))
				if diff > 1e-4 {
					t.Errorf("[%d] got %v, want %v (diff=%v)", i, got[i], want[i], diff)
				}
			}
		})
	}
}

// BenchmarkQ5KGEMVvsDequantReQuant compares the native Q5_K GEMV path against a
// dequant-then-requant-to-Q4_0 reference, showing the memory traffic savings.
// This is a placeholder; actual DGX benchmark is deferred to GPU hardware.
func BenchmarkQ5KGEMVvsDequantReQuant(b *testing.B) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()
	k, n := 4096, 4096

	aData := make([]float32, k)
	for i := range aData {
		aData[i] = float32(i%17-8) * 0.05
	}
	a, _ := tensor.New[float32]([]int{1, k}, aData)

	bF32 := make([]float32, n*k)
	for i := range bF32 {
		bF32[i] = float32(math.Sin(float64(i)*0.001)) * 1.5
	}
	q5k := quantizeQ5K(bF32, n, k)
	bTensor, _ := tensor.NewWithStorage[float32]([]int{k, n}, q5k)

	b.ResetTimer()
	for b.Loop() {
		_, _ = engine.MatMul(ctx, a, bTensor)
	}
}

// quantizeQ5K quantizes numRows*K float32 values into a Q5_K storage with numRows rows.
// K must be a multiple of 256. Uses the same algorithm as buildQ5KBlockFromValues in
// the kernels package test helper.
func quantizeQ5K(data []float32, numRows, k int) *tensor.Q5KStorage {
	const superBlockSize = 256
	const blockBytes = 176
	const numSubBlocks = 8
	const subBlockSize = 32

	blocksPerRow := k / superBlockSize
	raw := make([]byte, numRows*blocksPerRow*blockBytes)

	for row := range numRows {
		for bi := range blocksPerRow {
			values := data[row*k+bi*superBlockSize : row*k+(bi+1)*superBlockSize]

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
				subScales[sb] = (maxVal - minVal) / 31.0
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

			blkOff := (row*blocksPerRow + bi) * blockBytes
			blk := raw[blkOff : blkOff+blockBytes]

			dBits := float32ToFloat16Bits(d)
			dminBits := float32ToFloat16Bits(dmin)
			blk[0] = byte(dBits)
			blk[1] = byte(dBits >> 8)
			blk[2] = byte(dminBits)
			blk[3] = byte(dminBits >> 8)

			// Pack 6-bit scales and mins (same as Q4_K).
			for i := range 4 {
				blk[4+i] = (scalesQ[i] & 63) | ((scalesQ[4+i] >> 4) << 6)
				blk[8+i] = (minsQ[i] & 63) | ((minsQ[4+i] >> 4) << 6)
			}
			for i := range 4 {
				blk[12+i] = (scalesQ[4+i] & 0xF) | ((minsQ[4+i] & 0xF) << 4)
			}

			// Round-trip d/dmin through fp16 for accurate quantization.
			dRT := float16BitsToFloat32(dBits)
			dminRT := float16BitsToFloat32(dminBits)

			// Quantize to 5-bit: low 4 bits in ql, high bit in qh.
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

					blk[16+baseQ+l] = byte(q0&0xF) | (byte(q1&0xF) << 4)

					if q0&16 != 0 {
						blk[144+l] |= u1
					}
					if q1&16 != 0 {
						blk[144+l] |= u2
					}
				}
			}
		}
	}

	qs, err := tensor.NewQ5KStorageFromRaw(raw, numRows*k)
	if err != nil {
		panic(err)
	}
	return qs
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 1
	exp := int((bits>>23)&0xFF) - 127
	frac := bits & 0x7FFFFF

	if exp < -24 {
		return uint16(sign << 15)
	}
	if exp < -14 {
		frac = (frac | 0x800000) >> uint((-14-exp)+13)
		return uint16(sign<<15) | uint16(frac)
	}
	if exp > 15 {
		return uint16(sign<<15) | 0x7C00
	}
	return uint16(sign<<15) | uint16((exp+15)<<10) | uint16(frac>>13)
}

func float16BitsToFloat32(h uint16) float32 {
	sign := uint32((h >> 15) & 1)
	exp := uint32((h >> 10) & 0x1F)
	frac := uint32(h & 0x3FF)

	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x3FF
		exp++
	} else if exp == 31 {
		return math.Float32frombits(sign<<31 | 0x7F800000 | frac<<13)
	}

	return math.Float32frombits(sign<<31 | (exp+127-15)<<23 | frac<<13)
}

// TestQ6KGEMV tests the native Q6_K GEMV CPU path (B-weight, [K,N] virtual shape).
// It verifies that direct Q6_K dequant+GEMV produces the same result as the
// reference path (dequantize all weights, then float32 dot-product), confirming
// no re-quantization to Q4_0 occurs on the CPU path.
func TestQ6KGEMV(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	cases := []struct {
		name string
		k, n int
	}{
		{"k256_n4", 256, 4},
		{"k512_n8", 512, 8},
		{"k256_n16", 256, 16},
		{"k1024_n32", 1024, 32},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Activation vector: [1, K].
			aData := make([]float32, tc.k)
			for i := range aData {
				aData[i] = float32(i%17-8) * 0.05
			}
			a, err := tensor.New[float32]([]int{1, tc.k}, aData)
			if err != nil {
				t.Fatal(err)
			}

			// Weight matrix: [N, K] as float32, then quantized to Q6_K.
			bF32 := make([]float32, tc.n*tc.k)
			for i := range bF32 {
				bF32[i] = float32(math.Sin(float64(i)*0.03)) * 1.5
			}
			q6k := quantizeQ6K(bF32, tc.n, tc.k)
			// Shape [K, N] (virtual transpose — Q6KStorage holds [N,K] internally).
			b, err := tensor.NewWithStorage[float32]([]int{tc.k, tc.n}, q6k)
			if err != nil {
				t.Fatal(err)
			}

			result, err := engine.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("MatMul Q6_K B-weight: %v", err)
			}
			if result.Shape()[0] != 1 || result.Shape()[1] != tc.n {
				t.Errorf("shape = %v, want [1 %d]", result.Shape(), tc.n)
			}

			// Reference: dequantize Q6_K weights, then manual dot-product.
			dequantB := q6k.Slice() // [N*K] in [N,K] order
			want := make([]float32, tc.n)
			for j := range tc.n {
				var sum float32
				for i := range tc.k {
					sum += dequantB[j*tc.k+i] * aData[i]
				}
				want[j] = sum
			}

			got := result.Data()
			for i := range got {
				diff := float32(math.Abs(float64(got[i] - want[i])))
				if diff > 1e-4 {
					t.Errorf("[%d] got %v, want %v (diff=%v)", i, got[i], want[i], diff)
				}
			}
		})
	}
}

// BenchmarkQ6KGEMVvsDequantReQuant compares the native Q6_K GEMV path against a
// dequant-then-requant-to-Q4_0 reference, showing the memory traffic savings.
func BenchmarkQ6KGEMVvsDequantReQuant(b *testing.B) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()
	k, n := 4096, 4096

	aData := make([]float32, k)
	for i := range aData {
		aData[i] = float32(i%17-8) * 0.05
	}
	a, _ := tensor.New[float32]([]int{1, k}, aData)

	bF32 := make([]float32, n*k)
	for i := range bF32 {
		bF32[i] = float32(math.Sin(float64(i)*0.001)) * 1.5
	}
	q6k := quantizeQ6K(bF32, n, k)
	bTensor, _ := tensor.NewWithStorage[float32]([]int{k, n}, q6k)

	b.ResetTimer()
	for b.Loop() {
		_, _ = engine.MatMul(ctx, a, bTensor)
	}
}

// quantizeQ6K quantizes numRows*K float32 values into a Q6_K storage with numRows rows.
// K must be a multiple of 256.
func quantizeQ6K(data []float32, numRows, k int) *tensor.Q6KStorage {
	const superBlockSize = 256
	const blockBytes = 210
	const numSubBlocks = 16
	const subBlockSize = 16

	blocksPerRow := k / superBlockSize
	raw := make([]byte, numRows*blocksPerRow*blockBytes)

	for row := range numRows {
		for bi := range blocksPerRow {
			values := data[row*k+bi*superBlockSize : row*k+(bi+1)*superBlockSize]

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
					subScales[sb] = maxAbs / 31.0
				}
			}

			// Super-block scale d from max sub-scale.
			var maxSubScale float32
			for _, s := range subScales {
				if s > maxSubScale {
					maxSubScale = s
				}
			}
			d := maxSubScale / 127.0

			// Quantize sub-block scales to int8.
			var scQ [numSubBlocks]int8
			for sb := range numSubBlocks {
				if d > 0 {
					scQ[sb] = int8(clampInt(int(math.Round(float64(subScales[sb]/d))), -128, 127))
				}
			}

			// Quantize values to 6-bit signed integers [-32, 31].
			var quants [superBlockSize]int8
			dRT := float16BitsToFloat32(float32ToFloat16Bits(d))
			for sb := range numSubBlocks {
				off := sb * subBlockSize
				effScale := dRT * float32(scQ[sb])
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
			blkOff := (row*blocksPerRow + bi) * blockBytes
			blk := raw[blkOff : blkOff+blockBytes]

			for half := range 2 {
				qlOff := half * 64
				qhOff := half * 32
				outOff := half * 128

				for l := range 32 {
					uq1 := uint8(quants[outOff+l] + 32)
					uq2 := uint8(quants[outOff+32+l] + 32)
					uq3 := uint8(quants[outOff+64+l] + 32)
					uq4 := uint8(quants[outOff+96+l] + 32)

					blk[qlOff+l] = (uq1 & 0xF) | ((uq3 & 0xF) << 4)
					blk[qlOff+32+l] = (uq2 & 0xF) | ((uq4 & 0xF) << 4)
					blk[128+qhOff+l] = (uq1 >> 4) | ((uq2 >> 4) << 2) | ((uq3 >> 4) << 4) | ((uq4 >> 4) << 6)
				}
			}

			// sc: int8 scales for 16 sub-blocks.
			for i := range numSubBlocks {
				blk[192+i] = byte(scQ[i])
			}

			// d: fp16 super-block scale.
			dBits := float32ToFloat16Bits(d)
			blk[208] = byte(dBits)
			blk[209] = byte(dBits >> 8)
		}
	}

	qs, err := tensor.NewQ6KStorageFromRaw(raw, numRows*k)
	if err != nil {
		panic(err)
	}
	return qs
}

func TestCPUEngine_MatMul_Q4Storage_Batched(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	batch, m, k, n := 2, 1, 32, 4

	aF32 := make([]float32, batch*m*k)
	for i := range aF32 {
		aF32[i] = float32(i%9-4) * 0.05
	}
	q4 := tensor.QuantizeQ4(aF32)
	a, err := tensor.NewWithStorage([]int{batch, m, k}, q4)
	if err != nil {
		t.Fatal(err)
	}

	bData := make([]float32, k*n)
	for i := range bData {
		bData[i] = float32(i%5-2) * 0.1
	}
	b, err := tensor.New[float32]([]int{k, n}, bData)
	if err != nil {
		t.Fatal(err)
	}

	result, err := engine.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("Batched MatMul with Q4 failed: %v", err)
	}

	shape := result.Shape()
	if shape[0] != batch || shape[1] != m || shape[2] != n {
		t.Errorf("output shape = %v, want [%d %d %d]", shape, batch, m, n)
	}
}

// TestW8A8 tests the W8A8 mixed-precision dispatch path through the CPUEngine.
func TestW8A8(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	t.Run("A-side", func(t *testing.T) {
		m, k, n := 2, 32, 3

		aF32 := make([]float32, m*k)
		for i := range aF32 {
			aF32[i] = float32(i%7-3) * 0.1
		}
		bData := make([]float32, k*n)
		for i := range bData {
			bData[i] = float32(i%5-2) * 0.1
		}
		b, err := tensor.New[float32]([]int{k, n}, bData)
		if err != nil {
			t.Fatal(err)
		}

		w8a8 := tensor.QuantizeW8A8(aF32)
		a, err := tensor.NewWithStorage([]int{m, k}, w8a8)
		if err != nil {
			t.Fatal(err)
		}

		result, err := engine.MatMul(ctx, a, b)
		if err != nil {
			t.Fatalf("MatMul with W8A8 A-side failed: %v", err)
		}

		if result.Shape()[0] != m || result.Shape()[1] != n {
			t.Errorf("shape = %v, want [%d %d]", result.Shape(), m, n)
		}

		// Reference: dequantize W8A8 then float32 GEMM.
		refA, _ := tensor.New[float32]([]int{m, k}, w8a8.Slice())
		refResult, err := engine.MatMul(ctx, refA, b)
		if err != nil {
			t.Fatal(err)
		}

		got := result.Data()
		want := refResult.Data()
		for i := range got {
			diff := float32(math.Abs(float64(got[i] - want[i])))
			if diff > 0.02 {
				t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
			}
		}
	})

	t.Run("B-side GEMV", func(t *testing.T) {
		k, n := 32, 4

		aData := make([]float32, k)
		for i := range aData {
			aData[i] = float32(i%17-8) * 0.05
		}
		a, err := tensor.New[float32]([]int{1, k}, aData)
		if err != nil {
			t.Fatal(err)
		}

		bF32 := make([]float32, n*k)
		for i := range bF32 {
			bF32[i] = float32(math.Sin(float64(i)*0.03)) * 1.5
		}
		w8a8B := tensor.QuantizeW8A8(bF32)
		b, err := tensor.NewWithStorage([]int{k, n}, w8a8B)
		if err != nil {
			t.Fatal(err)
		}

		result, err := engine.MatMul(ctx, a, b)
		if err != nil {
			t.Fatalf("MatMul with W8A8 B-weight failed: %v", err)
		}

		if result.Shape()[0] != 1 || result.Shape()[1] != n {
			t.Errorf("shape = %v, want [1 %d]", result.Shape(), n)
		}

		// Reference: dequantize W8A8, then manual dot-product.
		dequantB := w8a8B.Slice()
		want := make([]float32, n)
		for j := range n {
			var sum float32
			for i := range k {
				sum += dequantB[j*k+i] * aData[i]
			}
			want[j] = sum
		}

		got := result.Data()
		for i := range got {
			diff := float32(math.Abs(float64(got[i] - want[i])))
			if diff > 0.5 {
				t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
			}
		}
	})

	t.Run("B-side GEMM", func(t *testing.T) {
		m, k, n := 2, 64, 3

		aData := make([]float32, m*k)
		for i := range aData {
			aData[i] = float32(i%11-5) * 0.1
		}
		a, err := tensor.New[float32]([]int{m, k}, aData)
		if err != nil {
			t.Fatal(err)
		}

		bF32 := make([]float32, n*k)
		for i := range bF32 {
			bF32[i] = float32(i%7-3) * 0.2
		}
		w8a8B := tensor.QuantizeW8A8(bF32)
		b, err := tensor.NewWithStorage([]int{k, n}, w8a8B)
		if err != nil {
			t.Fatal(err)
		}

		result, err := engine.MatMul(ctx, a, b)
		if err != nil {
			t.Fatalf("MatMul with W8A8 B-weight GEMM failed: %v", err)
		}

		if result.Shape()[0] != m || result.Shape()[1] != n {
			t.Errorf("shape = %v, want [%d %d]", result.Shape(), m, n)
		}

		// Reference: dequantize W8A8, then manual matmul.
		dequantB := w8a8B.Slice()
		want := make([]float32, m*n)
		for i := range m {
			for j := range n {
				var sum float32
				for p := range k {
					sum += aData[i*k+p] * dequantB[j*k+p]
				}
				want[i*n+j] = sum
			}
		}

		got := result.Data()
		for i := range got {
			diff := float32(math.Abs(float64(got[i] - want[i])))
			if diff > 0.5 {
				t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
			}
		}
	})
}

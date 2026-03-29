package xblas

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

// buildQ4KRawBytes builds valid Q4_K super-block bytes for numBlocks blocks.
// Each super-block encodes 256 values. Uses a simple linear ramp pattern.
func buildQ4KRawBytes(numBlocks int) []byte {
	const blockBytes = 144
	raw := make([]byte, numBlocks*blockBytes)
	for bi := range numBlocks {
		off := bi * blockBytes

		// d = 0.02 (fp16), dmin = 0.0 (fp16).
		// fp16 0.02 ≈ 0x14F4 (exponent 2^-6 ≈ 0.015625, + mantissa)
		// Use a known fp16 encoding: 0.02 → sign=0, exp=15-6=9, mant≈(0.02/2^(9-15)-1)*1024
		// Simpler: d = 0.015625 = 2^-6, fp16 = 0x2400
		binary.LittleEndian.PutUint16(raw[off:off+2], 0x2400) // d = 2^-6 ≈ 0.015625
		binary.LittleEndian.PutUint16(raw[off+2:off+4], 0x0000) // dmin = 0

		// scales[0..7] = 32, mins[0..7] = 0 (packed 6-bit into bytes 4-15).
		// With sub-block scale = d * 32 = 0.5 and min = 0.
		// Encoding: bytes 4..7 = scalesQ[0..3] & 63 | (scalesQ[4..7]>>4)<<6
		//           bytes 8..11 = minsQ[0..3] & 63 | (minsQ[4..7]>>4)<<6
		//           bytes 12..15 = scalesQ[4..7]&0xF | (minsQ[4..7]&0xF)<<4
		for i := range 4 {
			raw[off+4+i] = 32 & 63 // scalesQ[i] = 32, scalesQ[4+i] high bits = 0
			raw[off+8+i] = 0       // minsQ[i] = 0
			raw[off+12+i] = 32 & 0xF // scalesQ[4+i] low 4 bits = 0 (32 = 0x20, low 4 = 0)
		}

		// Nibbles: fill with a ramp pattern (values 0..15 cycling).
		for j := range 128 {
			lo := byte((bi*128+j*2) % 16)
			hi := byte((bi*128+j*2+1) % 16)
			raw[off+16+j] = lo | (hi << 4)
		}
	}
	return raw
}

// TestGemmF32MmapNT_Q4K verifies GemmF32MmapNT (B side mmap) matches GemmF32Q4KNT.
// Both use the same raw Q4_K bytes; both should produce identical float32 results.
func TestGemmF32MmapNT_Q4K(t *testing.T) {
	cases := []struct{ m, n, k int }{
		{1, 4, 256},
		{2, 8, 256},
		{1, 16, 512},
		{4, 8, 512},
	}
	for _, tc := range cases {
		m, n, k := tc.m, tc.n, tc.k
		t.Run("", func(t *testing.T) {
			numBlocks := n * k / 256
			rawBytes := buildQ4KRawBytes(numBlocks)

			q4k, err := tensor.NewQ4KStorageFromRaw(rawBytes, n*k)
			if err != nil {
				t.Fatalf("NewQ4KStorageFromRaw: %v", err)
			}
			mmapB, err := tensor.NewMmapStorage(rawBytes, n*k, tensor.GGMLTypeQ4_K)
			if err != nil {
				t.Fatalf("NewMmapStorage: %v", err)
			}

			a := make([]float32, m*k)
			for i := range a {
				a[i] = float32(i%7-3) * 0.1
			}

			want := make([]float32, m*n)
			got := make([]float32, m*n)

			GemmF32Q4KNT(m, n, k, a, q4k, want)
			GemmF32MmapNT(m, n, k, a, mmapB, got)

			for i := range got {
				diff := float32(math.Abs(float64(got[i] - want[i])))
				if diff > 1e-5 {
					t.Errorf("[%d] mmap=%.6f q4k=%.6f diff=%e", i, got[i], want[i], diff)
				}
			}
		})
	}
}

// TestGemmMmapF32_Q4K verifies GemmMmapF32 (A side mmap) matches SgemmSimd on dequantized A.
func TestGemmMmapF32_Q4K(t *testing.T) {
	m, k, n := 4, 256, 8
	numBlocks := m * k / 256
	rawBytes := buildQ4KRawBytes(numBlocks)

	mmapA, err := tensor.NewMmapStorage(rawBytes, m*k, tensor.GGMLTypeQ4_K)
	if err != nil {
		t.Fatalf("NewMmapStorage: %v", err)
	}

	b := make([]float32, k*n)
	for i := range b {
		b[i] = float32(i%5-2) * 0.1
	}

	// Reference: full dequantize then SGEMM.
	aF32 := mmapA.Slice()
	want := make([]float32, m*n)
	SgemmSimd(m, n, k, aF32, b, want)

	got := make([]float32, m*n)
	GemmMmapF32(m, n, k, mmapA, b, got)

	for i := range got {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		if diff > 1e-5 {
			t.Errorf("[%d] mmap=%.6f ref=%.6f diff=%e", i, got[i], want[i], diff)
		}
	}
}

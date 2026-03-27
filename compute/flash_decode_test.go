package compute

import (
	"math"
	"testing"
)

func TestFlashDecode_CPU(t *testing.T) {
	type testCase struct {
		name       string
		batch      int
		numQHeads  int
		numKVHeads int
		kvLen      int
		headDim    int
	}
	cases := []testCase{
		{"small", 1, 4, 4, 128, 64},
		{"medium", 1, 8, 8, 1024, 128},
		{"long_kv", 1, 4, 4, 8192, 64},
		{"GQA_4to1", 1, 8, 2, 1024, 128},
		{"GQA_8to4", 2, 8, 4, 256, 64},
		{"batch2", 2, 4, 4, 512, 64},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			Q, K, V := makeFlashDecodeInputs(tc.batch, tc.numQHeads, tc.numKVHeads, tc.kvLen, tc.headDim)

			// Reference output using FlashDecode.
			numBH := tc.batch * tc.numQHeads
			expected := make([]float32, numBH*tc.headDim)
			if err := FlashDecode(Q, K, V, expected, tc.batch, tc.numQHeads, tc.numKVHeads, tc.kvLen, tc.headDim); err != nil {
				t.Fatalf("FlashDecode: %v", err)
			}

			// Verify output is not all zeros.
			allZero := true
			for _, v := range expected {
				if v != 0 {
					allZero = false
					break
				}
			}
			if allZero {
				t.Fatal("FlashDecode output is all zeros")
			}

			// Verify finite values as a basic sanity check.
			for i, v := range expected {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("FlashDecode output[%d] = %v (not finite)", i, v)
				}
			}
		})
	}
}

func TestFlashDecodeSplitKV_MatchesReference(t *testing.T) {
	type testCase struct {
		name       string
		batch      int
		numQHeads  int
		numKVHeads int
		kvLen      int
		headDim    int
		chunkSize  int
	}
	cases := []testCase{
		{"kvLen128_chunk64", 1, 4, 4, 128, 64, 64},
		{"kvLen128_chunk32", 1, 4, 4, 128, 64, 32},
		{"kvLen1024_chunk256", 1, 8, 8, 1024, 128, 256},
		{"kvLen1024_chunk128", 1, 8, 8, 1024, 128, 128},
		{"kvLen8192_chunk512", 1, 4, 4, 8192, 64, 512},
		{"kvLen8192_chunk1024", 1, 4, 4, 8192, 64, 1024},
		{"GQA_kvLen1024_chunk128", 1, 8, 2, 1024, 128, 128},
		{"batch2_kvLen512_chunk64", 2, 4, 4, 512, 64, 64},
		{"chunk_equals_kvLen", 1, 4, 4, 256, 64, 256},
		{"chunk_exceeds_kvLen", 1, 4, 4, 100, 64, 256},
		{"single_kv", 1, 4, 4, 1, 64, 64},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			Q, K, V := makeFlashDecodeInputs(tc.batch, tc.numQHeads, tc.numKVHeads, tc.kvLen, tc.headDim)

			numBH := tc.batch * tc.numQHeads
			expected := make([]float32, numBH*tc.headDim)
			got := make([]float32, numBH*tc.headDim)

			if err := FlashDecode(Q, K, V, expected, tc.batch, tc.numQHeads, tc.numKVHeads, tc.kvLen, tc.headDim); err != nil {
				t.Fatalf("FlashDecode: %v", err)
			}
			if err := FlashDecodeSplitKV(Q, K, V, got, tc.batch, tc.numQHeads, tc.numKVHeads, tc.kvLen, tc.headDim, tc.chunkSize); err != nil {
				t.Fatalf("FlashDecodeSplitKV: %v", err)
			}

			tol := 1e-4
			mismatches := 0
			for i := range expected {
				diff := math.Abs(float64(got[i] - expected[i]))
				if diff > tol {
					if mismatches < 5 {
						t.Errorf("output[%d] = %f, want %f (diff %e)",
							i, got[i], expected[i], diff)
					}
					mismatches++
				}
			}
			if mismatches > 0 {
				t.Errorf("total mismatches: %d / %d", mismatches, len(expected))
			}
		})
	}
}

func TestFlashDecode_InvalidInputs(t *testing.T) {
	t.Run("non_multiple_heads", func(t *testing.T) {
		Q := make([]float32, 64)
		K := make([]float32, 64)
		V := make([]float32, 64)
		O := make([]float32, 64)
		err := FlashDecode(Q, K, V, O, 1, 5, 3, 1, 64)
		if err == nil {
			t.Fatal("expected error for non-multiple heads")
		}
	})

	t.Run("splitkv_zero_chunk", func(t *testing.T) {
		Q := make([]float32, 64)
		K := make([]float32, 64)
		V := make([]float32, 64)
		O := make([]float32, 64)
		err := FlashDecodeSplitKV(Q, K, V, O, 1, 1, 1, 1, 64, 0)
		if err == nil {
			t.Fatal("expected error for zero chunkSize")
		}
	})
}

// makeFlashDecodeInputs generates deterministic test data.
func makeFlashDecodeInputs(batch, numQHeads, numKVHeads, kvLen, headDim int) (Q, K, V []float32) {
	numBH := batch * numQHeads
	kvDim := numKVHeads * headDim

	Q = make([]float32, numBH*headDim)
	K = make([]float32, batch*kvLen*kvDim)
	V = make([]float32, batch*kvLen*kvDim)

	for i := range Q {
		Q[i] = float32(i%7-3) * 0.1
	}
	for i := range K {
		K[i] = float32(i%5-2) * 0.1
		V[i] = float32(i%11-5) * 0.1
	}
	return
}

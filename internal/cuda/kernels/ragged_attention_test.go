//go:build !cuda

package kernels

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// naiveRaggedAttention computes CPU reference for ragged batched attention.
// Q: [totalTokens, numQHeads, headDim] packed.
// K: [totalTokens, numKVHeads, headDim] packed.
// V: [totalTokens, numKVHeads, headDim] packed.
// seqLens: length of each sequence.
// Block-diagonal mask: each token only attends to tokens in its own sequence.
func naiveRaggedAttention(
	Q, K, V []float32,
	seqLens []int,
	numQHeads, numKVHeads, headDim int,
) []float32 {
	// Compute cumulative seq lens.
	cumSeqLens := make([]int, len(seqLens))
	totalTokens := 0
	for i, l := range seqLens {
		cumSeqLens[i] = totalTokens
		totalTokens += l
	}

	O := make([]float32, totalTokens*numQHeads*headDim)
	scale := 1.0 / math.Sqrt(float64(headDim))
	headRatio := numQHeads / numKVHeads

	for seqIdx, seqLen := range seqLens {
		seqStart := cumSeqLens[seqIdx]

		for t := 0; t < seqLen; t++ {
			tokenGlobal := seqStart + t

			for qh := 0; qh < numQHeads; qh++ {
				kvHead := qh / headRatio

				// Compute attention scores for this query against all keys in same sequence.
				scores := make([]float64, seqLen)
				maxScore := -math.MaxFloat64

				for j := 0; j < seqLen; j++ {
					kvToken := seqStart + j
					dot := 0.0
					for d := 0; d < headDim; d++ {
						qVal := float64(Q[(tokenGlobal*numQHeads+qh)*headDim+d])
						kVal := float64(K[(kvToken*numKVHeads+kvHead)*headDim+d])
						dot += qVal * kVal
					}
					scores[j] = dot * scale
					if scores[j] > maxScore {
						maxScore = scores[j]
					}
				}

				// Softmax.
				sum := 0.0
				for j := range scores {
					scores[j] = math.Exp(scores[j] - maxScore)
					sum += scores[j]
				}
				for j := range scores {
					scores[j] /= sum
				}

				// Weighted sum of V.
				oIdx := (tokenGlobal*numQHeads + qh) * headDim
				for d := 0; d < headDim; d++ {
					acc := 0.0
					for j := 0; j < seqLen; j++ {
						kvToken := seqStart + j
						vVal := float64(V[(kvToken*numKVHeads+kvHead)*headDim+d])
						acc += scores[j] * vVal
					}
					O[oIdx+d] = float32(acc)
				}
			}
		}
	}
	return O
}

// TestRaggedAttention verifies that two sequences of different lengths packed
// in a batch produce correct results and no cross-sequence attention contamination.
func TestRaggedAttention(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}
	if !IsRaggedAttentionSupported() {
		t.Skip("ragged attention kernel not loaded")
	}

	numQHeads := 4
	numKVHeads := 4
	headDim := 64
	seqLens := []int{3, 5}
	batch := len(seqLens)

	// Compute cumulative seq lens and total tokens.
	cumSeqLens := make([]int32, batch)
	totalTokens := 0
	for i, l := range seqLens {
		cumSeqLens[i] = int32(totalTokens)
		totalTokens += l
	}

	// Create Q, K, V on host.
	qSize := totalTokens * numQHeads * headDim
	kSize := totalTokens * numKVHeads * headDim
	Q := make([]float32, qSize)
	K := make([]float32, kSize)
	V := make([]float32, kSize)
	for i := range Q {
		Q[i] = float32(i%7-3) * 0.1
	}
	for i := range K {
		K[i] = float32(i%5-2) * 0.1
	}
	for i := range V {
		V[i] = float32(i%11-5) * 0.1
	}

	// CPU reference.
	expected := naiveRaggedAttention(Q, K, V, seqLens, numQHeads, numKVHeads, headDim)

	// Allocate device memory.
	devQ, err := cuda.Malloc(qSize * 4)
	if err != nil {
		t.Fatalf("Malloc Q: %v", err)
	}
	defer func() { _ = cuda.Free(devQ) }()

	devK, err := cuda.Malloc(kSize * 4)
	if err != nil {
		t.Fatalf("Malloc K: %v", err)
	}
	defer func() { _ = cuda.Free(devK) }()

	devV, err := cuda.Malloc(kSize * 4)
	if err != nil {
		t.Fatalf("Malloc V: %v", err)
	}
	defer func() { _ = cuda.Free(devV) }()

	devO, err := cuda.Malloc(qSize * 4)
	if err != nil {
		t.Fatalf("Malloc O: %v", err)
	}
	defer func() { _ = cuda.Free(devO) }()

	_ = cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), qSize*4, cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devK, unsafe.Pointer(&K[0]), kSize*4, cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devV, unsafe.Pointer(&V[0]), kSize*4, cuda.MemcpyHostToDevice)

	// Upload seq_lens and cum_seq_lens.
	seqLensI32 := make([]int32, batch)
	for i, l := range seqLens {
		seqLensI32[i] = int32(l)
	}

	devSeqLens, err := cuda.Malloc(batch * 4)
	if err != nil {
		t.Fatalf("Malloc seqLens: %v", err)
	}
	defer func() { _ = cuda.Free(devSeqLens) }()
	_ = cuda.Memcpy(devSeqLens, unsafe.Pointer(&seqLensI32[0]), batch*4, cuda.MemcpyHostToDevice)

	devCumSeqLens, err := cuda.Malloc(batch * 4)
	if err != nil {
		t.Fatalf("Malloc cumSeqLens: %v", err)
	}
	defer func() { _ = cuda.Free(devCumSeqLens) }()
	_ = cuda.Memcpy(devCumSeqLens, unsafe.Pointer(&cumSeqLens[0]), batch*4, cuda.MemcpyHostToDevice)

	// Create stream and run kernel.
	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	if err := RaggedAttentionForward(
		devQ, devK, devV, devO,
		devSeqLens, devCumSeqLens,
		batch, numQHeads, numKVHeads, headDim,
		stream.Ptr(),
	); err != nil {
		t.Fatalf("RaggedAttentionForward: %v", err)
	}
	_ = stream.Synchronize()

	// Copy result back.
	result := make([]float32, qSize)
	_ = cuda.Memcpy(unsafe.Pointer(&result[0]), devO, qSize*4, cuda.MemcpyDeviceToHost)

	// Compare with tolerance.
	tol := 1e-4
	mismatches := 0
	for i := range result {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > tol {
			if mismatches < 5 {
				t.Errorf("output[%d] = %f, want %f (diff %e)", i, result[i], expected[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches: %d / %d", mismatches, qSize)
	}

	// Cross-contamination check: run with seq2's Q zeroed out.
	// If there's no cross-contamination, seq1's output should be identical.
	Q2 := make([]float32, qSize)
	copy(Q2, Q)
	// Zero out seq2's Q values (tokens 3..7, all heads).
	for t := seqLens[0]; t < totalTokens; t++ {
		for h := 0; h < numQHeads; h++ {
			for d := 0; d < headDim; d++ {
				Q2[(t*numQHeads+h)*headDim+d] = 0
			}
		}
	}

	expected2 := naiveRaggedAttention(Q2, K, V, seqLens, numQHeads, numKVHeads, headDim)

	_ = cuda.Memcpy(devQ, unsafe.Pointer(&Q2[0]), qSize*4, cuda.MemcpyHostToDevice)

	if err := RaggedAttentionForward(
		devQ, devK, devV, devO,
		devSeqLens, devCumSeqLens,
		batch, numQHeads, numKVHeads, headDim,
		stream.Ptr(),
	); err != nil {
		t.Fatalf("RaggedAttentionForward (cross-contamination): %v", err)
	}
	_ = stream.Synchronize()

	result2 := make([]float32, qSize)
	_ = cuda.Memcpy(unsafe.Pointer(&result2[0]), devO, qSize*4, cuda.MemcpyDeviceToHost)

	// Seq1's output (first 3 tokens) should match the original run exactly,
	// because seq1 never attends to seq2's tokens.
	seq1Elems := seqLens[0] * numQHeads * headDim
	for i := 0; i < seq1Elems; i++ {
		diff := math.Abs(float64(result2[i] - result[i]))
		if diff > 1e-6 {
			t.Errorf("cross-contamination at output[%d]: original=%f, zeroed_q2=%f (diff %e)",
				i, result[i], result2[i], diff)
			break
		}
	}

	// Also verify seq2's output matches CPU reference with zeroed Q.
	seq2Start := seqLens[0] * numQHeads * headDim
	mismatches2 := 0
	for i := seq2Start; i < qSize; i++ {
		diff := math.Abs(float64(result2[i] - expected2[i]))
		if diff > tol {
			if mismatches2 < 5 {
				t.Errorf("zeroed_q2 output[%d] = %f, want %f (diff %e)", i, result2[i], expected2[i], diff)
			}
			mismatches2++
		}
	}
	if mismatches2 > 0 {
		t.Errorf("zeroed Q2 total mismatches: %d / %d", mismatches2, qSize-seq2Start)
	}
}

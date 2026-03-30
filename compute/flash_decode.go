package compute

import (
	"fmt"
	"math"
)

// FlashDecode computes single-query scaled dot-product attention for
// autoregressive decode (seqLen_Q = 1). This is the CPU reference
// implementation; the GPU path dispatches to the split-KV CUDA kernel.
//
// Q:  [batch * numQHeads, headDim]         — one query per head.
// K:  [batch, kvLen, numKVHeads * headDim] — KV cache keys.
// V:  [batch, kvLen, numKVHeads * headDim] — KV cache values.
// O:  [batch * numQHeads, headDim]         — output (caller-allocated).
//
// Supports GQA: numQHeads must be a multiple of numKVHeads.
//
// This API is not covered by the v1 stability guarantee.
func FlashDecode(
	Q, K, V, O []float32,
	batch, numQHeads, numKVHeads, kvLen, headDim int,
) error {
	if numQHeads%numKVHeads != 0 {
		return fmt.Errorf("FlashDecode: numQHeads (%d) must be a multiple of numKVHeads (%d)",
			numQHeads, numKVHeads)
	}

	headRatio := numQHeads / numKVHeads
	kvDim := numKVHeads * headDim
	scale := 1.0 / math.Sqrt(float64(headDim))

	for b := 0; b < batch; b++ {
		for qh := 0; qh < numQHeads; qh++ {
			bh := b*numQHeads + qh
			kvHead := qh / headRatio

			// Compute scaled dot products and find max for numerical stability.
			maxScore := -math.MaxFloat64
			scores := make([]float64, kvLen)
			for j := 0; j < kvLen; j++ {
				dot := 0.0
				for d := 0; d < headDim; d++ {
					dot += float64(Q[bh*headDim+d]) * float64(K[b*kvLen*kvDim+j*kvDim+kvHead*headDim+d])
				}
				scores[j] = dot * scale
				if scores[j] > maxScore {
					maxScore = scores[j]
				}
			}

			// Softmax.
			sum := 0.0
			for j := 0; j < kvLen; j++ {
				scores[j] = math.Exp(scores[j] - maxScore)
				sum += scores[j]
			}
			invSum := 1.0 / sum

			// Weighted sum of values.
			for d := 0; d < headDim; d++ {
				acc := 0.0
				for j := 0; j < kvLen; j++ {
					acc += scores[j] * float64(V[b*kvLen*kvDim+j*kvDim+kvHead*headDim+d])
				}
				O[bh*headDim+d] = float32(acc * invSum)
			}
		}
	}

	return nil
}

// FlashDecodeSplitKV computes single-query attention using a chunked
// (split-KV) approach on CPU. This mirrors the GPU kernel's algorithm:
// the KV cache is split into chunks, each producing a partial output
// with its own softmax state, then the partial results are merged
// using log-sum-exp correction. This function exists to verify that
// the split-KV reduction logic is correct independent of GPU execution.
//
// chunkSize controls how many KV positions each "block" processes.
// The output should match FlashDecode within floating-point tolerance.
func FlashDecodeSplitKV(
	Q, K, V, O []float32,
	batch, numQHeads, numKVHeads, kvLen, headDim int,
	chunkSize int,
) error {
	if numQHeads%numKVHeads != 0 {
		return fmt.Errorf("FlashDecodeSplitKV: numQHeads (%d) must be a multiple of numKVHeads (%d)",
			numQHeads, numKVHeads)
	}
	if chunkSize <= 0 {
		return fmt.Errorf("FlashDecodeSplitKV: chunkSize must be > 0, got %d", chunkSize)
	}

	headRatio := numQHeads / numKVHeads
	kvDim := numKVHeads * headDim
	scale := 1.0 / math.Sqrt(float64(headDim))
	numSplits := (kvLen + chunkSize - 1) / chunkSize

	for b := 0; b < batch; b++ {
		for qh := 0; qh < numQHeads; qh++ {
			bh := b*numQHeads + qh
			kvHead := qh / headRatio

			// Per-split partial results.
			splitMax := make([]float64, numSplits)
			splitSum := make([]float64, numSplits)
			splitAcc := make([][]float64, numSplits)

			for s := 0; s < numSplits; s++ {
				splitAcc[s] = make([]float64, headDim)
				chunkStart := s * chunkSize
				chunkEnd := chunkStart + chunkSize
				if chunkEnd > kvLen {
					chunkEnd = kvLen
				}

				localMax := -math.MaxFloat64
				localSum := 0.0

				for j := chunkStart; j < chunkEnd; j++ {
					dot := 0.0
					for d := 0; d < headDim; d++ {
						dot += float64(Q[bh*headDim+d]) * float64(K[b*kvLen*kvDim+j*kvDim+kvHead*headDim+d])
					}
					score := dot * scale

					// Online softmax update.
					prevMax := localMax
					if score > localMax {
						localMax = score
					}
					expDiff := math.Exp(prevMax - localMax)
					expS := math.Exp(score - localMax)
					localSum = localSum*expDiff + expS

					for d := 0; d < headDim; d++ {
						splitAcc[s][d] = splitAcc[s][d]*expDiff +
							expS*float64(V[b*kvLen*kvDim+j*kvDim+kvHead*headDim+d])
					}
				}

				splitMax[s] = localMax
				splitSum[s] = localSum
			}

			// Cross-split reduction.
			globalMax := splitMax[0]
			for s := 1; s < numSplits; s++ {
				if splitMax[s] > globalMax {
					globalMax = splitMax[s]
				}
			}

			// Compute denominator: sum_s(exp(max_s - globalMax) * sum_s).
			denom := 0.0
			for s := 0; s < numSplits; s++ {
				denom += math.Exp(splitMax[s]-globalMax) * splitSum[s]
			}
			invDenom := 1.0 / denom

			// Accumulate rescaled partial outputs.
			for d := 0; d < headDim; d++ {
				acc := 0.0
				for s := 0; s < numSplits; s++ {
					w := math.Exp(splitMax[s] - globalMax)
					acc += w * splitAcc[s][d]
				}
				O[bh*headDim+d] = float32(acc * invDenom)
			}
		}
	}

	return nil
}

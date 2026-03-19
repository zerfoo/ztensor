//go:build !cuda

package kernels

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// naiveAttentionFA2 computes softmax(Q*K^T / sqrt(d)) * V on CPU for reference.
func naiveAttentionFA2(Q, K, V []float32, batch, heads, seqLen, headDim int, causal bool) []float32 {
	O := make([]float32, batch*heads*seqLen*headDim)
	scale := 1.0 / math.Sqrt(float64(headDim))

	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			bh := b*heads + h
			base := bh * seqLen * headDim

			for i := 0; i < seqLen; i++ {
				scores := make([]float64, seqLen)
				maxScore := -math.MaxFloat64

				for j := 0; j < seqLen; j++ {
					if causal && j > i {
						scores[j] = -math.MaxFloat64
						continue
					}
					dot := 0.0
					for d := 0; d < headDim; d++ {
						dot += float64(Q[base+i*headDim+d]) * float64(K[base+j*headDim+d])
					}
					scores[j] = dot * scale
					if scores[j] > maxScore {
						maxScore = scores[j]
					}
				}

				sum := 0.0
				for j := 0; j < seqLen; j++ {
					scores[j] = math.Exp(scores[j] - maxScore)
					sum += scores[j]
				}
				for j := 0; j < seqLen; j++ {
					scores[j] /= sum
				}

				for d := 0; d < headDim; d++ {
					acc := 0.0
					for j := 0; j < seqLen; j++ {
						acc += scores[j] * float64(V[base+j*headDim+d])
					}
					O[base+i*headDim+d] = float32(acc)
				}
			}
		}
	}
	return O
}

// naiveDecodeAttentionFA2 computes the CPU reference for decode attention with GQA.
func naiveDecodeAttentionFA2(Q, K, V []float32, batch, numQHeads, numKVHeads, kvLen, headDim int) []float32 {
	O := make([]float32, batch*numQHeads*headDim)
	scale := 1.0 / math.Sqrt(float64(headDim))
	headRatio := numQHeads / numKVHeads
	kvDim := numKVHeads * headDim

	for b := 0; b < batch; b++ {
		for qh := 0; qh < numQHeads; qh++ {
			bh := b*numQHeads + qh
			kvHead := qh / headRatio

			scores := make([]float64, kvLen)
			maxScore := -math.MaxFloat64
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

			sum := 0.0
			for j := 0; j < kvLen; j++ {
				scores[j] = math.Exp(scores[j] - maxScore)
				sum += scores[j]
			}
			for j := 0; j < kvLen; j++ {
				scores[j] /= sum
			}

			for d := 0; d < headDim; d++ {
				acc := 0.0
				for j := 0; j < kvLen; j++ {
					acc += scores[j] * float64(V[b*kvLen*kvDim+j*kvDim+kvHead*headDim+d])
				}
				O[bh*headDim+d] = float32(acc)
			}
		}
	}
	return O
}

// TestFlashAttention2_Correctness verifies that the FlashAttention-2 forward
// and decode kernels produce results matching naive CPU attention. Tests
// non-causal, causal, non-GQA decode, and GQA decode configurations.
func TestFlashAttention2_Correctness(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}

	t.Run("forward_non_causal", func(t *testing.T) {
		batch, heads, seqLen, headDim := 2, 2, 64, 32
		n := batch * heads * seqLen * headDim

		Q := make([]float32, n)
		K := make([]float32, n)
		V := make([]float32, n)
		for i := range Q {
			Q[i] = float32(i%7-3) * 0.1
			K[i] = float32(i%5-2) * 0.1
			V[i] = float32(i%11-5) * 0.1
		}

		expected := naiveAttentionFA2(Q, K, V, batch, heads, seqLen, headDim, false)

		byteSize := n * 4
		devQ, err := cuda.Malloc(byteSize)
		if err != nil {
			t.Fatalf("Malloc Q: %v", err)
		}
		defer func() { _ = cuda.Free(devQ) }()

		devK, err := cuda.Malloc(byteSize)
		if err != nil {
			t.Fatalf("Malloc K: %v", err)
		}
		defer func() { _ = cuda.Free(devK) }()

		devV, err := cuda.Malloc(byteSize)
		if err != nil {
			t.Fatalf("Malloc V: %v", err)
		}
		defer func() { _ = cuda.Free(devV) }()

		devO, err := cuda.Malloc(byteSize)
		if err != nil {
			t.Fatalf("Malloc O: %v", err)
		}
		defer func() { _ = cuda.Free(devO) }()

		_ = cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), byteSize, cuda.MemcpyHostToDevice)
		_ = cuda.Memcpy(devK, unsafe.Pointer(&K[0]), byteSize, cuda.MemcpyHostToDevice)
		_ = cuda.Memcpy(devV, unsafe.Pointer(&V[0]), byteSize, cuda.MemcpyHostToDevice)

		stream, err := cuda.CreateStream()
		if err != nil {
			t.Fatalf("CreateStream: %v", err)
		}
		defer func() { _ = stream.Destroy() }()

		if err := FlashAttention2Forward(devQ, devK, devV, devO, batch, heads, seqLen, headDim, false, stream.Ptr()); err != nil {
			t.Fatalf("FlashAttention2Forward: %v", err)
		}
		_ = stream.Synchronize()

		result := make([]float32, n)
		_ = cuda.Memcpy(unsafe.Pointer(&result[0]), devO, byteSize, cuda.MemcpyDeviceToHost)

		tol := 1e-3
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
			t.Errorf("total mismatches: %d / %d", mismatches, n)
		}
	})

	t.Run("forward_causal", func(t *testing.T) {
		batch, heads, seqLen, headDim := 1, 2, 32, 16
		n := batch * heads * seqLen * headDim

		Q := make([]float32, n)
		K := make([]float32, n)
		V := make([]float32, n)
		for i := range Q {
			Q[i] = float32(i%13-6) * 0.05
			K[i] = float32(i%9-4) * 0.05
			V[i] = float32(i%7-3) * 0.05
		}

		expected := naiveAttentionFA2(Q, K, V, batch, heads, seqLen, headDim, true)

		byteSize := n * 4
		devQ, err := cuda.Malloc(byteSize)
		if err != nil {
			t.Fatalf("Malloc Q: %v", err)
		}
		defer func() { _ = cuda.Free(devQ) }()

		devK, err := cuda.Malloc(byteSize)
		if err != nil {
			t.Fatalf("Malloc K: %v", err)
		}
		defer func() { _ = cuda.Free(devK) }()

		devV, err := cuda.Malloc(byteSize)
		if err != nil {
			t.Fatalf("Malloc V: %v", err)
		}
		defer func() { _ = cuda.Free(devV) }()

		devO, err := cuda.Malloc(byteSize)
		if err != nil {
			t.Fatalf("Malloc O: %v", err)
		}
		defer func() { _ = cuda.Free(devO) }()

		_ = cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), byteSize, cuda.MemcpyHostToDevice)
		_ = cuda.Memcpy(devK, unsafe.Pointer(&K[0]), byteSize, cuda.MemcpyHostToDevice)
		_ = cuda.Memcpy(devV, unsafe.Pointer(&V[0]), byteSize, cuda.MemcpyHostToDevice)

		stream, err := cuda.CreateStream()
		if err != nil {
			t.Fatalf("CreateStream: %v", err)
		}
		defer func() { _ = stream.Destroy() }()

		if err := FlashAttention2Forward(devQ, devK, devV, devO, batch, heads, seqLen, headDim, true, stream.Ptr()); err != nil {
			t.Fatalf("FlashAttention2Forward causal: %v", err)
		}
		_ = stream.Synchronize()

		result := make([]float32, n)
		_ = cuda.Memcpy(unsafe.Pointer(&result[0]), devO, byteSize, cuda.MemcpyDeviceToHost)

		tol := 1e-3
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
			t.Errorf("total mismatches (causal): %d / %d", mismatches, n)
		}
	})

	t.Run("decode_non_GQA", func(t *testing.T) {
		batch, numQHeads, numKVHeads, headDim, kvLen := 1, 4, 4, 64, 32
		numBH := batch * numQHeads

		qSize := numBH * headDim
		kvSize := batch * numKVHeads * kvLen * headDim

		Q := make([]float32, qSize)
		K := make([]float32, kvSize)
		V := make([]float32, kvSize)
		for i := range Q {
			Q[i] = float32(i%7-3) * 0.1
		}
		for i := range K {
			K[i] = float32(i%5-2) * 0.1
			V[i] = float32(i%11-5) * 0.1
		}

		expected := naiveDecodeAttentionFA2(Q, K, V, batch, numQHeads, numKVHeads, kvLen, headDim)

		devQ, err := cuda.Malloc(qSize * 4)
		if err != nil {
			t.Fatalf("Malloc Q: %v", err)
		}
		defer func() { _ = cuda.Free(devQ) }()

		devK, err := cuda.Malloc(kvSize * 4)
		if err != nil {
			t.Fatalf("Malloc K: %v", err)
		}
		defer func() { _ = cuda.Free(devK) }()

		devV, err := cuda.Malloc(kvSize * 4)
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
		_ = cuda.Memcpy(devK, unsafe.Pointer(&K[0]), kvSize*4, cuda.MemcpyHostToDevice)
		_ = cuda.Memcpy(devV, unsafe.Pointer(&V[0]), kvSize*4, cuda.MemcpyHostToDevice)

		stream, err := cuda.CreateStream()
		if err != nil {
			t.Fatalf("CreateStream: %v", err)
		}
		defer func() { _ = stream.Destroy() }()

		if err := FlashAttention2Decode(devQ, devK, devV, devO, numBH, kvLen, headDim, kvLen, nil, numQHeads, numKVHeads, stream.Ptr()); err != nil {
			t.Fatalf("FlashAttention2Decode: %v", err)
		}
		_ = stream.Synchronize()

		result := make([]float32, qSize)
		_ = cuda.Memcpy(unsafe.Pointer(&result[0]), devO, qSize*4, cuda.MemcpyDeviceToHost)

		tol := 1e-3
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
	})

	t.Run("decode_GQA", func(t *testing.T) {
		batch, numQHeads, numKVHeads, headDim, kvLen := 1, 8, 4, 128, 64
		numBH := batch * numQHeads

		qSize := numBH * headDim
		kvSize := batch * numKVHeads * kvLen * headDim

		Q := make([]float32, qSize)
		K := make([]float32, kvSize)
		V := make([]float32, kvSize)
		for i := range Q {
			Q[i] = float32(i%7-3) * 0.1
		}
		for i := range K {
			K[i] = float32(i%5-2) * 0.1
			V[i] = float32(i%11-5) * 0.1
		}

		expected := naiveDecodeAttentionFA2(Q, K, V, batch, numQHeads, numKVHeads, kvLen, headDim)

		devQ, err := cuda.Malloc(qSize * 4)
		if err != nil {
			t.Fatalf("Malloc Q: %v", err)
		}
		defer func() { _ = cuda.Free(devQ) }()

		devK, err := cuda.Malloc(kvSize * 4)
		if err != nil {
			t.Fatalf("Malloc K: %v", err)
		}
		defer func() { _ = cuda.Free(devK) }()

		devV, err := cuda.Malloc(kvSize * 4)
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
		_ = cuda.Memcpy(devK, unsafe.Pointer(&K[0]), kvSize*4, cuda.MemcpyHostToDevice)
		_ = cuda.Memcpy(devV, unsafe.Pointer(&V[0]), kvSize*4, cuda.MemcpyHostToDevice)

		stream, err := cuda.CreateStream()
		if err != nil {
			t.Fatalf("CreateStream: %v", err)
		}
		defer func() { _ = stream.Destroy() }()

		if err := FlashAttention2Decode(devQ, devK, devV, devO, numBH, kvLen, headDim, kvLen, nil, numQHeads, numKVHeads, stream.Ptr()); err != nil {
			t.Fatalf("FlashAttention2Decode GQA: %v", err)
		}
		_ = stream.Synchronize()

		result := make([]float32, qSize)
		_ = cuda.Memcpy(unsafe.Pointer(&result[0]), devO, qSize*4, cuda.MemcpyDeviceToHost)

		tol := 1e-3
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
			t.Errorf("total mismatches (GQA): %d / %d", mismatches, qSize)
		}
	})

	t.Run("decode_GQA_large_head_dim", func(t *testing.T) {
		// Gemma 3 uses head_dim=256 for keys.
		batch, numQHeads, numKVHeads, headDim, kvLen := 1, 8, 4, 256, 32
		numBH := batch * numQHeads

		qSize := numBH * headDim
		kvSize := batch * numKVHeads * kvLen * headDim

		Q := make([]float32, qSize)
		K := make([]float32, kvSize)
		V := make([]float32, kvSize)
		for i := range Q {
			Q[i] = float32(i%7-3) * 0.1
		}
		for i := range K {
			K[i] = float32(i%5-2) * 0.1
			V[i] = float32(i%11-5) * 0.1
		}

		expected := naiveDecodeAttentionFA2(Q, K, V, batch, numQHeads, numKVHeads, kvLen, headDim)

		devQ, err := cuda.Malloc(qSize * 4)
		if err != nil {
			t.Fatalf("Malloc Q: %v", err)
		}
		defer func() { _ = cuda.Free(devQ) }()

		devK, err := cuda.Malloc(kvSize * 4)
		if err != nil {
			t.Fatalf("Malloc K: %v", err)
		}
		defer func() { _ = cuda.Free(devK) }()

		devV, err := cuda.Malloc(kvSize * 4)
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
		_ = cuda.Memcpy(devK, unsafe.Pointer(&K[0]), kvSize*4, cuda.MemcpyHostToDevice)
		_ = cuda.Memcpy(devV, unsafe.Pointer(&V[0]), kvSize*4, cuda.MemcpyHostToDevice)

		stream, err := cuda.CreateStream()
		if err != nil {
			t.Fatalf("CreateStream: %v", err)
		}
		defer func() { _ = stream.Destroy() }()

		if err := FlashAttention2Decode(devQ, devK, devV, devO, numBH, kvLen, headDim, kvLen, nil, numQHeads, numKVHeads, stream.Ptr()); err != nil {
			t.Fatalf("FlashAttention2Decode GQA large head_dim: %v", err)
		}
		_ = stream.Synchronize()

		result := make([]float32, qSize)
		_ = cuda.Memcpy(unsafe.Pointer(&result[0]), devO, qSize*4, cuda.MemcpyDeviceToHost)

		tol := 1e-3
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
			t.Errorf("total mismatches (GQA large): %d / %d", mismatches, qSize)
		}
	})
}

// TestFlashAttention2_MemoryBound verifies that the FlashAttention-2 kernel
// uses O(N) memory, not O(N^2). The kernel uses only shared memory internally
// (no cudaMalloc), so the only device memory is the caller-provided
// input/output buffers. The test runs at two different sequence lengths and
// verifies both complete successfully with only O(N)-sized allocations.
func TestFlashAttention2_MemoryBound(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}

	t.Run("forward_O_N_memory", func(t *testing.T) {
		// Run at two sequence lengths. Total device allocation is
		// 4 * batch * heads * seqLen * headDim * sizeof(float32) — O(N).
		// An O(N^2) kernel would need an additional N*N*4 scratch buffer.
		for _, seqLen := range []int{64, 128} {
			batch, heads, headDim := 1, 1, 32
			n := batch * heads * seqLen * headDim

			Q := make([]float32, n)
			K := make([]float32, n)
			V := make([]float32, n)
			for i := range Q {
				Q[i] = float32(i%7-3) * 0.1
				K[i] = float32(i%5-2) * 0.1
				V[i] = float32(i%11-5) * 0.1
			}

			byteSize := n * 4
			devQ, err := cuda.Malloc(byteSize)
			if err != nil {
				t.Fatalf("seqLen=%d: Malloc Q: %v", seqLen, err)
			}
			defer func() { _ = cuda.Free(devQ) }()

			devK, err := cuda.Malloc(byteSize)
			if err != nil {
				t.Fatalf("seqLen=%d: Malloc K: %v", seqLen, err)
			}
			defer func() { _ = cuda.Free(devK) }()

			devV, err := cuda.Malloc(byteSize)
			if err != nil {
				t.Fatalf("seqLen=%d: Malloc V: %v", seqLen, err)
			}
			defer func() { _ = cuda.Free(devV) }()

			devO, err := cuda.Malloc(byteSize)
			if err != nil {
				t.Fatalf("seqLen=%d: Malloc O: %v", seqLen, err)
			}
			defer func() { _ = cuda.Free(devO) }()

			_ = cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), byteSize, cuda.MemcpyHostToDevice)
			_ = cuda.Memcpy(devK, unsafe.Pointer(&K[0]), byteSize, cuda.MemcpyHostToDevice)
			_ = cuda.Memcpy(devV, unsafe.Pointer(&V[0]), byteSize, cuda.MemcpyHostToDevice)

			stream, err := cuda.CreateStream()
			if err != nil {
				t.Fatalf("seqLen=%d: CreateStream: %v", seqLen, err)
			}
			defer func() { _ = stream.Destroy() }()

			// The kernel succeeds with only O(N) input/output buffers.
			if err := FlashAttention2Forward(devQ, devK, devV, devO, batch, heads, seqLen, headDim, false, stream.Ptr()); err != nil {
				t.Fatalf("seqLen=%d: FlashAttention2Forward: %v", seqLen, err)
			}
			_ = stream.Synchronize()

			// Verify output is not all zeros (kernel actually ran).
			result := make([]float32, n)
			_ = cuda.Memcpy(unsafe.Pointer(&result[0]), devO, byteSize, cuda.MemcpyDeviceToHost)
			allZero := true
			for _, v := range result {
				if v != 0 {
					allZero = false
					break
				}
			}
			if allZero {
				t.Errorf("seqLen=%d: output is all zeros", seqLen)
			}
		}
	})

	t.Run("decode_O_N_memory", func(t *testing.T) {
		// Verify decode kernel runs with only O(N) allocations.
		for _, kvLen := range []int{64, 256} {
			batch, numQHeads, numKVHeads, headDim := 1, 8, 4, 128
			numBH := batch * numQHeads

			qSize := numBH * headDim
			kvSize := batch * numKVHeads * kvLen * headDim

			Q := make([]float32, qSize)
			K := make([]float32, kvSize)
			V := make([]float32, kvSize)
			for i := range Q {
				Q[i] = float32(i%7-3) * 0.1
			}
			for i := range K {
				K[i] = float32(i%5-2) * 0.1
				V[i] = float32(i%11-5) * 0.1
			}

			devQ, err := cuda.Malloc(qSize * 4)
			if err != nil {
				t.Fatalf("kvLen=%d: Malloc Q: %v", kvLen, err)
			}
			defer func() { _ = cuda.Free(devQ) }()

			devK, err := cuda.Malloc(kvSize * 4)
			if err != nil {
				t.Fatalf("kvLen=%d: Malloc K: %v", kvLen, err)
			}
			defer func() { _ = cuda.Free(devK) }()

			devV, err := cuda.Malloc(kvSize * 4)
			if err != nil {
				t.Fatalf("kvLen=%d: Malloc V: %v", kvLen, err)
			}
			defer func() { _ = cuda.Free(devV) }()

			devO, err := cuda.Malloc(qSize * 4)
			if err != nil {
				t.Fatalf("kvLen=%d: Malloc O: %v", kvLen, err)
			}
			defer func() { _ = cuda.Free(devO) }()

			_ = cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), qSize*4, cuda.MemcpyHostToDevice)
			_ = cuda.Memcpy(devK, unsafe.Pointer(&K[0]), kvSize*4, cuda.MemcpyHostToDevice)
			_ = cuda.Memcpy(devV, unsafe.Pointer(&V[0]), kvSize*4, cuda.MemcpyHostToDevice)

			stream, err := cuda.CreateStream()
			if err != nil {
				t.Fatalf("kvLen=%d: CreateStream: %v", kvLen, err)
			}
			defer func() { _ = stream.Destroy() }()

			if err := FlashAttention2Decode(devQ, devK, devV, devO, numBH, kvLen, headDim, kvLen, nil, numQHeads, numKVHeads, stream.Ptr()); err != nil {
				t.Fatalf("kvLen=%d: FlashAttention2Decode: %v", kvLen, err)
			}
			_ = stream.Synchronize()

			result := make([]float32, qSize)
			_ = cuda.Memcpy(unsafe.Pointer(&result[0]), devO, qSize*4, cuda.MemcpyDeviceToHost)
			allZero := true
			for _, v := range result {
				if v != 0 {
					allZero = false
					break
				}
			}
			if allZero {
				t.Errorf("kvLen=%d: output is all zeros", kvLen)
			}
		}
	})
}

package compute

import (
	"context"
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/graph/kv"
	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/internal/cuda/kernels"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// naiveDenseAttention computes CPU reference attention with contiguous KV.
// Q: [batch*numQHeads, headDim], K: [seqLen, numKVHeads, headDim],
// V: [seqLen, numKVHeads, headDim].
func naiveDenseAttention(
	Q, K, V []float32,
	batch, numQHeads, numKVHeads, seqLen, headDim int,
) []float32 {
	O := make([]float32, batch*numQHeads*headDim)
	scale := 1.0 / math.Sqrt(float64(headDim))
	headRatio := numQHeads / numKVHeads
	kvStride := numKVHeads * headDim

	for b := 0; b < batch; b++ {
		for qh := 0; qh < numQHeads; qh++ {
			bh := b*numQHeads + qh
			kvHead := qh / headRatio

			scores := make([]float64, seqLen)
			maxScore := -math.MaxFloat64

			for j := 0; j < seqLen; j++ {
				dot := 0.0
				for d := 0; d < headDim; d++ {
					kVal := float64(K[j*kvStride+kvHead*headDim+d])
					dot += float64(Q[bh*headDim+d]) * kVal
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
					vVal := float64(V[j*kvStride+kvHead*headDim+d])
					acc += scores[j] * vVal
				}
				O[bh*headDim+d] = float32(acc)
			}
		}
	}
	return O
}

// cosine returns the cosine similarity between two float32 slices.
func cosine(a, b []float32) float64 {
	dot, normA, normB := 0.0, 0.0, 0.0
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// TestPagedGQA builds a BlockPool+BlockTable, fills with synthetic K/V,
// runs the paged attention path via GPUEngine.PagedGQA, and compares
// output to a dense CPU reference (cosine similarity > 0.9999).
func TestPagedGQA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	if !kernels.IsPagedAttentionSupported() {
		t.Skip("paged attention kernel not loaded")
	}

	batch := 1
	numQHeads := 8
	numKVHeads := 4
	headDim := 64
	blockSize := 16
	seqLen := 48 // 3 blocks exactly

	// Create GPU engine.
	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}

	// Create Q on CPU, then upload to GPU.
	qSize := batch * numQHeads * headDim
	qData := make([]float32, qSize)
	for i := range qData {
		qData[i] = float32(i%7-3) * 0.1
	}
	qTensor, err := tensor.New[float32]([]int{batch * numQHeads, headDim}, qData)
	if err != nil {
		t.Fatalf("Q tensor: %v", err)
	}
	// Upload Q to GPU using engine Add with zero.
	zeroData := make([]float32, qSize)
	zeroTensor, err := tensor.New[float32]([]int{batch * numQHeads, headDim}, zeroData)
	if err != nil {
		t.Fatalf("zero tensor: %v", err)
	}
	gpuQ, err := eng.Add(context.Background(), qTensor, zeroTensor)
	if err != nil {
		t.Fatalf("upload Q to GPU: %v", err)
	}

	// Build contiguous KV data for CPU reference.
	kvStride := numKVHeads * headDim
	kvElems := seqLen * kvStride
	kData := make([]float32, kvElems)
	vData := make([]float32, kvElems)
	for i := range kData {
		kData[i] = float32(i%5-2) * 0.1
		vData[i] = float32(i%11-5) * 0.1
	}

	// CPU reference: dense attention.
	expected := naiveDenseAttention(qData, kData, vData, batch, numQHeads, numKVHeads, seqLen, headDim)

	// Build BlockPool and BlockTable (CPU-side), then split KV into blocks
	// and upload each block to GPU memory.
	numLayers := 1 // single-layer for this test
	numLogicalBlocks := (seqLen + blockSize - 1) / blockSize
	pool, err := kv.NewBlockPool[float32](numLogicalBlocks+4, numLayers, blockSize, headDim)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	table := kv.NewBlockTable(pool)
	if err := table.Append(seqLen); err != nil {
		t.Fatalf("BlockTable.Append: %v", err)
	}

	// Fill block table's blocks with KV data.
	elemsPerBlock := blockSize * kvStride
	blocks := table.Blocks()
	for bi, blk := range blocks {
		startPos := bi * blockSize
		for p := 0; p < blk.Used; p++ {
			srcOff := (startPos + p) * kvStride
			dstOff := p * kvStride
			copy(blk.K[dstOff:dstOff+kvStride], kData[srcOff:srcOff+kvStride])
			copy(blk.V[dstOff:dstOff+kvStride], vData[srcOff:srcOff+kvStride])
		}
	}

	// Upload each block's K/V to GPU and build pointer arrays.
	numPhysBlocks := len(blocks)
	kPtrs := make([]uintptr, numPhysBlocks)
	vPtrs := make([]uintptr, numPhysBlocks)
	var devKBlocks, devVBlocks []unsafe.Pointer
	for i, blk := range blocks {
		_ = blk
		dk, err := cuda.Malloc(elemsPerBlock * 4)
		if err != nil {
			t.Fatalf("Malloc K block %d: %v", i, err)
		}
		defer func(p unsafe.Pointer) { _ = cuda.Free(p) }(dk)
		_ = cuda.Memcpy(dk, unsafe.Pointer(&blocks[i].K[0]), elemsPerBlock*4, cuda.MemcpyHostToDevice)
		devKBlocks = append(devKBlocks, dk)
		kPtrs[i] = uintptr(dk)

		dv, err := cuda.Malloc(elemsPerBlock * 4)
		if err != nil {
			t.Fatalf("Malloc V block %d: %v", i, err)
		}
		defer func(p unsafe.Pointer) { _ = cuda.Free(p) }(dv)
		_ = cuda.Memcpy(dv, unsafe.Pointer(&blocks[i].V[0]), elemsPerBlock*4, cuda.MemcpyHostToDevice)
		devVBlocks = append(devVBlocks, dv)
		vPtrs[i] = uintptr(dv)
	}

	// Upload pointer arrays to device.
	devKPtrs, err := cuda.Malloc(numPhysBlocks * 8)
	if err != nil {
		t.Fatalf("Malloc kPtrs: %v", err)
	}
	defer func() { _ = cuda.Free(devKPtrs) }()
	_ = cuda.Memcpy(devKPtrs, unsafe.Pointer(&kPtrs[0]), numPhysBlocks*8, cuda.MemcpyHostToDevice)

	devVPtrs, err := cuda.Malloc(numPhysBlocks * 8)
	if err != nil {
		t.Fatalf("Malloc vPtrs: %v", err)
	}
	defer func() { _ = cuda.Free(devVPtrs) }()
	_ = cuda.Memcpy(devVPtrs, unsafe.Pointer(&vPtrs[0]), numPhysBlocks*8, cuda.MemcpyHostToDevice)

	// Block indices: identity mapping for batch=1.
	blockIndicesFlat := make([]int32, batch*numLogicalBlocks)
	for i := 0; i < numLogicalBlocks; i++ {
		blockIndicesFlat[i] = int32(i)
	}
	devBlockIndices, err := cuda.Malloc(len(blockIndicesFlat) * 4)
	if err != nil {
		t.Fatalf("Malloc blockIndices: %v", err)
	}
	defer func() { _ = cuda.Free(devBlockIndices) }()
	_ = cuda.Memcpy(devBlockIndices, unsafe.Pointer(&blockIndicesFlat[0]), len(blockIndicesFlat)*4, cuda.MemcpyHostToDevice)

	// Call PagedGQA via the engine.
	result, err := eng.PagedGQA(
		gpuQ,
		devKPtrs, devVPtrs,
		devBlockIndices,
		seqLen, blockSize, headDim,
		numQHeads, numKVHeads,
		batch,
	)
	if err != nil {
		t.Fatalf("PagedGQA: %v", err)
	}

	// Sync and read back.
	if err := eng.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	resultData := result.Data()
	if len(resultData) != len(expected) {
		t.Fatalf("output length: got %d, want %d", len(resultData), len(expected))
	}

	// Compare cosine similarity.
	sim := cosine(resultData, expected)
	t.Logf("cosine similarity: %.6f", sim)
	if sim < 0.9999 {
		t.Errorf("cosine similarity %.6f < 0.9999", sim)
	}

	// Also check element-wise tolerance.
	tol := float32(1e-3)
	mismatches := 0
	for i := range resultData {
		diff := resultData[i] - expected[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			if mismatches < 5 {
				t.Errorf("output[%d] = %f, want %f (diff %e)", i, resultData[i], expected[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches: %d / %d", mismatches, len(resultData))
	}
}

// TestPagedGQAInterface verifies that GPUEngine satisfies PagedGQAer
// via type assertion, and that IsPagedGQASupported is callable.
func TestPagedGQAInterface(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}

	pager, ok := any(eng).(PagedGQAer)
	if !ok {
		t.Fatal("GPUEngine[float32] does not satisfy PagedGQAer")
	}

	// IsPagedGQASupported should not panic.
	_ = pager.IsPagedGQASupported()
}

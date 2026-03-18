//go:build !cuda

package kernels

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// naivePagedAttention computes CPU reference for paged attention.
// Q: [batch*numQHeads, headDim].
// kBlocks/vBlocks: slices of block data, each [blockSize, numKVHeads, headDim].
// blockIndices: [batch][numLogicalBlocks] maps logical block to physical block.
// seqLen: actual number of valid KV positions.
func naivePagedAttention(
	Q []float32,
	kBlocks, vBlocks [][]float32,
	blockIndices [][]int,
	batch, numQHeads, numKVHeads, seqLen, blockSize, headDim int,
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
				lb := j / blockSize
				posInBlock := j % blockSize
				physBlock := blockIndices[b][lb]

				dot := 0.0
				for d := 0; d < headDim; d++ {
					kVal := float64(kBlocks[physBlock][posInBlock*kvStride+kvHead*headDim+d])
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
					lb := j / blockSize
					posInBlock := j % blockSize
					physBlock := blockIndices[b][lb]
					vVal := float64(vBlocks[physBlock][posInBlock*kvStride+kvHead*headDim+d])
					acc += scores[j] * vVal
				}
				O[bh*headDim+d] = float32(acc)
			}
		}
	}
	return O
}

// TestPagedAttentionVsContiguous creates a contiguous KV cache, splits it into
// paged blocks, runs paged attention, and compares against naive CPU reference.
func TestPagedAttentionVsContiguous(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}
	if !IsPagedAttentionSupported() {
		t.Skip("paged attention kernel not loaded")
	}

	batch := 1
	numQHeads := 4
	numKVHeads := 4
	headDim := 64
	blockSize := 16
	seqLen := 48 // 3 blocks exactly

	numLogicalBlocks := (seqLen + blockSize - 1) / blockSize
	kvStride := numKVHeads * headDim
	elemsPerBlock := blockSize * kvStride

	// Create Q.
	qSize := batch * numQHeads * headDim
	Q := make([]float32, qSize)
	for i := range Q {
		Q[i] = float32(i%7-3) * 0.1
	}

	// Create KV blocks on host.
	numPhysBlocks := numLogicalBlocks // 1:1 mapping for simplicity
	kBlocks := make([][]float32, numPhysBlocks)
	vBlocks := make([][]float32, numPhysBlocks)
	for b := 0; b < numPhysBlocks; b++ {
		kBlocks[b] = make([]float32, elemsPerBlock)
		vBlocks[b] = make([]float32, elemsPerBlock)
		for i := range kBlocks[b] {
			kBlocks[b][i] = float32((b*elemsPerBlock+i)%5-2) * 0.1
			vBlocks[b][i] = float32((b*elemsPerBlock+i)%11-5) * 0.1
		}
	}

	// Block indices: identity mapping, one batch.
	blockIndicesHost := make([][]int, batch)
	blockIndicesHost[0] = make([]int, numLogicalBlocks)
	for i := 0; i < numLogicalBlocks; i++ {
		blockIndicesHost[0][i] = i
	}

	// CPU reference.
	expected := naivePagedAttention(Q, kBlocks, vBlocks, blockIndicesHost, batch, numQHeads, numKVHeads, seqLen, blockSize, headDim)

	// Allocate device memory for Q and O.
	devQ, err := cuda.Malloc(qSize * 4)
	if err != nil {
		t.Fatalf("Malloc Q: %v", err)
	}
	defer func() { _ = cuda.Free(devQ) }()

	devO, err := cuda.Malloc(qSize * 4)
	if err != nil {
		t.Fatalf("Malloc O: %v", err)
	}
	defer func() { _ = cuda.Free(devO) }()

	_ = cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), qSize*4, cuda.MemcpyHostToDevice)

	// Allocate device memory for each KV block and build pointer arrays.
	devKBlocks := make([]unsafe.Pointer, numPhysBlocks)
	devVBlocks := make([]unsafe.Pointer, numPhysBlocks)
	kPtrs := make([]uintptr, numPhysBlocks)
	vPtrs := make([]uintptr, numPhysBlocks)
	for b := 0; b < numPhysBlocks; b++ {
		dk, err := cuda.Malloc(elemsPerBlock * 4)
		if err != nil {
			t.Fatalf("Malloc K block %d: %v", b, err)
		}
		defer func(p unsafe.Pointer) { _ = cuda.Free(p) }(dk)
		devKBlocks[b] = dk
		_ = cuda.Memcpy(dk, unsafe.Pointer(&kBlocks[b][0]), elemsPerBlock*4, cuda.MemcpyHostToDevice)
		kPtrs[b] = uintptr(dk)

		dv, err := cuda.Malloc(elemsPerBlock * 4)
		if err != nil {
			t.Fatalf("Malloc V block %d: %v", b, err)
		}
		defer func(p unsafe.Pointer) { _ = cuda.Free(p) }(dv)
		devVBlocks[b] = dv
		_ = cuda.Memcpy(dv, unsafe.Pointer(&vBlocks[b][0]), elemsPerBlock*4, cuda.MemcpyHostToDevice)
		vPtrs[b] = uintptr(dv)
	}

	// Upload block pointer arrays to device.
	devKPtrs, err := cuda.Malloc(numPhysBlocks * 8) // 8 bytes per pointer
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

	// Upload block indices (int32 on device).
	blockIndicesFlat := make([]int32, batch*numLogicalBlocks)
	for b := 0; b < batch; b++ {
		for i := 0; i < numLogicalBlocks; i++ {
			blockIndicesFlat[b*numLogicalBlocks+i] = int32(blockIndicesHost[b][i])
		}
	}
	devBlockIndices, err := cuda.Malloc(len(blockIndicesFlat) * 4)
	if err != nil {
		t.Fatalf("Malloc blockIndices: %v", err)
	}
	defer func() { _ = cuda.Free(devBlockIndices) }()
	_ = cuda.Memcpy(devBlockIndices, unsafe.Pointer(&blockIndicesFlat[0]), len(blockIndicesFlat)*4, cuda.MemcpyHostToDevice)

	// Create stream and run kernel.
	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	if err := PagedAttentionForward(
		devQ, devO,
		devKPtrs, devVPtrs,
		devBlockIndices,
		seqLen, blockSize, headDim,
		numQHeads, numKVHeads,
		batch,
		stream.Ptr(),
	); err != nil {
		t.Fatalf("PagedAttentionForward: %v", err)
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
}

// TestPagedAttentionGQA verifies paged attention with grouped-query attention
// (numQHeads > numKVHeads).
func TestPagedAttentionGQA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}
	if !IsPagedAttentionSupported() {
		t.Skip("paged attention kernel not loaded")
	}

	batch := 1
	numQHeads := 8
	numKVHeads := 4
	headDim := 64
	blockSize := 16
	seqLen := 32 // 2 blocks

	numLogicalBlocks := (seqLen + blockSize - 1) / blockSize
	kvStride := numKVHeads * headDim
	elemsPerBlock := blockSize * kvStride

	qSize := batch * numQHeads * headDim
	Q := make([]float32, qSize)
	for i := range Q {
		Q[i] = float32(i%7-3) * 0.1
	}

	numPhysBlocks := numLogicalBlocks
	kBlocks := make([][]float32, numPhysBlocks)
	vBlocks := make([][]float32, numPhysBlocks)
	for b := 0; b < numPhysBlocks; b++ {
		kBlocks[b] = make([]float32, elemsPerBlock)
		vBlocks[b] = make([]float32, elemsPerBlock)
		for i := range kBlocks[b] {
			kBlocks[b][i] = float32((b*elemsPerBlock+i)%5-2) * 0.1
			vBlocks[b][i] = float32((b*elemsPerBlock+i)%11-5) * 0.1
		}
	}

	blockIndicesHost := make([][]int, batch)
	blockIndicesHost[0] = make([]int, numLogicalBlocks)
	for i := 0; i < numLogicalBlocks; i++ {
		blockIndicesHost[0][i] = i
	}

	expected := naivePagedAttention(Q, kBlocks, vBlocks, blockIndicesHost, batch, numQHeads, numKVHeads, seqLen, blockSize, headDim)

	devQ, err := cuda.Malloc(qSize * 4)
	if err != nil {
		t.Fatalf("Malloc Q: %v", err)
	}
	defer func() { _ = cuda.Free(devQ) }()

	devO, err := cuda.Malloc(qSize * 4)
	if err != nil {
		t.Fatalf("Malloc O: %v", err)
	}
	defer func() { _ = cuda.Free(devO) }()

	_ = cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), qSize*4, cuda.MemcpyHostToDevice)

	devKBlocks := make([]unsafe.Pointer, numPhysBlocks)
	devVBlocks := make([]unsafe.Pointer, numPhysBlocks)
	kPtrs := make([]uintptr, numPhysBlocks)
	vPtrs := make([]uintptr, numPhysBlocks)
	for b := 0; b < numPhysBlocks; b++ {
		dk, err := cuda.Malloc(elemsPerBlock * 4)
		if err != nil {
			t.Fatalf("Malloc K block %d: %v", b, err)
		}
		defer func(p unsafe.Pointer) { _ = cuda.Free(p) }(dk)
		devKBlocks[b] = dk
		_ = cuda.Memcpy(dk, unsafe.Pointer(&kBlocks[b][0]), elemsPerBlock*4, cuda.MemcpyHostToDevice)
		kPtrs[b] = uintptr(dk)

		dv, err := cuda.Malloc(elemsPerBlock * 4)
		if err != nil {
			t.Fatalf("Malloc V block %d: %v", b, err)
		}
		defer func(p unsafe.Pointer) { _ = cuda.Free(p) }(dv)
		devVBlocks[b] = dv
		_ = cuda.Memcpy(dv, unsafe.Pointer(&vBlocks[b][0]), elemsPerBlock*4, cuda.MemcpyHostToDevice)
		vPtrs[b] = uintptr(dv)
	}

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

	blockIndicesFlat := make([]int32, batch*numLogicalBlocks)
	for b := 0; b < batch; b++ {
		for i := 0; i < numLogicalBlocks; i++ {
			blockIndicesFlat[b*numLogicalBlocks+i] = int32(blockIndicesHost[b][i])
		}
	}
	devBlockIndices, err := cuda.Malloc(len(blockIndicesFlat) * 4)
	if err != nil {
		t.Fatalf("Malloc blockIndices: %v", err)
	}
	defer func() { _ = cuda.Free(devBlockIndices) }()
	_ = cuda.Memcpy(devBlockIndices, unsafe.Pointer(&blockIndicesFlat[0]), len(blockIndicesFlat)*4, cuda.MemcpyHostToDevice)

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	if err := PagedAttentionForward(
		devQ, devO,
		devKPtrs, devVPtrs,
		devBlockIndices,
		seqLen, blockSize, headDim,
		numQHeads, numKVHeads,
		batch,
		stream.Ptr(),
	); err != nil {
		t.Fatalf("PagedAttentionForward GQA: %v", err)
	}
	_ = stream.Synchronize()

	result := make([]float32, qSize)
	_ = cuda.Memcpy(unsafe.Pointer(&result[0]), devO, qSize*4, cuda.MemcpyDeviceToHost)

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
		t.Errorf("total mismatches: %d / %d (GQA)", mismatches, qSize)
	}
}

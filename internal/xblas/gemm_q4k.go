package xblas

import (
	"runtime"
	"sync"
	"unsafe"

	"github.com/zerfoo/ztensor/tensor"
)

// GemmQ4KF32 computes C = A * B where A is Q4_K quantized and B, C are float32.
// A has logical shape (m, k), B has shape (k, n), C has shape (m, n).
// K must be a multiple of 256 (Q4_K super-block size).
// Uses fused dequant+multiply to avoid heap-allocating the full dequantized A.
func GemmQ4KF32(m, n, k int, a *tensor.Q4KStorage, b, c []float32) {
	if k%256 != 0 {
		// Fallback: dequant then dense SGEMM.
		af32 := make([]float32, a.Len())
		a.Dequantize(af32)
		SgemmSimd(m, n, k, af32, b, c)
		return
	}

	superBlocksPerRow := k / 256

	// Zero output.
	for i := range c {
		c[i] = 0
	}

	// Stack buffer for one sub-block (32 values).
	var buf [32]float32

	for i := range m {
		cRow := c[i*n : (i+1)*n]
		for sbi := range superBlocksPerRow {
			blkIdx := i*superBlocksPerRow + sbi
			// Each Q4_K super-block contains 8 sub-blocks of 32 values each.
			for sub := range 8 {
				kBase := sbi*256 + sub*32
				a.DequantizeSubBlock(blkIdx, sub, buf[:])
				for p := range 32 {
					if aVal := buf[p]; aVal != 0 {
						sgemmAccRow(unsafe.Pointer(&cRow[0]), unsafe.Pointer(&b[(kBase+p)*n]), aVal, n)
					}
				}
			}
		}
	}
}

// GemmF32Q4KNT computes C = A * B^T where A is float32 [M,K] and B is Q4_K [N,K].
// B is stored in row-major Q4_K format. K must be a multiple of 256.
func GemmF32Q4KNT(m, n, k int, a []float32, b *tensor.Q4KStorage, c []float32) {
	if k%256 != 0 {
		// Fallback: dequant, transpose, regular SGEMM.
		bF32 := make([]float32, n*k)
		b.Dequantize(bF32)
		bT := make([]float32, k*n)
		for r := range n {
			for col := range k {
				bT[col*n+r] = bF32[r*k+col]
			}
		}
		SgemmSimd(m, n, k, a, bT, c)
		return
	}

	superBlocksPerRow := k / 256

	// M=1 GEMV: parallelize across N (rows of B) when beneficial.
	if m == 1 && n*k >= q4GemvParallelThreshold {
		nCores := runtime.NumCPU()
		nCores = min(nCores, n/4)
		if nCores > 1 {
			gemmF32Q4KNTParallel(n, a, b, c, superBlocksPerRow, nCores)
			return
		}
	}

	// Stack buffer for one sub-block.
	var buf [32]float32

	for i := range m {
		aRow := a[i*k : (i+1)*k]
		cRow := c[i*n : (i+1)*n]
		for j := range n {
			var sum float32
			for sbi := range superBlocksPerRow {
				blkIdx := j*superBlocksPerRow + sbi
				for sub := range 8 {
					kBase := sbi*256 + sub*32
					b.DequantizeSubBlock(blkIdx, sub, buf[:])
					for p := range 32 {
						sum += aRow[kBase+p] * buf[p]
					}
				}
			}
			cRow[j] = sum
		}
	}
}

// gemmF32Q4KNTParallel parallelizes the M=1 GEMV across N rows of B.
func gemmF32Q4KNTParallel(n int, a []float32, b *tensor.Q4KStorage, c []float32, superBlocksPerRow, nCores int) {
	var wg sync.WaitGroup
	chunkSize := (n + nCores - 1) / nCores

	for core := range nCores {
		jStart := core * chunkSize
		jEnd := jStart + chunkSize
		if jEnd > n {
			jEnd = n
		}
		if jStart >= jEnd {
			break
		}
		wg.Add(1)
		go func(jS, jE int) {
			defer wg.Done()
			var buf [32]float32
			for j := jS; j < jE; j++ {
				var sum float32
				for sbi := range superBlocksPerRow {
					blkIdx := j*superBlocksPerRow + sbi
					for sub := range 8 {
						kBase := sbi*256 + sub*32
						b.DequantizeSubBlock(blkIdx, sub, buf[:])
						for p := range 32 {
							sum += a[kBase+p] * buf[p]
						}
					}
				}
				c[j] = sum
			}
		}(jStart, jEnd)
	}
	wg.Wait()
}

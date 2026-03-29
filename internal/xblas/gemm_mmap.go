package xblas

import (
	"runtime"
	"sync"

	"github.com/zerfoo/ztensor/tensor"
)

// GemmF32MmapNT computes C = A * B^T where A is float32 [M,K] and B is
// MmapStorage [N,K]. Streams B per-superblock without ever materializing the
// full dequantized weight matrix. This is what allows inference on models
// larger than available RAM: only one 256-element superblock (≤210 bytes) is
// decoded to float32 at a time, instead of allocating the full N*K*4 bytes.
//
// Supported qtypes for streaming: Q4_K, Q5_K, Q6_K.
// All other qtypes fall back to b.Slice() (full decode, cached by sync.Once).
func GemmF32MmapNT(m, n, k int, a []float32, b *tensor.MmapStorage, c []float32) {
	switch b.QType() {
	case tensor.GGMLTypeQ4_K:
		gemmF32MmapQ4KNT(m, n, k, a, b, c)
	case tensor.GGMLTypeQ5_K:
		gemmF32MmapQ5KNT(m, n, k, a, b, c)
	case tensor.GGMLTypeQ6_K:
		gemmF32MmapQ6KNT(m, n, k, a, b, c)
	default:
		// Q4_0, Q8_0, F16, BF16, F32: these are typically small tensors (norms,
		// biases, embeddings). Fall back to full dequantize; sync.Once caches it.
		bF32 := b.Slice()
		for i := range m {
			for j := range n {
				var sum float32
				for p := range k {
					sum += a[i*k+p] * bF32[j*k+p]
				}
				c[i*n+j] = sum
			}
		}
	}
}

// GemmMmapF32 computes C = A * B where A is MmapStorage [M,K] and B is float32 [K,N].
// Streams A per-superblock to avoid materializing the full dequantized tensor.
func GemmMmapF32(m, n, k int, a *tensor.MmapStorage, b, c []float32) {
	switch a.QType() {
	case tensor.GGMLTypeQ4_K:
		gemmMmapQ4KF32(m, n, k, a, b, c)
	default:
		aF32 := a.Slice()
		SgemmSimd(m, n, k, aF32, b, c)
	}
}

// gemmF32MmapQ4KNT streams Q4_K superblocks from mmap'd B [N,K] per row.
// Uses a 256-float32 stack buffer; peak extra alloc = 1 KB per goroutine.
func gemmF32MmapQ4KNT(m, n, k int, a []float32, b *tensor.MmapStorage, c []float32) {
	if k%256 != 0 {
		bF32 := b.Slice()
		for i := range m {
			for j := range n {
				var sum float32
				for p := range k {
					sum += a[i*k+p] * bF32[j*k+p]
				}
				c[i*n+j] = sum
			}
		}
		return
	}
	blocksPerRow := k / 256
	if m == 1 && n*k >= mmapGemvParallelThreshold {
		nCores := min(runtime.NumCPU(), n/4)
		if nCores > 1 {
			gemmF32MmapQ4KNTParallel(n, a, b, c, blocksPerRow, nCores)
			return
		}
	}
	var buf [256]float32
	for i := range m {
		aRow := a[i*k:]
		for j := range n {
			var sum float32
			for bi := range blocksPerRow {
				blkIdx := j*blocksPerRow + bi
				tensor.DequantizeQ4K(b.Q4KBlockRaw(blkIdx), buf[:])
				kBase := bi * 256
				for p := range 256 {
					sum += aRow[kBase+p] * buf[p]
				}
			}
			c[i*n+j] = sum
		}
	}
}

const mmapGemvParallelThreshold = 256 * 256

func gemmF32MmapQ4KNTParallel(n int, a []float32, b *tensor.MmapStorage, c []float32, blocksPerRow, nCores int) {
	var wg sync.WaitGroup
	chunkSize := (n + nCores - 1) / nCores
	for t := range nCores {
		jStart := t * chunkSize
		jEnd := min(jStart+chunkSize, n)
		if jStart >= n {
			break
		}
		wg.Add(1)
		go func(jS, jE int) {
			defer wg.Done()
			var buf [256]float32
			for j := jS; j < jE; j++ {
				var sum float32
				for bi := range blocksPerRow {
					blkIdx := j*blocksPerRow + bi
					tensor.DequantizeQ4K(b.Q4KBlockRaw(blkIdx), buf[:])
					kBase := bi * 256
					for p := range 256 {
						sum += a[kBase+p] * buf[p]
					}
				}
				c[j] = sum
			}
		}(jStart, jEnd)
	}
	wg.Wait()
}

// gemmF32MmapQ5KNT streams Q5_K superblocks from mmap'd B [N,K].
func gemmF32MmapQ5KNT(m, n, k int, a []float32, b *tensor.MmapStorage, c []float32) {
	if k%256 != 0 {
		bF32 := b.Slice()
		for i := range m {
			for j := range n {
				var sum float32
				for p := range k {
					sum += a[i*k+p] * bF32[j*k+p]
				}
				c[i*n+j] = sum
			}
		}
		return
	}
	blocksPerRow := k / 256
	var buf [256]float32
	for i := range m {
		aRow := a[i*k:]
		for j := range n {
			var sum float32
			for bi := range blocksPerRow {
				blkIdx := j*blocksPerRow + bi
				tensor.DequantizeQ5K(b.Q5KBlockRaw(blkIdx), buf[:])
				kBase := bi * 256
				for p := range 256 {
					sum += aRow[kBase+p] * buf[p]
				}
			}
			c[i*n+j] = sum
		}
	}
}

// gemmF32MmapQ6KNT streams Q6_K superblocks from mmap'd B [N,K].
func gemmF32MmapQ6KNT(m, n, k int, a []float32, b *tensor.MmapStorage, c []float32) {
	if k%256 != 0 {
		bF32 := b.Slice()
		for i := range m {
			for j := range n {
				var sum float32
				for p := range k {
					sum += a[i*k+p] * bF32[j*k+p]
				}
				c[i*n+j] = sum
			}
		}
		return
	}
	blocksPerRow := k / 256
	var buf [256]float32
	for i := range m {
		aRow := a[i*k:]
		for j := range n {
			var sum float32
			for bi := range blocksPerRow {
				blkIdx := j*blocksPerRow + bi
				tensor.DequantizeQ6K(b.Q6KBlockRaw(blkIdx), buf[:])
				kBase := bi * 256
				for p := range 256 {
					sum += aRow[kBase+p] * buf[p]
				}
			}
			c[i*n+j] = sum
		}
	}
}

// gemmMmapQ4KF32 streams Q4_K rows of mmap'd A [M,K], multiplies by B [K,N].
func gemmMmapQ4KF32(m, n, k int, a *tensor.MmapStorage, b, c []float32) {
	if k%256 != 0 {
		aF32 := a.Slice()
		SgemmSimd(m, n, k, aF32, b, c)
		return
	}
	for i := range c {
		c[i] = 0
	}
	blocksPerRow := k / 256
	var buf [256]float32
	for i := range m {
		cRow := c[i*n : (i+1)*n]
		for bi := range blocksPerRow {
			blkIdx := i*blocksPerRow + bi
			tensor.DequantizeQ4K(a.Q4KBlockRaw(blkIdx), buf[:])
			kBase := bi * 256
			for p := range 256 {
				aVal := buf[p]
				if aVal == 0 {
					continue
				}
				bRow := b[(kBase+p)*n:]
				for j := range n {
					cRow[j] += aVal * bRow[j]
				}
			}
		}
	}
}

//go:build !arm64 && !amd64

package xblas

import "unsafe"

// SgemmSimd falls back to a naive triple-loop SGEMM on unsupported architectures.
func SgemmSimd(m, n, k int, a, b, c []float32) {
	lda, ldb, ldc := k, n, n
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += a[i*lda+p] * b[p*ldb+j]
			}
			c[i*ldc+j] = sum
		}
	}
}

// sgemmAccRow computes c[j] += aVal * b[j] for j=0..n-1 (scalar fallback).
func sgemmAccRow(cPtr, bPtr unsafe.Pointer, aVal float32, n int) {
	c := unsafe.Slice((*float32)(cPtr), n)
	b := unsafe.Slice((*float32)(bPtr), n)
	for j := range n {
		c[j] += aVal * b[j]
	}
}

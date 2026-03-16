//go:build !arm64 && !amd64

package xblas

import (
	"unsafe"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// SgemmSimd falls back to gonum BLAS on unsupported architectures.
func SgemmSimd(m, n, k int, a, b, c []float32) {
	alpha, beta := float32(1), float32(0)
	A := blas32.General{Rows: m, Cols: k, Data: a, Stride: k}
	B := blas32.General{Rows: k, Cols: n, Data: b, Stride: n}
	C := blas32.General{Rows: m, Cols: n, Data: c, Stride: n}
	blas32.Gemm(blas.NoTrans, blas.NoTrans, alpha, A, B, beta, C)
}

// sgemmAccRow computes c[j] += aVal * b[j] for j=0..n-1 (scalar fallback).
func sgemmAccRow(cPtr, bPtr unsafe.Pointer, aVal float32, n int) {
	c := unsafe.Slice((*float32)(cPtr), n)
	b := unsafe.Slice((*float32)(bPtr), n)
	for j := range n {
		c[j] += aVal * b[j]
	}
}

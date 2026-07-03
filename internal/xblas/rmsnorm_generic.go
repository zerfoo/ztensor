//go:build !arm64

package xblas

import (
	"math"
	"unsafe"
)

func RMSNormF32(out, x, weight *float32, dim int, eps float32, scale *float32) {
	xSlice := unsafe.Slice(x, dim)
	wSlice := unsafe.Slice(weight, dim)
	oSlice := unsafe.Slice(out, dim)
	// Fixed-order pairwise sum of squares in float32 -- matches the fp32
	// accumulation of the arm64 NEON path (rmsnorm_arm64.s) while shrinking the
	// error of the naive scalar fold this fallback previously used.
	sumSq := pairwiseSumF32(dim, func(i int) float32 { return xSlice[i] * xSlice[i] })
	s := float32(1.0 / math.Sqrt(float64(sumSq/float32(dim)+eps)))
	for i := range dim {
		oSlice[i] = xSlice[i] * s * wSlice[i]
	}
	*scale = s
}

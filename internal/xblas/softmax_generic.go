package xblas

import (
	"math"
	"unsafe"
)

// SoftmaxF32 computes softmax(x) in-place for a float32 vector of length n.
//
// Numerically stable: subtracts the row max before exp and accumulates the
// denominator in float64. Because the max element contributes exp(0)=1, the
// sum is always >= 1, so the final division never hits 0/0.
//
// This is the single correct implementation for every platform. The previous
// arm64 NEON kernel (internal/xblas/softmax_arm64.s) returned NaN for finite
// large-spread inputs — e.g. an attention-score row whose max is ~120 with
// several entries far below it — because its poly-exp/ldexp vector path
// mishandled deeply-negative (x-max) terms and corrupted the running sum. That
// kernel is retired in favor of correctness; a verified SIMD version can be
// reintroduced later behind the same signature. See softmax_test.go for the
// regression rows.
func SoftmaxF32(data *float32, n int) {
	if n <= 0 {
		return
	}
	s := unsafe.Slice(data, n)

	mx := s[0]
	for i := 1; i < n; i++ {
		if s[i] > mx {
			mx = s[i]
		}
	}

	var sum float64
	for i := range s {
		e := math.Exp(float64(s[i]) - float64(mx))
		s[i] = float32(e)
		sum += e
	}

	inv := 1.0 / sum
	for i := range s {
		s[i] = float32(float64(s[i]) * inv)
	}
}

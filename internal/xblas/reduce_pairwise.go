package xblas

// pairwiseBlock is the serial base-case size for fixed-order pairwise
// reduction, matching numpy's pairwise blocksize.
const pairwiseBlock = 128

// pairwiseSumF32 sums n float32 values in a fixed, chunk-independent order using
// a recursive tree (pairwise) reduction. Element i is read via get(i).
//
// The accumulation dtype stays float32 (matching the NEON RMSNorm asm path);
// only the order changes. Because the tree shape is a pure function of n, the
// result is bitwise-stable run to run, and its rounding error grows as
// O(log n) rather than O(n) for a naive scalar fold. See the arm64 asm in
// rmsnorm_arm64.s for the SIMD counterpart used on aarch64.
func pairwiseSumF32(n int, get func(i int) float32) float32 {
	if n <= 0 {
		return 0
	}

	return pairwiseSumRangeF32(0, n, get)
}

func pairwiseSumRangeF32(lo, hi int, get func(i int) float32) float32 {
	n := hi - lo
	if n <= pairwiseBlock {
		acc := get(lo)
		for i := lo + 1; i < hi; i++ {
			acc += get(i)
		}

		return acc
	}

	half := n / 2
	half -= half % pairwiseBlock
	if half == 0 {
		half = pairwiseBlock
	}

	return pairwiseSumRangeF32(lo, lo+half, get) + pairwiseSumRangeF32(lo+half, hi, get)
}

package numeric

// pairwiseBlock is the serial base-case size for fixed-order pairwise
// reduction, matching numpy's pairwise blocksize.
const pairwiseBlock = 128

// pairwiseSum sums s in a fixed, chunk-independent order using a recursive tree
// (pairwise) reduction. The accumulation dtype is the element type; only the
// order changes relative to a naive left-to-right fold, giving run-to-run
// bitwise stability and O(log n) rather than O(n) rounding-error growth.
//
// It is instantiated for the float element types whose Sum feeds reduction
// statistics; integer Sums are exact under any order and keep their simple fold.
func pairwiseSum[T ~float32 | ~float64](s []T) T {
	if len(s) == 0 {
		return 0
	}

	return pairwiseSumRange(s, 0, len(s))
}

func pairwiseSumRange[T ~float32 | ~float64](s []T, lo, hi int) T {
	n := hi - lo
	if n <= pairwiseBlock {
		acc := s[lo]
		for i := lo + 1; i < hi; i++ {
			acc += s[i]
		}

		return acc
	}

	half := n / 2
	half -= half % pairwiseBlock
	if half == 0 {
		half = pairwiseBlock
	}

	return pairwiseSumRange(s, lo, lo+half) + pairwiseSumRange(s, lo+half, hi)
}

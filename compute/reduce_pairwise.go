package compute

// pairwiseBlock is the serial base-case size for fixed-order pairwise
// reduction. It matches numpy's pairwise blocksize: large enough to amortize
// the recursion, small enough to preserve the O(log n) error bound.
const pairwiseBlock = 128

// pairwiseReduce sums n elements in a fixed, worker-independent order using a
// recursive tree (pairwise) reduction. Element i is read via get(i) and
// combined with add; zero is returned only for n <= 0.
//
// The split points are a pure function of n (recursive halving aligned to
// pairwiseBlock), so the result is bitwise-identical regardless of how the
// surrounding loop is parallelized or chunked. Compared with a naive
// left-to-right fold, the accumulated rounding error grows as O(log n) instead
// of O(n), which tightens agreement with a higher-precision oracle without
// changing the accumulation dtype. This is the canonical fixed-order
// accumulation for CPU reductions; see docs/design.md "Reduction accumulation".
func pairwiseReduce[T any](n int, zero T, get func(i int) T, add func(a, b T) T) T {
	if n <= 0 {
		return zero
	}
	return pairwiseRange(0, n, get, add)
}

// pairwiseRange reduces the half-open index range [lo, hi) using the fixed-order
// pairwise tree. The caller guarantees hi > lo.
func pairwiseRange[T any](lo, hi int, get func(i int) T, add func(a, b T) T) T {
	n := hi - lo
	if n <= pairwiseBlock {
		acc := get(lo)
		for i := lo + 1; i < hi; i++ {
			acc = add(acc, get(i))
		}

		return acc
	}

	// Split at half, snapped down to a pairwiseBlock boundary so the tree shape
	// depends only on n, never on runtime chunking.
	half := n / 2
	half -= half % pairwiseBlock
	if half == 0 {
		half = pairwiseBlock
	}

	return add(pairwiseRange(lo, lo+half, get, add), pairwiseRange(lo+half, hi, get, add))
}

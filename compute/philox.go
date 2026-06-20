package compute

// Philox4x32-10 counter-based RNG.
//
// Philox is a stateless, counter-based pseudo-random number generator: the
// output is a pure function of a (key, counter) pair, with no carried state.
// That property is exactly what dropout needs for CPU-GPU parity -- the same
// (seed, element offset) produces the same uniform draw on the CPU engine (this
// Go implementation) and on the GB10 GPU (the CUDA kernel in
// internal/cuda/kernels/dropout.cu, which mirrors these constants bit-for-bit).
//
// Because the mask is a pure function of (seed, offset, p), dropout never caches
// the mask for backward: backward recomputes it deterministically from the same
// inputs. This keeps the op capture-safe and avoids pinning a saved mask across
// arena resets (ztensor ADR 006, the SaveForBackward lifetime contract).
//
// Reference: Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as
// 1, 2, 3" (SC'11). The 10-round Philox4x32 variant is the standard generator
// PyTorch/cuRAND use for dropout, so matching it here keeps the PyTorch-oracle
// gate meaningful.

const (
	philoxM0 uint32 = 0xD2511F53
	philoxM1 uint32 = 0xCD9E8D57
	philoxW0 uint32 = 0x9E3779B9 // Weyl constant for key[0] (golden ratio)
	philoxW1 uint32 = 0xBB67AE85 // Weyl constant for key[1] (sqrt(3)-1)
	philoxRounds      = 10
)

// philoxMulhilo computes the 64-bit product a*b and returns (hi, lo) 32-bit
// halves, matching the CUDA __umulhi/lo decomposition used in dropout.cu.
func philoxMulhilo(a, b uint32) (hi, lo uint32) {
	product := uint64(a) * uint64(b)
	hi = uint32(product >> 32)
	lo = uint32(product)
	return hi, lo
}

// philox4x32 runs the 10-round Philox4x32 bijection on a 128-bit counter
// (four uint32 words) under a 64-bit key (two uint32 words), returning four
// uint32 outputs. Identical word-for-word to the CUDA device function.
func philox4x32(ctr [4]uint32, key [2]uint32) [4]uint32 {
	c := ctr
	k := key
	for i := 0; i < philoxRounds; i++ {
		hi0, lo0 := philoxMulhilo(philoxM0, c[0])
		hi1, lo1 := philoxMulhilo(philoxM1, c[2])
		c = [4]uint32{
			hi1 ^ c[1] ^ k[0],
			lo1,
			hi0 ^ c[3] ^ k[1],
			lo0,
		}
		// Bump the key (Weyl sequence) for the next round.
		k[0] += philoxW0
		k[1] += philoxW1
	}
	return c
}

// philoxUniform returns a single uniform float64 in [0, 1) for the element at
// linear index `offset` under the 64-bit `seed`. The seed splits into the two
// key words; the counter holds the offset (low/high words) and two zero lanes.
// Only the first output lane is consumed, which is sufficient for a one-draw-
// per-element dropout mask and keeps the CPU and GPU draws trivially aligned.
func philoxUniform(seed uint64, offset uint64) float64 {
	key := [2]uint32{uint32(seed), uint32(seed >> 32)}
	ctr := [4]uint32{uint32(offset), uint32(offset >> 32), 0, 0}
	out := philox4x32(ctr, key)
	// Map a 32-bit word to [0,1): divide by 2^32. Matches the CUDA kernel's
	// (out.x * (1.0f / 4294967296.0f)) but in float64 for the CPU/gradcheck
	// path; the keep/drop decision (u >= p) is identical given the same word.
	return float64(out[0]) * (1.0 / 4294967296.0)
}

// dropoutKeep reports whether the element at `offset` is kept (true) or dropped
// (false) for drop probability p under seed. Keep iff the uniform draw is >= p,
// so p=0 keeps everything and p=1 drops everything. Identical decision on CPU
// and GPU because philoxUniform is bit-reproducible across both.
func dropoutKeep(seed uint64, offset uint64, p float64) bool {
	return philoxUniform(seed, offset) >= p
}

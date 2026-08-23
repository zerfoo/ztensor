package compute

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda/kernels"
	"github.com/zerfoo/ztensor/tensor"
)

// ADR-091 harness 2 (engine parity) for the fused GQA head-expansion kernel,
// the op behind ztensor#180.
//
// Two hazards this file is written against, both live in this repo's history:
//
//  1. GPUEngine.RepeatInterleave falls back to the generic Reshape -> Repeat ->
//     Reshape chain on ANY failure, including "the kernel is not in the
//     deployed libkernels.so". A parity test that only compares values would
//     therefore go green while the fused kernel never ran -- the exact vacuous
//     gate this repo keeps rediscovering. So the test asserts the fused kernel
//     is present FIRST, and fails rather than skips if it is not.
//
//  2. A head-expansion bug is a MAPPING bug: output head q must read KV head
//     q/rep. If every KV head held similar data, a wrong mapping would still
//     compare equal (docs/lore.md L-0009). So each element carries its own
//     source coordinates, and the test proves that encoding is discriminating
//     before it trusts the comparison.

// riEncode returns a value that uniquely identifies the source coordinate and
// is exactly representable in float32, so the comparison below can be exact.
func riEncode(b, kv, s, d int) float32 {
	return float32(((b*64+kv)*64+s)*64 + d)
}

// riReference computes the expected [B, numKV*rep, S, D] output independently
// of any engine: output[b][q][s][d] = input[b][q/rep][s][d]. This is a closed
// form, not a second call into ztensor, so a mapping bug shared by Repeat and
// RepeatInterleave cannot hide behind it.
func riReference(b, numKV, s, d, rep int) []float32 {
	numQ := numKV * rep
	out := make([]float32, b*numQ*s*d)
	i := 0
	for bi := 0; bi < b; bi++ {
		for q := 0; q < numQ; q++ {
			for si := 0; si < s; si++ {
				for di := 0; di < d; di++ {
					out[i] = riEncode(bi, q/rep, si, di)
					i++
				}
			}
		}
	}
	return out
}

func TestGPUEngine_RepeatInterleave_CPUParity(t *testing.T) {
	gpuEng := newTestGPUEngine(t) // skips when CUDA is unavailable
	ctx := context.Background()

	// Hazard 1. Not a skip: on the GB10 this test exists to prove the FUSED
	// path works, and a green that silently measured the fallback is worse
	// than a red. If this fires, rebuild libkernels.so from this tree
	// (internal/cuda/kernels: make CUDA_ARCH=sm_121 shared) and redeploy it to
	// the directory on LD_LIBRARY_PATH.
	if !kernels.IsRepeatInterleaveF32Supported() {
		t.Fatal("launch_repeat_interleave_f32 is not in the loaded libkernels.so; " +
			"GPUEngine.RepeatInterleave would silently fall back and this test " +
			"would prove nothing about the fused kernel (ztensor#180)")
	}

	cases := []struct {
		name                string
		B, numKV, S, D, rep int
	}{
		// The head counts from the ztensor#180 repro (zerfoo TestGPUParity_GQA).
		{"issue180_repro_1x2x2x8_rep2", 1, 2, 2, 8, 2},
		// rep=1 is the identity case: numQ == numKV.
		{"identity_rep1", 2, 4, 3, 16, 1},
		// Llama-ish 32 query heads over 8 KV heads.
		{"llama_8kv_rep4", 1, 8, 7, 64, 4},
		// Batch > 1 exercises the b term of the index decomposition, which a
		// single-batch test cannot see at all.
		{"batch3_rep3", 3, 2, 5, 32, 3},
		// Total elements not a multiple of the 256-thread block: exercises the
		// kernel's `idx >= total` tail guard.
		{"ragged_tail", 1, 3, 5, 7, 2},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := make([]float32, tc.B*tc.numKV*tc.S*tc.D)
			i := 0
			for b := 0; b < tc.B; b++ {
				for kv := 0; kv < tc.numKV; kv++ {
					for s := 0; s < tc.S; s++ {
						for d := 0; d < tc.D; d++ {
							in[i] = riEncode(b, kv, s, d)
							i++
						}
					}
				}
			}

			want := riReference(tc.B, tc.numKV, tc.S, tc.D, tc.rep)

			// Hazard 2, self-test: prove the reference actually distinguishes
			// KV heads before trusting equality against it. If numKV > 1 the
			// slice for output head 0 must differ from the slice for output
			// head `rep`, which reads a DIFFERENT KV head. Without this, a
			// degenerate encoding would make any mapping compare equal.
			if tc.numKV > 1 {
				stride := tc.S * tc.D
				same := true
				for j := 0; j < stride; j++ {
					if want[j] != want[tc.rep*stride+j] {
						same = false
						break
					}
				}
				if same {
					t.Fatalf("reference is position-blind: output heads 0 and %d are "+
						"identical, so a wrong KV-head mapping could not be detected", tc.rep)
				}
			}

			a, err := tensor.New[float32]([]int{tc.B, tc.numKV, tc.S, tc.D}, in)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			got, err := gpuEng.RepeatInterleave(ctx, a, 1, tc.rep)
			if err != nil {
				t.Fatalf("RepeatInterleave: %v", err)
			}

			wantShape := []int{tc.B, tc.numKV * tc.rep, tc.S, tc.D}
			gotShape := got.Shape()
			if len(gotShape) != len(wantShape) {
				t.Fatalf("shape rank: got %v want %v", gotShape, wantShape)
			}
			for j := range wantShape {
				if gotShape[j] != wantShape[j] {
					t.Fatalf("shape: got %v want %v", gotShape, wantShape)
				}
			}

			gd := got.Data()
			if len(gd) != len(want) {
				t.Fatalf("length: got %d want %d", len(gd), len(want))
			}
			// Exact equality is the correct tolerance: repeat-interleave is a
			// pure gather, no arithmetic is performed on the values, so any
			// nonzero difference is a real defect. A float tolerance here
			// would be the ztensor#182 mistake (a bound looser than the
			// measured error, which can no longer fail).
			mismatches := 0
			for j := range gd {
				if gd[j] != want[j] {
					if mismatches < 5 {
						t.Errorf("element %d: got %v want %v", j, gd[j], want[j])
					}
					mismatches++
				}
			}
			if mismatches > 0 {
				t.Fatalf("%d/%d elements differ from the reference expansion", mismatches, len(gd))
			}
		})
	}
}

// TestGPUEngine_RepeatInterleave_UnsupportedFallsBack pins the contract the
// zerfoo caller relies on: for shapes/axes the fused kernel does not cover,
// RepeatInterleave must return a correct result via the generic path rather
// than error or crash.
func TestGPUEngine_RepeatInterleave_UnsupportedFallsBack(t *testing.T) {
	gpuEng := newTestGPUEngine(t)
	ctx := context.Background()

	// axis != 1 is outside the fused kernel's contract.
	in := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	a, err := tensor.New[float32]([]int{1, 2, 2, 2}, in)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	got, err := gpuEng.RepeatInterleave(ctx, a, 2, 2)
	if err != nil {
		t.Fatalf("RepeatInterleave on the fallback path: %v", err)
	}
	if got.Size() != len(in)*2 {
		t.Fatalf("fallback size: got %d want %d", got.Size(), len(in)*2)
	}
}

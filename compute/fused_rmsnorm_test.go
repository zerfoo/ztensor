package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestFusedRMSNorm_MatchesUnfused(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	cases := []struct {
		name    string
		shape   []int
		epsilon float32
	}{
		{"2D_4x8", []int{4, 8}, 1e-6},
		{"3D_2x3x16", []int{2, 3, 16}, 1e-5},
		{"large_1x128x1152", []int{1, 128, 1152}, 1e-6},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			D := tc.shape[len(tc.shape)-1]

			input, _ := tensor.New[float32](tc.shape, nil)
			if err := engine.RandomUniform(ctx, input, -1, 1); err != nil {
				t.Fatal(err)
			}

			weight, _ := tensor.New[float32]([]int{D}, nil)
			if err := engine.RandomUniform(ctx, weight, 0.5, 1.5); err != nil {
				t.Fatal(err)
			}

			// Unfused path: x * rsqrt(mean(x^2) + eps) * weight
			squared, _ := engine.Mul(ctx, input, input, nil)
			lastDim := len(tc.shape) - 1
			meanSq, _ := engine.ReduceMean(ctx, squared, lastDim, true)
			meanSqPlusEps, _ := engine.AddScalar(ctx, meanSq, tc.epsilon, nil)
			rsqrt, _ := engine.Rsqrt(ctx, meanSqPlusEps, nil)
			normalized, _ := engine.Mul(ctx, input, rsqrt, nil)
			unfused, _ := engine.Mul(ctx, normalized, weight, nil)

			// Fused path.
			fused, _, err := FusedRMSNorm(input, weight, tc.epsilon)
			if err != nil {
				t.Fatal(err)
			}

			// Compare.
			unfusedData := unfused.Data()
			fusedData := fused.Data()
			if len(unfusedData) != len(fusedData) {
				t.Fatalf("size mismatch: %d vs %d", len(unfusedData), len(fusedData))
			}

			maxDiff := float64(0)
			for i := range unfusedData {
				diff := math.Abs(float64(unfusedData[i] - fusedData[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			if maxDiff > 1e-5 {
				t.Errorf("max diff %e exceeds tolerance 1e-5", maxDiff)
			}
		})
	}
}

func TestFusedRMSNorm_KnownValues(t *testing.T) {
	// Simple case: input=[1,2,3,4], weight=[1,1,1,1], eps=0
	// mean(x^2) = (1+4+9+16)/4 = 7.5
	// rsqrt(7.5) = 1/sqrt(7.5) ~= 0.3651484
	// output = input * 0.3651484 * 1 = [0.3651, 0.7303, 1.0954, 1.4606]
	input, _ := tensor.New([]int{1, 4}, []float32{1, 2, 3, 4})
	weight, _ := tensor.New([]int{4}, []float32{1, 1, 1, 1})

	out, _, err := FusedRMSNorm(input, weight, 0)
	if err != nil {
		t.Fatal(err)
	}

	expected := 1.0 / math.Sqrt(7.5)
	data := out.Data()
	for i, v := range data {
		want := float32(float64(i+1) * expected)
		if math.Abs(float64(v-want)) > 1e-5 {
			t.Errorf("data[%d]: want %f, got %f", i, want, v)
		}
	}
}

func BenchmarkFusedRMSNorm(b *testing.B) {
	sizes := []struct {
		name  string
		shape []int
	}{
		{"1x128x1152", []int{1, 128, 1152}},
		{"1x256x2048", []int{1, 256, 2048}},
	}

	for _, tc := range sizes {
		D := tc.shape[len(tc.shape)-1]
		input := allocF32(tc.shape)
		weight := allocF32([]int{D})
		e := newEngineF32()
		fillUniform(e, input, -1, 1)
		fillUniform(e, weight, 0.5, 1.5)

		b.Run("fused/"+tc.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, _, err := FusedRMSNorm(input, weight, 1e-6); err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run("unfused/"+tc.name, func(b *testing.B) {
			ctx := context.Background()
			for i := 0; i < b.N; i++ {
				sq, _ := e.Mul(ctx, input, input, nil)
				mean, _ := e.ReduceMean(ctx, sq, len(tc.shape)-1, true)
				eps, _ := e.AddScalar(ctx, mean, float32(1e-6), nil)
				rsqrt, _ := e.Rsqrt(ctx, eps, nil)
				norm, _ := e.Mul(ctx, input, rsqrt, nil)
				_, _ = e.Mul(ctx, norm, weight, nil)
			}
		})
	}
}

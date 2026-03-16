package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestFusedSiLUGate_MatchesUnfused(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	cases := []struct {
		name  string
		shape []int
	}{
		{"2D_4x8", []int{4, 8}},
		{"3D_2x3x16", []int{2, 3, 16}},
		{"large_1x128x1152", []int{1, 128, 1152}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gate, _ := tensor.New[float32](tc.shape, nil)
			up, _ := tensor.New[float32](tc.shape, nil)
			if err := engine.RandomUniform(ctx, gate, -2, 2); err != nil {
				t.Fatal(err)
			}
			if err := engine.RandomUniform(ctx, up, -2, 2); err != nil {
				t.Fatal(err)
			}

			// Unfused: sigmoid(gate) * gate * up
			sigGate, _ := engine.UnaryOp(ctx, gate, func(v float32) float32 {
				return float32(1.0 / (1.0 + math.Exp(-float64(v))))
			})
			siluGate, _ := engine.Mul(ctx, gate, sigGate, nil)
			unfused, _ := engine.Mul(ctx, siluGate, up, nil)

			// Fused path.
			fused, err := FusedSiLUGate(gate, up)
			if err != nil {
				t.Fatal(err)
			}

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

func TestFusedSiLUGate_KnownValues(t *testing.T) {
	// silu(0) = 0, silu(1) = 1*sigmoid(1) = 0.7310586
	// gate=[0,1], up=[2,3]
	// out[0] = silu(0) * 2 = 0
	// out[1] = silu(1) * 3 = 0.7310586 * 3 = 2.1931758
	gate, _ := tensor.New([]int{1, 2}, []float32{0, 1})
	up, _ := tensor.New([]int{1, 2}, []float32{2, 3})

	out, err := FusedSiLUGate(gate, up)
	if err != nil {
		t.Fatal(err)
	}

	data := out.Data()
	sigmoid1 := 1.0 / (1.0 + math.Exp(-1.0))
	expected := []float32{0, float32(1.0 * sigmoid1 * 3.0)}

	for i, v := range data {
		if math.Abs(float64(v-expected[i])) > 1e-5 {
			t.Errorf("data[%d]: want %f, got %f", i, expected[i], v)
		}
	}
}

func TestFusedSiLUGate_ShapeMismatch(t *testing.T) {
	gate, _ := tensor.New([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	up, _ := tensor.New([]int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})

	_, err := FusedSiLUGate(gate, up)
	if err == nil {
		t.Error("expected error for shape mismatch")
	}
}

func BenchmarkFusedSiLUGate(b *testing.B) {
	sizes := []struct {
		name  string
		shape []int
	}{
		{"1x128x1152", []int{1, 128, 1152}},
		{"1x256x2048", []int{1, 256, 2048}},
	}

	for _, tc := range sizes {
		gate := allocF32(tc.shape)
		up := allocF32(tc.shape)
		e := newEngineF32()
		fillUniform(e, gate, -2, 2)
		fillUniform(e, up, -2, 2)

		b.Run("fused/"+tc.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := FusedSiLUGate(gate, up); err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run("unfused/"+tc.name, func(b *testing.B) {
			ctx := context.Background()
			for i := 0; i < b.N; i++ {
				sig, _ := e.UnaryOp(ctx, gate, func(v float32) float32 {
					return float32(1.0 / (1.0 + math.Exp(-float64(v))))
				})
				silu, _ := e.Mul(ctx, gate, sig, nil)
				_, _ = e.Mul(ctx, silu, up, nil)
			}
		})
	}
}

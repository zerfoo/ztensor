package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestFusedRoPE_MatchesUnfused(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	cases := []struct {
		name      string
		batch     int
		seqLen    int
		headDim   int
		rotaryDim int
	}{
		{"full_1x4x8", 1, 4, 8, 8},
		{"full_2x3x16", 2, 3, 16, 16},
		{"partial_1x4x8_rot6", 1, 4, 8, 6},
		{"large_1x128x256", 1, 128, 256, 256},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			halfRotary := tc.rotaryDim / 2

			input, _ := tensor.New[float32]([]int{tc.batch, tc.seqLen, tc.headDim}, nil)
			if err := engine.RandomUniform(ctx, input, -1, 1); err != nil {
				t.Fatal(err)
			}

			// Build cos/sin tables.
			cosData := make([]float32, tc.seqLen*halfRotary)
			sinData := make([]float32, tc.seqLen*halfRotary)
			for s := range tc.seqLen {
				for i := range halfRotary {
					angle := float64(s) / math.Pow(10000.0, float64(2*i)/float64(tc.rotaryDim))
					cosData[s*halfRotary+i] = float32(math.Cos(angle))
					sinData[s*halfRotary+i] = float32(math.Sin(angle))
				}
			}
			cosAngles, _ := tensor.New([]int{tc.seqLen, halfRotary}, cosData)
			sinAngles, _ := tensor.New([]int{tc.seqLen, halfRotary}, sinData)

			// Unfused path (replicate layer logic).
			xRot0, _ := input.Slice([2]int{0, tc.batch}, [2]int{0, tc.seqLen}, [2]int{0, halfRotary})
			xRot1, _ := input.Slice([2]int{0, tc.batch}, [2]int{0, tc.seqLen}, [2]int{halfRotary, tc.rotaryDim})

			t1, _ := engine.Mul(ctx, xRot0, cosAngles)
			t2, _ := engine.Mul(ctx, xRot1, sinAngles)
			rotX0, _ := engine.Sub(ctx, t1, t2)

			m1, _ := engine.Mul(ctx, xRot1, cosAngles)
			m2, _ := engine.Mul(ctx, xRot0, sinAngles)
			rotX1, _ := engine.Add(ctx, m1, m2)

			parts := []*tensor.TensorNumeric[float32]{rotX0, rotX1}
			if tc.rotaryDim < tc.headDim {
				pass, _ := input.Slice([2]int{0, tc.batch}, [2]int{0, tc.seqLen}, [2]int{tc.rotaryDim, tc.headDim})
				parts = append(parts, pass)
			}
			unfused, _ := engine.Concat(ctx, parts, 2)

			// Fused path.
			fused, err := FusedRoPE(input, cosAngles, sinAngles, tc.rotaryDim)
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

func TestFusedRoPE_KnownValues(t *testing.T) {
	// input = [[1,2,3,4]], cos=[1,0], sin=[0,1] (90-degree rotation at pos 1)
	// For pos 0: cos=[1,0], sin=[0,1] => no rotation at dim 0, full rotation at dim 1
	// Actually let's use simple known angles.
	// input shape [1,1,4], rotaryDim=4, halfRotary=2
	// cos = [c0, c1], sin = [s0, s1]
	// out[0] = in[0]*c0 - in[2]*s0
	// out[1] = in[1]*c1 - in[3]*s1
	// out[2] = in[2]*c0 + in[0]*s0
	// out[3] = in[3]*c1 + in[1]*s1
	input, _ := tensor.New([]int{1, 1, 4}, []float32{1, 2, 3, 4})
	cos, _ := tensor.New([]int{1, 2}, []float32{0.5, 0.8})
	sin, _ := tensor.New([]int{1, 2}, []float32{0.3, 0.6})

	out, err := FusedRoPE(input, cos, sin, 4)
	if err != nil {
		t.Fatal(err)
	}

	data := out.Data()
	expected := []float32{
		1*0.5 - 3*0.3,  // = 0.5 - 0.9 = -0.4
		2*0.8 - 4*0.6,  // = 1.6 - 2.4 = -0.8
		3*0.5 + 1*0.3,  // = 1.5 + 0.3 = 1.8
		4*0.8 + 2*0.6,  // = 3.2 + 1.2 = 4.4
	}

	for i, v := range data {
		if math.Abs(float64(v-expected[i])) > 1e-5 {
			t.Errorf("data[%d]: want %f, got %f", i, expected[i], v)
		}
	}
}

func BenchmarkFusedRoPE(b *testing.B) {
	sizes := []struct {
		name    string
		batch   int
		seqLen  int
		headDim int
	}{
		{"1x128x256", 1, 128, 256},
		{"4x64x128", 4, 64, 128},
	}

	for _, tc := range sizes {
		halfDim := tc.headDim / 2
		input := allocF32([]int{tc.batch, tc.seqLen, tc.headDim})
		cosA := allocF32([]int{tc.seqLen, halfDim})
		sinA := allocF32([]int{tc.seqLen, halfDim})
		e := newEngineF32()
		fillUniform(e, input, -1, 1)
		fillUniform(e, cosA, -1, 1)
		fillUniform(e, sinA, -1, 1)

		b.Run("fused/"+tc.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := FusedRoPE(input, cosA, sinA, tc.headDim); err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run("unfused/"+tc.name, func(b *testing.B) {
			ctx := context.Background()
			for i := 0; i < b.N; i++ {
				x0, _ := input.Slice([2]int{0, tc.batch}, [2]int{0, tc.seqLen}, [2]int{0, halfDim})
				x1, _ := input.Slice([2]int{0, tc.batch}, [2]int{0, tc.seqLen}, [2]int{halfDim, tc.headDim})
				t1, _ := e.Mul(ctx, x0, cosA)
				t2, _ := e.Mul(ctx, x1, sinA)
				r0, _ := e.Sub(ctx, t1, t2)
				m1, _ := e.Mul(ctx, x1, cosA)
				m2, _ := e.Mul(ctx, x0, sinA)
				r1, _ := e.Add(ctx, m1, m2)
				_, _ = e.Concat(ctx, []*tensor.TensorNumeric[float32]{r0, r1}, 2)
			}
		})
	}
}

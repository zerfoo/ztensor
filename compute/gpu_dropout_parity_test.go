package compute

// CPU-GPU parity for the dropout op (BPB.3a). The mask is drawn from the same
// Philox4x32-10 generator on both sides (compute/philox.go for the CPU engine,
// internal/cuda/kernels/dropout.cu for the GPU), keyed by (seed, element
// offset). For an identical seed the masks are bit-identical, so GPU and CPU
// dropout produce identical kept/dropped lanes; the kept lanes are scaled by
// the same host-computed invKeep=1/(1-p) float32, so the outputs match exactly.
// This is the CPU-GPU parity gate; it runs on the GB10 via Spark (cuda tag) and
// skips cleanly on machines without a GPU.

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestGPUDropout_CPUParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()
	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	const seed = uint64(0x9e3779b97f4a7c15)
	shapes := [][]int{{8, 8}, {1, 257}, {4, 32}}
	probs := []float64{0.0, 0.1, 0.3, 0.5, 0.9}

	for _, shape := range shapes {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i%13) - 6.0 + 0.25
		}
		for _, p := range probs {
			for _, training := range []bool{true, false} {
				cx, err := tensor.New[float32](shape, append([]float32(nil), data...))
				if err != nil {
					t.Fatal(err)
				}
				gx, err := tensor.New[float32](shape, append([]float32(nil), data...))
				if err != nil {
					t.Fatal(err)
				}
				cy, err := cpuEng.Dropout(ctx, cx, p, seed, training)
				if err != nil {
					t.Fatalf("CPU Dropout (p=%g train=%v): %v", p, training, err)
				}
				gy, err := gpuEng.Dropout(ctx, gx, p, seed, training)
				if err != nil {
					t.Fatalf("GPU Dropout (p=%g train=%v): %v", p, training, err)
				}
				gd := gy.Data() // GPUStorage.Slice copies D2H
				cd := cy.Data()
				if len(gd) != len(cd) {
					t.Fatalf("len mismatch: gpu %d cpu %d", len(gd), len(cd))
				}
				for i := range cd {
					if gd[i] != cd[i] {
						t.Fatalf("dropout parity mismatch shape=%v p=%g train=%v at %d: gpu=%g cpu=%g",
							shape, p, training, i, gd[i], cd[i])
					}
				}
			}
		}
	}
}

// TestGPUDropout_Backward_CPUParity checks the backward (masked-scale of the
// upstream gradient) matches CPU bit-for-bit under the same seed.
func TestGPUDropout_Backward_CPUParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()
	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	const seed = uint64(12345)
	const p = 0.4
	n := 128
	g := make([]float32, n)
	for i := range g {
		g[i] = float32(i) - 64
	}
	cg, err := tensor.New[float32]([]int{16, 8}, append([]float32(nil), g...))
	if err != nil {
		t.Fatal(err)
	}
	gg, err := tensor.New[float32]([]int{16, 8}, append([]float32(nil), g...))
	if err != nil {
		t.Fatal(err)
	}
	cdx, err := cpuEng.DropoutBackward(ctx, cg, p, seed, true)
	if err != nil {
		t.Fatal(err)
	}
	gdx, err := gpuEng.DropoutBackward(ctx, gg, p, seed, true)
	if err != nil {
		t.Fatal(err)
	}
	cd, gd := cdx.Data(), gdx.Data()
	for i := range cd {
		if gd[i] != cd[i] {
			t.Fatalf("dropout backward parity mismatch at %d: gpu=%g cpu=%g", i, gd[i], cd[i])
		}
	}
}

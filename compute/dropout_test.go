package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestPhiloxDeterminism: the same (seed, offset) always yields the same draw,
// and different offsets/seeds yield different draws (so the mask is not constant).
func TestPhiloxDeterminism(t *testing.T) {
	const seed = uint64(0x9e3779b97f4a7c15)
	for off := uint64(0); off < 16; off++ {
		a := philoxUniform(seed, off)
		b := philoxUniform(seed, off)
		if a != b {
			t.Fatalf("philoxUniform not deterministic at offset %d: %g != %g", off, a, b)
		}
		if a < 0 || a >= 1 {
			t.Fatalf("philoxUniform offset %d out of [0,1): %g", off, a)
		}
	}
	if philoxUniform(seed, 0) == philoxUniform(seed, 1) {
		t.Fatal("philoxUniform identical for offsets 0 and 1 (mask would be constant)")
	}
	if philoxUniform(1, 0) == philoxUniform(2, 0) {
		t.Fatal("philoxUniform identical for seeds 1 and 2")
	}
}

// TestPhiloxUniformMean: over many draws the empirical mean is ~0.5, confirming
// the [0,1) mapping is unbiased (so 1-p is the true keep rate).
func TestPhiloxUniformMean(t *testing.T) {
	const n = 1 << 16
	var sum float64
	for i := uint64(0); i < n; i++ {
		sum += philoxUniform(0xdeadbeefcafef00d, i)
	}
	mean := sum / float64(n)
	if math.Abs(mean-0.5) > 0.01 {
		t.Fatalf("philox uniform mean %g not within 0.01 of 0.5", mean)
	}
}

// TestDropout_EvalIdentity: eval mode returns an exact copy regardless of p.
func TestDropout_EvalIdentity(t *testing.T) {
	e := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()
	x, err := tensor.New[float32]([]int{2, 4}, []float32{1, -2, 3, -4, 5, -6, 7, -8})
	if err != nil {
		t.Fatal(err)
	}
	y, err := e.Dropout(ctx, x, 0.5, 42, false)
	if err != nil {
		t.Fatalf("Dropout eval: %v", err)
	}
	for i, v := range y.Data() {
		if v != x.Data()[i] {
			t.Fatalf("eval mode not identity at %d: got %g want %g", i, v, x.Data()[i])
		}
	}
}

// TestDropout_PZeroIdentity: training mode with p=0 is exact identity (no scale,
// no drop).
func TestDropout_PZeroIdentity(t *testing.T) {
	e := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()
	x, err := tensor.New[float32]([]int{2, 4}, []float32{1, -2, 3, -4, 5, -6, 7, -8})
	if err != nil {
		t.Fatal(err)
	}
	y, err := e.Dropout(ctx, x, 0, 42, true)
	if err != nil {
		t.Fatalf("Dropout p=0: %v", err)
	}
	for i, v := range y.Data() {
		if v != x.Data()[i] {
			t.Fatalf("p=0 not identity at %d: got %g want %g", i, v, x.Data()[i])
		}
	}
}

// TestDropout_MaskDeterminism: same seed => identical output across calls; kept
// elements are scaled by 1/(1-p), dropped ones are zero, and the same Philox
// draw governs which is which.
func TestDropout_MaskDeterminism(t *testing.T) {
	e := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()
	const p = 0.3
	const seed = uint64(123456789)
	n := 64
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i + 1)
	}
	x, err := tensor.New[float32]([]int{8, 8}, data)
	if err != nil {
		t.Fatal(err)
	}
	y1, err := e.Dropout(ctx, x, p, seed, true)
	if err != nil {
		t.Fatal(err)
	}
	y2, err := e.Dropout(ctx, x, p, seed, true)
	if err != nil {
		t.Fatal(err)
	}
	scale := float32(1.0 / (1.0 - p))
	keptSeen, dropSeen := false, false
	for i := 0; i < n; i++ {
		if y1.Data()[i] != y2.Data()[i] {
			t.Fatalf("mask not deterministic at %d: %g != %g", i, y1.Data()[i], y2.Data()[i])
		}
		keep := dropoutKeep(seed, uint64(i), p)
		var want float32
		if keep {
			want = x.Data()[i] * scale
			keptSeen = true
		} else {
			dropSeen = true
		}
		if y1.Data()[i] != want {
			t.Fatalf("element %d keep=%v: got %g want %g", i, keep, y1.Data()[i], want)
		}
	}
	if !keptSeen || !dropSeen {
		t.Fatalf("expected both kept and dropped elements (kept=%v drop=%v)", keptSeen, dropSeen)
	}
}

// TestDropout_ExpectedMeanScale: E[y] == E[x] under inverted dropout because the
// 1/(1-p) scale compensates for the (1-p) keep rate. Checked over a large
// constant input so the empirical mean is tightly concentrated.
func TestDropout_ExpectedMeanScale(t *testing.T) {
	e := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()
	const p = 0.4
	n := 1 << 14
	data := make([]float32, n)
	for i := range data {
		data[i] = 2.0
	}
	x, err := tensor.New[float32]([]int{n}, data)
	if err != nil {
		t.Fatal(err)
	}
	y, err := e.Dropout(ctx, x, p, 0xabcdef, true)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, v := range y.Data() {
		sum += float64(v)
	}
	mean := sum / float64(n)
	if math.Abs(mean-2.0) > 0.05 {
		t.Fatalf("inverted-dropout mean %g not within 0.05 of input mean 2.0", mean)
	}
}

// TestDropout_BackwardMatchesMask: backward applies the SAME mask/scale as
// forward (dropout is linear in its input given the mask), and recomputes the
// mask deterministically rather than relying on a cached one.
func TestDropout_BackwardMatchesMask(t *testing.T) {
	e := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()
	const p = 0.25
	const seed = uint64(777)
	n := 32
	g := make([]float32, n)
	for i := range g {
		g[i] = float32(i) - 16
	}
	gt, err := tensor.New[float32]([]int{4, 8}, g)
	if err != nil {
		t.Fatal(err)
	}
	dx, err := e.DropoutBackward(ctx, gt, p, seed, true)
	if err != nil {
		t.Fatal(err)
	}
	scale := float32(1.0 / (1.0 - p))
	for i := 0; i < n; i++ {
		var want float32
		if dropoutKeep(seed, uint64(i), p) {
			want = g[i] * scale
		}
		if dx.Data()[i] != want {
			t.Fatalf("backward element %d: got %g want %g", i, dx.Data()[i], want)
		}
	}
	// Eval-mode backward is a pass-through.
	dxe, err := e.DropoutBackward(ctx, gt, p, seed, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if dxe.Data()[i] != g[i] {
			t.Fatalf("eval backward not pass-through at %d: got %g want %g", i, dxe.Data()[i], g[i])
		}
	}
}

// TestDropout_InvalidP: p outside [0,1) is rejected.
func TestDropout_InvalidP(t *testing.T) {
	e := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()
	x, err := tensor.New[float32]([]int{2}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []float64{-0.1, 1.0, 1.5} {
		if _, err := e.Dropout(ctx, x, p, 0, true); err == nil {
			t.Fatalf("expected error for p=%g, got nil", p)
		}
	}
}

package oracle

import (
	"math"
	"path/filepath"
	"testing"
)

// fastMathTanh emulates the historical ztensor#125 bug class WITHOUT torch:
// the GPU GELU kernel built with global --use_fast_math computed tanh through
// an unsaturated approximation whose output kept growing past |x| ~ 9 instead
// of clamping to +-1 (the raw cubic GELU tanh-argument overflow). Inside the
// well-behaved range it matches math.Tanh; beyond it, it grows linearly the
// way the unclamped fast-math expansion did.
func fastMathTanh(x float64) float64 {
	const sat = 9.0
	if math.Abs(x) <= sat {
		return math.Tanh(x)
	}
	return math.Tanh(sat) * x / sat // |result| > 1, growing with |x|
}

// writeTanhBundle writes a complete Tanh case bundle whose recorded "ztensor"
// forward output and input gradient are produced by the supplied tanh
// implementation. Inputs deliberately include |x| > 9 -- the overflow region
// where the fast-math approximation diverges from the true tanh.
func writeTanhBundle(t *testing.T, dir string, tanh func(float64) float64) *Bundle {
	t.Helper()
	inputs := []float64{-14, -9.5, -0.3, 0.6, 9.7, 12}
	upstream := []float64{1, -0.5, 0.75, -1, 0.25, 0.9}
	shape := []int{2, 3}

	fwd := make([]float64, len(inputs))
	grad := make([]float64, len(inputs))
	for i, x := range inputs {
		y := float64(float32(tanh(x))) // f32 storage, like the real kernel
		fwd[i] = y
		grad[i] = float64(float32(upstream[i] * (1 - y*y)))
	}

	in := TensorRef{Name: "x0", File: "input_0.bin", Shape: shape, DType: DTypeFloat32}
	up := TensorRef{Name: "upstream", File: "upstream.bin", Shape: shape, DType: DTypeFloat32}
	fw := TensorRef{Name: "forward", File: "forward.bin", Shape: shape, DType: DTypeFloat32}
	gi := TensorRef{Name: "x0", File: "grad_input_0.bin", Shape: shape, DType: DTypeFloat32}
	for _, w := range []struct {
		ref  TensorRef
		vals []float64
	}{{in, inputs}, {up, upstream}, {fw, fwd}, {gi, grad}} {
		if err := WriteTensorFile(filepath.Join(dir, w.ref.File), w.ref.DType, w.vals); err != nil {
			t.Fatal(err)
		}
	}
	m := &Manifest{
		Op:         "Tanh",
		TorchExpr:  "torch.tanh(x0)",
		DType:      DTypeFloat32,
		Seed:       125, // ztensor#125, the bug this fixture encodes
		Tolerance:  toleranceFor("Tanh"),
		Inputs:     []TensorRef{in},
		Upstream:   up,
		Forward:    fw,
		InputGrads: []TensorRef{gi},
	}
	if err := WriteBundle(dir, m); err != nil {
		t.Fatal(err)
	}
	b, err := ReadBundle(dir)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// checkTanhBundle is the Go reference checker: it mimics exactly what
// scripts/oracle/run_oracle.py computes on the DGX, with math.Tanh standing
// in for torch.tanh as the ground truth (legitimate here: torch.tanh is
// correctly-rounded-class and saturates, as math.Tanh does). It reads the
// bundle, recomputes forward and backward references from the recorded
// inputs and upstream, and diffs the recorded ztensor outputs against them
// within the manifest tolerances.
func checkTanhBundle(t *testing.T, b *Bundle) (fwdStats, gradStats DiffStats) {
	t.Helper()
	m := b.Manifest
	inputs, err := b.Tensor(m.Inputs[0])
	if err != nil {
		t.Fatal(err)
	}
	upstream, err := b.Tensor(m.Upstream)
	if err != nil {
		t.Fatal(err)
	}
	recordedFwd, err := b.Tensor(m.Forward)
	if err != nil {
		t.Fatal(err)
	}
	recordedGrad, err := b.Tensor(m.InputGrads[0])
	if err != nil {
		t.Fatal(err)
	}

	// Ground truth at f32, the dtype the bundle declares.
	refFwd := make([]float64, len(inputs))
	refGrad := make([]float64, len(inputs))
	for i, x := range inputs {
		y := float64(float32(math.Tanh(x)))
		refFwd[i] = y
		refGrad[i] = float64(float32(upstream[i] * (1 - y*y)))
	}

	fwdStats = Diff(recordedFwd, refFwd, m.Tolerance.FwdAtol, m.Tolerance.FwdRtol)
	gradStats = Diff(recordedGrad, refGrad, m.Tolerance.GradAtol, m.Tolerance.GradRtol)
	return fwdStats, gradStats
}

// TestRedProofFastMathTanhFails is the mandatory CI-side red proof for the
// oracle harness (plan T1.3): a bundle recorded with the unsaturated
// fast-math tanh MUST fail the diff against the true tanh, in both forward
// and backward, proving the harness would have flagged ztensor#125. The real
// torch comparison runs on the DGX; this test proves the diff/report logic.
func TestRedProofFastMathTanhFails(t *testing.T) {
	b := writeTanhBundle(t, t.TempDir(), fastMathTanh)
	fwd, grad := checkTanhBundle(t, b)

	if fwd.Pass {
		t.Fatalf("red proof FAILED: fast-math tanh forward passed the oracle diff (stats %+v)", fwd)
	}
	// At x=14 the unsaturated approximation yields ~1.555 vs tanh's 1.0 --
	// the divergence is gross, not marginal. Require a diff far beyond the
	// tolerance so a future tolerance loosening cannot silently absorb it.
	if fwd.MaxAbs < 0.1 {
		t.Fatalf("fast-math tanh max abs diff %v suspiciously small; fixture no longer encodes the overflow", fwd.MaxAbs)
	}
	if grad.Pass {
		t.Fatalf("red proof FAILED: fast-math tanh backward passed the oracle diff (stats %+v)", grad)
	}
	if fwd.Mismatches == 0 || fwd.Mismatches >= fwd.Checked {
		t.Fatalf("expected only the |x|>9 elements to mismatch, got %d/%d", fwd.Mismatches, fwd.Checked)
	}
}

// TestGreenProofTrueTanhPasses is the green twin: the same bundle recorded
// with the correct saturating tanh passes the identical checker, proving the
// red proof fails because of the numerics, not because of the harness.
func TestGreenProofTrueTanhPasses(t *testing.T) {
	b := writeTanhBundle(t, t.TempDir(), math.Tanh)
	fwd, grad := checkTanhBundle(t, b)
	if !fwd.Pass {
		t.Fatalf("true tanh forward failed the oracle diff: %+v", fwd)
	}
	if !grad.Pass {
		t.Fatalf("true tanh backward failed the oracle diff: %+v", grad)
	}
	if fwd.Checked != 6 || grad.Checked != 6 {
		t.Fatalf("checked %d/%d elements, want 6/6", fwd.Checked, grad.Checked)
	}
}

// TestDiffNaNFails pins the NaN policy: a NaN anywhere is an automatic
// failure (NaNs are the symptom this whole plan exists to catch).
func TestDiffNaNFails(t *testing.T) {
	s := Diff([]float64{1, math.NaN()}, []float64{1, 1}, 1, 1)
	if s.Pass || s.Mismatches != 1 || !math.IsInf(s.MaxAbs, 1) {
		t.Fatalf("NaN did not fail the diff: %+v", s)
	}
	// NaN in the reference also fails.
	s = Diff([]float64{1, 1}, []float64{math.NaN(), 1}, 1, 1)
	if s.Pass {
		t.Fatalf("NaN reference did not fail the diff: %+v", s)
	}
}

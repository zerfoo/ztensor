package oracle

import "math"

// DiffStats summarizes an elementwise comparison of a recorded ztensor tensor
// against a reference (torch, or a Go ground truth in CI tests). The pass
// criterion per element is |got-ref| <= atol + rtol*|ref|; any NaN on either
// side fails the element. This logic is mirrored exactly by
// scripts/oracle/run_oracle.py -- the CI red-proof test exercises the Go copy
// so the report semantics are proven without a torch dependency.
type DiffStats struct {
	// MaxAbs is the largest |got-ref| (Inf when a NaN was seen).
	MaxAbs float64 `json:"max_abs"`
	// MaxRel is the largest |got-ref| / max(|ref|, 1e-12).
	MaxRel float64 `json:"max_rel"`
	// Checked is the number of elements compared.
	Checked int `json:"checked"`
	// Mismatches is the number of out-of-tolerance elements.
	Mismatches int `json:"mismatches"`
	// Pass is true when every element was within tolerance.
	Pass bool `json:"pass"`
}

// Diff compares got against ref within atol/rtol. The slices must be the same
// length; the caller validates shapes via the manifest.
func Diff(got, ref []float64, atol, rtol float64) DiffStats {
	s := DiffStats{Checked: len(got), Pass: true}
	for i := range got {
		g, r := got[i], ref[i]
		if math.IsNaN(g) || math.IsNaN(r) {
			s.Mismatches++
			s.Pass = false
			s.MaxAbs = math.Inf(1)
			s.MaxRel = math.Inf(1)
			continue
		}
		abs := math.Abs(g - r)
		rel := abs / math.Max(math.Abs(r), 1e-12)
		if abs > s.MaxAbs {
			s.MaxAbs = abs
		}
		if rel > s.MaxRel {
			s.MaxRel = rel
		}
		if abs > atol+rtol*math.Abs(r) {
			s.Mismatches++
			s.Pass = false
		}
	}
	return s
}

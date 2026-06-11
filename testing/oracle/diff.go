package oracle

import (
	"encoding/json"
	"fmt"
	"math"
)

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

// nonFiniteFloat is a float64 whose JSON form survives NaN/Inf: a NaN
// comparison drives MaxAbs/MaxRel to +Inf, which strict JSON cannot carry as
// a number (Go's encoding/json errors on it; Python's json module emits the
// non-strict bare Infinity literal). Non-finite values are encoded as the
// quoted strings "NaN", "Infinity", "-Infinity" instead.
type nonFiniteFloat float64

func (f nonFiniteFloat) MarshalJSON() ([]byte, error) {
	v := float64(f)
	switch {
	case math.IsNaN(v):
		return []byte(`"NaN"`), nil
	case math.IsInf(v, 1):
		return []byte(`"Infinity"`), nil
	case math.IsInf(v, -1):
		return []byte(`"-Infinity"`), nil
	}
	return json.Marshal(v)
}

func (f *nonFiniteFloat) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		switch s {
		case "NaN":
			*f = nonFiniteFloat(math.NaN())
		case "Infinity":
			*f = nonFiniteFloat(math.Inf(1))
		case "-Infinity":
			*f = nonFiniteFloat(math.Inf(-1))
		default:
			return fmt.Errorf("oracle: %q is not a non-finite float marker", s)
		}
		return nil
	}
	var v float64
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	*f = nonFiniteFloat(v)
	return nil
}

// diffStatsJSON is the wire form of DiffStats with non-finite-safe floats.
type diffStatsJSON struct {
	MaxAbs     nonFiniteFloat `json:"max_abs"`
	MaxRel     nonFiniteFloat `json:"max_rel"`
	Checked    int            `json:"checked"`
	Mismatches int            `json:"mismatches"`
	Pass       bool           `json:"pass"`
}

// MarshalJSON encodes the stats with NaN/Inf-safe max_abs/max_rel, so a
// poison-NaN verdict (MaxAbs=+Inf) serializes into the report instead of
// failing the whole report write.
func (s DiffStats) MarshalJSON() ([]byte, error) {
	return json.Marshal(diffStatsJSON{
		MaxAbs:     nonFiniteFloat(s.MaxAbs),
		MaxRel:     nonFiniteFloat(s.MaxRel),
		Checked:    s.Checked,
		Mismatches: s.Mismatches,
		Pass:       s.Pass,
	})
}

// UnmarshalJSON parses both plain numbers and the quoted non-finite markers.
func (s *DiffStats) UnmarshalJSON(b []byte) error {
	var w diffStatsJSON
	if err := json.Unmarshal(b, &w); err != nil {
		return err
	}
	*s = DiffStats{
		MaxAbs:     float64(w.MaxAbs),
		MaxRel:     float64(w.MaxRel),
		Checked:    w.Checked,
		Mismatches: w.Mismatches,
		Pass:       w.Pass,
	}
	return nil
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

// Package gradcheck provides a finite-difference gradient checker for
// graph.Node implementations, modeled on torch.autograd.gradcheck and the
// PyTorch OpInfo registry (see zerfoo docs/adr/091).
//
// The checker compares a node's analytic Backward against float64 central
// differences computed on the CPU engine. Each finite-difference evaluation
// constructs a FRESH node instance via the supplied constructor closure, so
// nodes that cache intermediates during Forward (softmax output, layernorm
// statistics, ...) cannot leak stale state between perturbed evaluations.
package gradcheck

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Default checker constants (float64, CPU).
const (
	// DefaultEps is the base central-difference step. The effective step for
	// an element x is DefaultEps * max(1, |x|).
	DefaultEps = 1e-6
	// DefaultAtol is the default absolute tolerance at float64.
	DefaultAtol = 1e-7
	// DefaultRtol is the default relative tolerance at float64.
	DefaultRtol = 1e-5
	// DefaultMaxMismatches caps how many per-element mismatches are recorded
	// in a Report before the checker stops collecting (it keeps counting).
	DefaultMaxMismatches = 10
)

// Tolerance bundles absolute and relative comparison tolerances.
// A mismatch is flagged when |analytic-numeric| > Atol + Rtol*max(|analytic|, |numeric|).
type Tolerance struct {
	Atol float64
	Rtol float64
}

// Config controls a single gradcheck run. The zero value selects defaults.
type Config struct {
	// Eps is the base perturbation step (default DefaultEps). The effective
	// step is scaled by max(1, |x|) per element.
	Eps float64
	// Tol holds the comparison tolerances (defaults DefaultAtol/DefaultRtol).
	Tol Tolerance
	// Upstream optionally supplies the upstream gradient dL/dy. When nil, a
	// deterministic pseudo-random upstream with entries in +-[0.25, 1.0] is
	// generated from Seed; randomized upstreams catch structural Jacobian
	// errors (e.g. transposed gradients) that an all-ones upstream can mask.
	Upstream *tensor.TensorNumeric[float64]
	// Seed seeds the generated upstream gradient (ignored if Upstream != nil).
	Seed int64
	// MaxMismatches caps recorded mismatches (default DefaultMaxMismatches).
	MaxMismatches int
}

func (c *Config) withDefaults() Config {
	out := Config{}
	if c != nil {
		out = *c
	}
	if out.Eps <= 0 {
		out.Eps = DefaultEps
	}
	if out.Tol.Atol <= 0 {
		out.Tol.Atol = DefaultAtol
	}
	if out.Tol.Rtol <= 0 {
		out.Tol.Rtol = DefaultRtol
	}
	if out.MaxMismatches <= 0 {
		out.MaxMismatches = DefaultMaxMismatches
	}
	return out
}

// Mismatch records one element where analytic and numerical gradients
// disagree beyond tolerance.
type Mismatch struct {
	// Target identifies the differentiated tensor, e.g. "input[1]" or
	// `param["gamma"]`.
	Target string
	// Index is the flat element index within Target.
	Index    int
	Analytic float64
	Numeric  float64
}

func (m Mismatch) String() string {
	return fmt.Sprintf("%s[%d]: analytic=%.12g numeric=%.12g |diff|=%.3g",
		m.Target, m.Index, m.Analytic, m.Numeric, math.Abs(m.Analytic-m.Numeric))
}

// Report is the outcome of one gradcheck run.
type Report struct {
	// Op is the OpType of the checked node.
	Op string
	// Checked is the total number of differentiated elements compared.
	Checked int
	// MismatchCount is the total number of out-of-tolerance elements.
	MismatchCount int
	// Mismatches holds up to Config.MaxMismatches recorded mismatches.
	Mismatches []Mismatch
}

// OK reports whether every compared element was within tolerance.
func (r *Report) OK() bool { return r.MismatchCount == 0 }

func (r *Report) String() string {
	if r.OK() {
		return fmt.Sprintf("gradcheck %s: OK (%d elements)", r.Op, r.Checked)
	}
	s := fmt.Sprintf("gradcheck %s: %d/%d elements out of tolerance\n", r.Op, r.MismatchCount, r.Checked)
	for _, m := range r.Mismatches {
		s += "  " + m.String() + "\n"
	}
	return s
}

// MakeNodeFn constructs a fresh node instance. It is invoked once for the
// analytic pass and once per finite-difference Forward evaluation, so any
// state the node caches in Forward is never reused across evaluations.
//
// Constructors must be deterministic in parameter ORDER and SHAPES: the
// checker copies the reference instance's parameter values into every fresh
// instance before evaluating, so randomly initialized parameter VALUES are
// fine.
type MakeNodeFn func() (graph.Node[float64], error)

// Check verifies the analytic Backward of the node produced by makeNode
// against float64 central finite differences, for every element of every
// input tensor and every trainable parameter.
//
// The returned error reports mechanical failures (Forward/Backward errors,
// shape disagreements); gradient disagreements are reported via Report.
func Check(ctx context.Context, makeNode MakeNodeFn, inputs []*tensor.TensorNumeric[float64], cfg *Config) (*Report, error) {
	c := cfg.withDefaults()

	ref, err := makeNode()
	if err != nil {
		return nil, fmt.Errorf("gradcheck: constructing reference node: %w", err)
	}

	// Snapshot reference parameter values; every fresh instance is reset to
	// these so the function under differentiation is well-defined.
	refParams := ref.Parameters()
	baseParams := make([][]float64, len(refParams))
	for i, p := range refParams {
		baseParams[i] = append([]float64(nil), p.Value.Data()...)
	}

	// eval runs Forward on a FRESH node instance with the given parameter
	// values and returns a copy of the flattened output.
	eval := func(paramVals [][]float64) ([]float64, error) {
		n, err := makeNode()
		if err != nil {
			return nil, fmt.Errorf("gradcheck: constructing node: %w", err)
		}
		ps := n.Parameters()
		if len(ps) != len(paramVals) {
			return nil, fmt.Errorf("gradcheck: constructor returned %d parameters, reference had %d", len(ps), len(paramVals))
		}
		for i, p := range ps {
			if len(p.Value.Data()) != len(paramVals[i]) {
				return nil, fmt.Errorf("gradcheck: parameter %q size changed between instances", p.Name)
			}
			copy(p.Value.Data(), paramVals[i])
		}
		y, err := n.Forward(ctx, inputs...)
		if err != nil {
			return nil, fmt.Errorf("gradcheck: Forward: %w", err)
		}
		return append([]float64(nil), y.Data()...), nil
	}

	// Analytic pass on the reference instance.
	y, err := ref.Forward(ctx, inputs...)
	if err != nil {
		return nil, fmt.Errorf("gradcheck: reference Forward: %w", err)
	}

	upstream := c.Upstream
	if upstream == nil {
		upstream, err = randomUpstream(y.Shape(), c.Seed)
		if err != nil {
			return nil, err
		}
	}
	if len(upstream.Data()) != len(y.Data()) {
		return nil, fmt.Errorf("gradcheck: upstream has %d elements, output has %d", len(upstream.Data()), len(y.Data()))
	}
	g := upstream.Data()

	for _, p := range refParams {
		if p.Gradient != nil {
			for i := range p.Gradient.Data() {
				p.Gradient.Data()[i] = 0
			}
		}
	}

	analytic, err := ref.Backward(ctx, types.FullBackprop, upstream, inputs...)
	if err != nil {
		return nil, fmt.Errorf("gradcheck: Backward: %w", err)
	}
	if len(analytic) != len(inputs) {
		return nil, fmt.Errorf("gradcheck: Backward returned %d input gradients, want %d", len(analytic), len(inputs))
	}

	report := &Report{Op: ref.OpType()}

	record := func(target string, idx int, a, n float64) {
		diff := math.Abs(a - n)
		bound := c.Tol.Atol + c.Tol.Rtol*math.Max(math.Abs(a), math.Abs(n))
		report.Checked++
		if diff > bound || math.IsNaN(diff) {
			report.MismatchCount++
			if len(report.Mismatches) < c.MaxMismatches {
				report.Mismatches = append(report.Mismatches, Mismatch{Target: target, Index: idx, Analytic: a, Numeric: n})
			}
		}
	}

	// centralDiff perturbs one element accessed through get/set, re-runs the
	// forward on fresh instances, and contracts with the upstream gradient.
	centralDiff := func(orig float64, set func(v float64), paramVals [][]float64) (float64, error) {
		h := c.Eps * math.Max(1, math.Abs(orig))
		set(orig + h)
		fPlus, err := eval(paramVals)
		if err != nil {
			return 0, err
		}
		set(orig - h)
		fMinus, err := eval(paramVals)
		if err != nil {
			return 0, err
		}
		set(orig)
		if len(fPlus) != len(g) || len(fMinus) != len(g) {
			return 0, fmt.Errorf("gradcheck: output size changed across perturbed evaluations")
		}
		num := 0.0
		for i := range g {
			num += g[i] * (fPlus[i] - fMinus[i]) / (2 * h)
		}
		return num, nil
	}

	// Inputs.
	for j, in := range inputs {
		grad := analytic[j]
		if grad == nil {
			return nil, fmt.Errorf("gradcheck: Backward returned nil gradient for input[%d] (non-differentiable inputs are not yet supported)", j)
		}
		if len(grad.Data()) != len(in.Data()) {
			return nil, fmt.Errorf("gradcheck: gradient for input[%d] has %d elements, input has %d", j, len(grad.Data()), len(in.Data()))
		}
		data := in.Data()
		for k := range data {
			num, err := centralDiff(data[k], func(v float64) { data[k] = v }, baseParams)
			if err != nil {
				return nil, err
			}
			record(fmt.Sprintf("input[%d]", j), k, grad.Data()[k], num)
		}
	}

	// Parameters: gradients are read from the reference instance's
	// Parameter.Gradient after Backward (the accumulation convention).
	for pi, p := range refParams {
		if p.Gradient == nil {
			return nil, fmt.Errorf("gradcheck: parameter %q has nil Gradient after Backward", p.Name)
		}
		if len(p.Gradient.Data()) != len(baseParams[pi]) {
			return nil, fmt.Errorf("gradcheck: gradient for parameter %q has %d elements, value has %d", p.Name, len(p.Gradient.Data()), len(baseParams[pi]))
		}
		vals := baseParams[pi]
		for k := range vals {
			num, err := centralDiff(vals[k], func(v float64) { vals[k] = v }, baseParams)
			if err != nil {
				return nil, err
			}
			record(fmt.Sprintf("param[%q]", p.Name), k, p.Gradient.Data()[k], num)
		}
	}

	return report, nil
}

// randomUpstream builds a deterministic upstream gradient with entries of
// magnitude in [0.25, 1.0] and pseudo-random sign.
func randomUpstream(shape []int, seed int64) (*tensor.TensorNumeric[float64], error) {
	size := 1
	for _, d := range shape {
		size *= d
	}
	r := newRand(seed)
	data := make([]float64, size)
	for i := range data {
		v := 0.25 + 0.75*r.Float64()
		if r.Float64() < 0.5 {
			v = -v
		}
		data[i] = v
	}
	return tensor.New[float64](shape, data)
}

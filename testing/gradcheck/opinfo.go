package gradcheck

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func newRand(seed int64) *rand.Rand {
	// math/rand (not crypto) is intentional: we need cheap, deterministic,
	// reproducible test inputs, not security.
	return rand.New(rand.NewSource(seed)) //nolint:gosec
}

// Sampler draws one input element. OpInfo entries use samplers to steer
// inputs into the op's differentiable domain, exactly as PyTorch OpInfo
// entries constrain sample inputs (e.g. positive-only for log/sqrt, away
// from the kink at zero for relu-like ops, no ties for max-like reductions).
type Sampler func(r *rand.Rand) float64

// DomainDefault samples uniformly from +-[0.1, 2.0]. The exclusion zone
// around zero keeps central differences well-conditioned for ops with
// curvature or kinks near the origin.
func DomainDefault(r *rand.Rand) float64 {
	v := 0.1 + 1.9*r.Float64()
	if r.Float64() < 0.5 {
		v = -v
	}
	return v
}

// DomainPositive samples uniformly from [0.5, 2.5] for ops only defined (or
// only differentiable) on positive inputs: log, sqrt, rsqrt, div/pow bases.
func DomainPositive(r *rand.Rand) float64 {
	return 0.5 + 2.0*r.Float64()
}

// DomainAwayFromZero samples +-[0.5, 2.0], keeping a wide margin from the
// non-differentiable point at zero (relu/leaky-relu kink, abs). With
// eps-scaled steps of ~1e-6, a 0.5 margin guarantees both perturbed
// evaluations stay on the same side of the kink.
func DomainAwayFromZero(r *rand.Rand) float64 {
	v := 0.5 + 1.5*r.Float64()
	if r.Float64() < 0.5 {
		v = -v
	}
	return v
}

// OpInfo describes one op for the registry: how to build a fresh node, what
// input shapes to feed it, the sampling domain per input, and per-op
// tolerance overrides. Mirrors torch.testing._internal.opinfo.
type OpInfo struct {
	// Name is the registry key (unique).
	Name string
	// Make constructs a FRESH node instance on the given engine. Called once
	// per finite-difference evaluation; see MakeNodeFn for the determinism
	// contract.
	Make func(e compute.Engine[float64]) (graph.Node[float64], error)
	// InputShapes lists the shape of every Forward input, in order.
	InputShapes [][]int
	// Domains optionally overrides the input sampler. Length 0 applies
	// DomainDefault to all inputs; length 1 applies Domains[0] to all
	// inputs; otherwise it must match len(InputShapes).
	Domains []Sampler
	// Tol optionally overrides the default tolerances for this op.
	Tol *Tolerance
	// Eps optionally overrides the base finite-difference step.
	Eps float64
	// Seed seeds input generation and the upstream gradient (0 is valid).
	Seed int64
}

// SampleInputs generates the deterministic input tensors for this entry.
func (op *OpInfo) SampleInputs() ([]*tensor.TensorNumeric[float64], error) {
	if len(op.Domains) > 1 && len(op.Domains) != len(op.InputShapes) {
		return nil, fmt.Errorf("gradcheck: op %q has %d domains for %d inputs", op.Name, len(op.Domains), len(op.InputShapes))
	}
	r := newRand(op.Seed)
	inputs := make([]*tensor.TensorNumeric[float64], len(op.InputShapes))
	for i, shape := range op.InputShapes {
		sample := Sampler(DomainDefault)
		switch {
		case len(op.Domains) == 1:
			sample = op.Domains[0]
		case len(op.Domains) > 1:
			sample = op.Domains[i]
		}
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float64, size)
		for k := range data {
			data[k] = sample(r)
		}
		t, err := tensor.New[float64](shape, data)
		if err != nil {
			return nil, fmt.Errorf("gradcheck: building input %d for op %q: %w", i, op.Name, err)
		}
		inputs[i] = t
	}
	return inputs, nil
}

// Run executes gradcheck for this entry on a float64 CPU engine and returns
// the report. All checking is float64 on CPU by design (ADR 091): precision
// first; GPU parity is a separate harness.
func (op *OpInfo) Run(ctx context.Context) (*Report, error) {
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	inputs, err := op.SampleInputs()
	if err != nil {
		return nil, err
	}
	cfg := &Config{Seed: op.Seed, Eps: op.Eps}
	if op.Tol != nil {
		cfg.Tol = *op.Tol
	}
	return Check(ctx, func() (graph.Node[float64], error) { return op.Make(engine) }, inputs, cfg)
}

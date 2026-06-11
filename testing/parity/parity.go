// Package parity implements the engine-parity-under-arena-stress harness
// (zerfoo docs/adr/091 harness #2, plan T1.2, issue #128).
//
// The harness runs the SAME op set through two engines at float32 -- a
// reference (CPU) and a candidate (GPU on the GB10; an arena-stressed CPU
// engine in CI) -- forward AND backward, and diffs forward outputs, input
// gradients, and parameter gradients per op within per-op tolerances.
//
// Critically, ops run in an INTERLEAVED schedule:
//
//	opA.Forward, opB.Forward, ..., opZ.Forward,
//	[arena reset],
//	opZ.Backward, ..., opB.Backward, opA.Backward
//
// with a deliberately small arena on the candidate side, so buffers are
// reused between an op's forward and its backward. Single-op tests cannot
// see this schedule shape; it is exactly what exposed the LayerNorm /
// QK-norm cached-intermediate corruption class (zerfoo#842, zerfoo#845).
// Two schedule variants run:
//
//   - ScheduleNoReset: pure kernel parity -- no allocator interference.
//   - ScheduleResetBetween: the Wolf hazard -- the candidate's arena is
//     Reset between the forward and backward phases. Ops that honor the
//     save-for-backward contract (graph.SaverAware + Saver.SaveForBackward,
//     ztensor ADR 006) or recompute from live inputs must still be correct;
//     ops that cache a forward intermediate in a struct field without the
//     contract read recycled (and, under ZTENSOR_ARENA_POISON=1, NaN-
//     poisoned) memory and are flagged red, attributed to op + schedule.
//
// The harness acts as the graph executor: it wires a pinning Saver into
// every SaverAware node (mirroring graph.Builder), pins what nodes save,
// and releases each node's pins after its Backward returns (mirroring
// graph.releaseSaved).
//
// CI vs GB10 split: the runner, the diff logic, the host-backed-arena
// candidate (StressEngine), and the red-proof fixtures all run in ordinary
// CI with no GPU. The real CPU-vs-GPU comparison is a test gated on
// cuda.Available() plus a Spark pod for the GB10 (scripts/parity/).
//
// Reports serialize to JSON in the style of the PyTorch-oracle harness
// (testing/oracle, scripts/oracle/run_oracle.py): per-op max_abs/max_rel
// for forward and every gradient, plus passed/failed/errored totals.
package parity

import (
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/gradcheck"
	"github.com/zerfoo/ztensor/testing/oracle"
)

// t32 abbreviates the float32 tensor type used throughout the harness.
type t32 = tensor.TensorNumeric[float32]

// Op describes one operation under parity test: how to build a fresh node on
// a given float32 engine, the deterministic input recipe, and an optional
// tolerance override. The gradcheck OpInfo registry is the canonical source
// (RegistryOps); fixtures add entries with custom Make functions.
type Op struct {
	// Name identifies the op in reports.
	Name string
	// Make constructs a FRESH node on the given engine. Called once per
	// (side, schedule) run, so cached state never leaks across runs.
	Make func(e compute.Engine[float32]) (graph.Node[float32], error)
	// InputShapes lists the shape of every Forward input, in order.
	InputShapes [][]int
	// Domains optionally overrides the input sampler per input (gradcheck
	// semantics: empty applies the default to all inputs).
	Domains []gradcheck.Sampler
	// Seed seeds input and upstream-gradient generation. Both sides of a
	// comparison derive byte-identical f32 inputs from it.
	Seed int64
	// Tol optionally overrides ToleranceFor(Name).
	Tol *oracle.Tolerance
}

// sampleInputs generates the op's deterministic float32 inputs by reusing
// the gradcheck registry's float64 sampling and rounding to f32 -- the exact
// convention of the oracle generator, so the recorded f32 values are
// canonical and identical on both sides.
func (op Op) sampleInputs() ([]*t32, error) {
	info := gradcheck.OpInfo{
		Name:        op.Name,
		InputShapes: op.InputShapes,
		Domains:     op.Domains,
		Seed:        op.Seed,
	}
	in64, err := info.SampleInputs()
	if err != nil {
		return nil, err
	}
	out := make([]*t32, len(in64))
	for i, t := range in64 {
		data := make([]float32, len(t.Data()))
		for k, v := range t.Data() {
			data[k] = float32(v)
		}
		out[i], err = tensor.New[float32](t.Shape(), data)
		if err != nil {
			return nil, fmt.Errorf("parity: building input %d for op %q: %w", i, op.Name, err)
		}
	}
	return out, nil
}

// tolerance resolves the op's comparison tolerance.
func (op Op) tolerance() oracle.Tolerance {
	if op.Tol != nil {
		return *op.Tol
	}
	return ToleranceFor(op.Name)
}

// RegistryOps adapts the gradcheck OpInfo registry -- the single op
// inventory shared by all three ADR 091 harnesses -- to float32 parity ops.
// Node construction goes through gradcheck.NewRegistryNode, the single
// source of truth for constructor arguments.
func RegistryOps() []Op {
	infos := gradcheck.Registry()
	ops := make([]Op, 0, len(infos))
	for _, info := range infos {
		name := info.Name
		ops = append(ops, Op{
			Name: name,
			Make: func(e compute.Engine[float32]) (graph.Node[float32], error) {
				return gradcheck.NewRegistryNode[float32](name, e)
			},
			InputShapes: info.InputShapes,
			Domains:     info.Domains,
			Seed:        info.Seed,
		})
	}
	return ops
}

// DefaultTolerance is the f32 CPU-vs-GPU comparison bar for elementwise ops:
// ~1e-5 relative. Different-but-correct implementations of the same
// elementwise function agree to a few ULP at f32.
var DefaultTolerance = oracle.Tolerance{
	FwdAtol:  1e-6,
	FwdRtol:  1e-5,
	GradAtol: 1e-6,
	GradRtol: 1e-4,
}

// toleranceOverrides loosens ops whose CPU and GPU implementations
// legitimately reduce in different orders at f32 (cuBLAS tiling vs
// sequential CPU loops; parallel softmax/norm denominators). Expected
// reduction-order divergence is exactly why per-op tolerances exist --
// the gate is "within f32 reduction noise", not bitwise equality.
var toleranceOverrides = map[string]oracle.Tolerance{
	"MatMul":            {FwdAtol: 1e-6, FwdRtol: 1e-4, GradAtol: 1e-6, GradRtol: 1e-3},
	"Softmax":           {FwdAtol: 1e-6, FwdRtol: 1e-4, GradAtol: 1e-6, GradRtol: 1e-3},
	"LayerNorm":         {FwdAtol: 1e-5, FwdRtol: 1e-4, GradAtol: 1e-5, GradRtol: 1e-3},
	"ReduceSum":         {FwdAtol: 1e-6, FwdRtol: 1e-4, GradAtol: 1e-6, GradRtol: 1e-4},
	"ReduceMean":        {FwdAtol: 1e-6, FwdRtol: 1e-4, GradAtol: 1e-6, GradRtol: 1e-4},
	"HadamardTransform": {FwdAtol: 1e-6, FwdRtol: 1e-4, GradAtol: 1e-6, GradRtol: 1e-3},
}

// ToleranceFor returns the per-op tolerance, falling back to
// DefaultTolerance. Op.Tol overrides both.
func ToleranceFor(op string) oracle.Tolerance {
	if t, ok := toleranceOverrides[op]; ok {
		return t
	}
	return DefaultTolerance
}

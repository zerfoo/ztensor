package gradcheck

import (
	"context"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestRegistry runs gradcheck for every registered op: analytic Backward vs
// float64 central differences on the CPU engine. A wrong Jacobian in any
// registered op fails here as a named test instead of a NaN on the DGX.
func TestRegistry(t *testing.T) {
	ctx := context.Background()
	seen := map[string]bool{}
	for _, op := range Registry() {
		if seen[op.Name] {
			t.Fatalf("duplicate registry entry %q", op.Name)
		}
		seen[op.Name] = true
		t.Run(op.Name, func(t *testing.T) {
			report, err := op.Run(ctx)
			if err != nil {
				t.Fatalf("gradcheck %s: %v", op.Name, err)
			}
			if !report.OK() {
				t.Fatalf("%s", report)
			}
			if report.Checked == 0 {
				t.Fatalf("gradcheck %s compared zero elements", op.Name)
			}
		})
	}
}

// newBadTanhNode is the red-proof fixture: forward is a correct tanh, but
// Backward deliberately returns TWICE the true gradient. gradcheck MUST flag
// it -- this proves the harness catches the wrong-Jacobian bug class.
func newBadTanhNode(e engineT) *opNode {
	return unary("BadTanh",
		func(ctx context.Context, x t64) (t64, error) { return e.Tanh(ctx, x) },
		func(ctx context.Context, g, x, _ t64) (t64, error) {
			dx, err := e.TanhPrime(ctx, x, g)
			if err != nil {
				return nil, err
			}
			return e.MulScalar(ctx, dx, 2) // wrong: 2x the true Jacobian
		})
}

// TestRedProofWrongJacobianFails asserts that gradcheck FAILS the
// deliberately-wrong-Jacobian fixture. If this test fails, the checker has
// lost its teeth.
func TestRedProofWrongJacobianFails(t *testing.T) {
	op := OpInfo{
		Name: "BadTanh", Seed: 99,
		Make:        func(e engineT) (graph.Node[float64], error) { return newBadTanhNode(e), nil },
		InputShapes: [][]int{{2, 3}},
	}
	report, err := op.Run(context.Background())
	if err != nil {
		t.Fatalf("gradcheck mechanical failure: %v", err)
	}
	if report.OK() {
		t.Fatalf("red proof FAILED: gradcheck passed a node whose Backward returns 2x the true gradient")
	}
	if report.MismatchCount != report.Checked {
		t.Logf("note: %d/%d elements flagged (expected all for a 2x Jacobian)", report.MismatchCount, report.Checked)
	}
	if !strings.Contains(report.String(), "out of tolerance") {
		t.Fatalf("report should describe the failure, got: %s", report)
	}
}

// statefulNode caches its forward output and DESTROYS it at the end of every
// Forward call (worst-case statefulness). With fresh-node-per-evaluation the
// checker must still pass it; reusing one instance across finite-difference
// evaluations would corrupt the analytic Backward.
type statefulNode struct {
	*opNode
	calls int
}

func newStatefulSquare(e engineT) *statefulNode {
	s := &statefulNode{}
	s.opNode = unary("StatefulSquare",
		func(ctx context.Context, x t64) (t64, error) {
			s.calls++
			if s.calls > 1 {
				// A second Forward on the same instance poisons the cache the
				// way stale arena reuse did in the LayerNorm GPU bug.
				return e.MulScalar(ctx, x, 0)
			}
			return e.Mul(ctx, x, x)
		},
		func(ctx context.Context, g, x, _ t64) (t64, error) {
			xg, err := e.Mul(ctx, g, x)
			if err != nil {
				return nil, err
			}
			return e.MulScalar(ctx, xg, 2)
		})
	return s
}

// TestFreshNodePerEvaluation proves the checker never reuses a node instance
// across Forward evaluations: the fixture returns garbage on any second
// Forward of the same instance, and gradcheck must still pass.
func TestFreshNodePerEvaluation(t *testing.T) {
	op := OpInfo{
		Name: "StatefulSquare", Seed: 7,
		Make:        func(e engineT) (graph.Node[float64], error) { return newStatefulSquare(e), nil },
		InputShapes: [][]int{{2, 2}},
	}
	report, err := op.Run(context.Background())
	if err != nil {
		t.Fatalf("gradcheck mechanical failure: %v", err)
	}
	if !report.OK() {
		t.Fatalf("checker reused a node instance across evaluations:\n%s", report)
	}
}

// TestCheckExplicitUpstream covers the supplied-upstream path.
func TestCheckExplicitUpstream(t *testing.T) {
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	x, err := tensor.New[float64]([]int{2, 2}, []float64{0.3, -0.8, 1.2, -1.5})
	if err != nil {
		t.Fatal(err)
	}
	up, err := tensor.New[float64]([]int{2, 2}, []float64{1, 1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	report, err := Check(context.Background(),
		func() (graph.Node[float64], error) { return newTanhNode(engine), nil },
		[]*tensor.TensorNumeric[float64]{x},
		&Config{Upstream: up})
	if err != nil {
		t.Fatal(err)
	}
	if !report.OK() {
		t.Fatalf("tanh with all-ones upstream failed:\n%s", report)
	}
}

// TestParameterGradients exercises the Parameters() path directly: the
// layernorm wrapper's gamma/beta gradients must match finite differences.
func TestParameterGradients(t *testing.T) {
	op := OpInfo{
		Name: "LayerNormParams", Seed: 5,
		Make:        func(e engineT) (graph.Node[float64], error) { return newLayerNormNode(e, 3) },
		InputShapes: [][]int{{2, 3}},
	}
	report, err := op.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !report.OK() {
		t.Fatalf("layernorm parameter gradcheck failed:\n%s", report)
	}
	// 6 input elems + 3 gamma + 3 beta.
	if report.Checked != 12 {
		t.Fatalf("checked %d elements, want 12 (inputs + gamma + beta)", report.Checked)
	}
}

// TestConfigDefaults pins the documented defaults.
func TestConfigDefaults(t *testing.T) {
	c := (*Config)(nil).withDefaults()
	if c.Eps != DefaultEps || c.Tol.Atol != DefaultAtol || c.Tol.Rtol != DefaultRtol || c.MaxMismatches != DefaultMaxMismatches {
		t.Fatalf("unexpected defaults: %+v", c)
	}
}

package parity

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/types"
)

// Red-proof fixture (plan T1.2 acceptance; ADR 091 "each harness must
// encode at least one historically-fixed bug as a regression fixture").
//
// cachedCubeNode computes y = x^3 in two engine steps and CACHES the x^2
// intermediate in a struct field for Backward (dx = 3 * x^2 * g). With
// useContract=false this is byte-for-byte the pre-fix LayerNorm shape
// (zerfoo#842): a forward intermediate, allocated by the engine (so
// arena-backed on GPU and on the StressEngine), read by Backward through a
// struct field with no SaveForBackward registration. Under
// ScheduleResetBetween with poison the cached span is NaN-filled before
// Backward reads it, and the harness flags the op red. The
// useContract=true twin registers the same intermediate through the
// contract and stays green: the pin survives the reset.
type cachedCubeNode struct {
	graph.NoParameters[float32]
	engine      compute.Engine[float32]
	useContract bool
	saver       graph.Saver[float32]
	cached      *t32 // the x^2 intermediate: the deprecated raw cache
}

func (n *cachedCubeNode) OpType() string {
	if n.useContract {
		return FixtureContractCache
	}
	return FixtureRawCache
}
func (n *cachedCubeNode) Attributes() map[string]interface{} { return nil }
func (n *cachedCubeNode) SetSaver(s graph.Saver[float32])    { n.saver = s }
func (n *cachedCubeNode) OutputShape() []int {
	if n.cached == nil {
		return nil
	}
	return n.cached.Shape()
}

func (n *cachedCubeNode) Forward(ctx context.Context, inputs ...*t32) (*t32, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("%s: want 1 input, got %d", n.OpType(), len(inputs))
	}
	x := inputs[0]
	x2, err := n.engine.Mul(ctx, x, x)
	if err != nil {
		return nil, err
	}
	n.cached = x2
	if n.useContract && n.saver != nil {
		n.saver.SaveForBackward(x2)
	}
	return n.engine.Mul(ctx, x2, x)
}

func (n *cachedCubeNode) Backward(ctx context.Context, _ types.BackwardMode, g *t32, _ ...*t32) ([]*t32, error) {
	if n.cached == nil {
		return nil, errors.New(n.OpType() + ": Backward called before Forward")
	}
	// dx = g * 3 * x^2, read from the cached intermediate.
	three, err := n.engine.MulScalar(ctx, n.cached, 3)
	if err != nil {
		return nil, err
	}
	dx, err := n.engine.Mul(ctx, g, three)
	if err != nil {
		return nil, err
	}
	return []*t32{dx}, nil
}

var _ graph.SaverAware[float32] = (*cachedCubeNode)(nil)

// Fixture op names as they appear in reports.
const (
	// FixtureRawCache is the contract-VIOLATING twin: it must be flagged
	// red under ScheduleResetBetween with poison enabled.
	FixtureRawCache = "FixtureCachedIntermediateRaw"
	// FixtureContractCache is the contract-honoring twin: it must stay
	// green under every schedule.
	FixtureContractCache = "FixtureCachedIntermediateContract"
)

// FixtureOps returns the red-proof op pair. The raw-cache twin proves the
// harness CAN catch the cached-intermediate corruption class (assert red);
// the contract twin proves the save-for-backward contract is the fix
// (assert green).
func FixtureOps() []Op {
	return []Op{
		{
			Name: FixtureRawCache,
			Make: func(e compute.Engine[float32]) (graph.Node[float32], error) {
				return &cachedCubeNode{engine: e, useContract: false}, nil
			},
			InputShapes: [][]int{{2, 3}},
			Seed:        101,
		},
		{
			Name: FixtureContractCache,
			Make: func(e compute.Engine[float32]) (graph.Node[float32], error) {
				return &cachedCubeNode{engine: e, useContract: true}, nil
			},
			InputShapes: [][]int{{2, 3}},
			Seed:        102,
		},
	}
}

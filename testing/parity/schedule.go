package parity

import (
	"context"
	"fmt"
	"sync"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/oracle"
	"github.com/zerfoo/ztensor/types"
)

// Schedule selects the interleaved-run variant.
type Schedule string

const (
	// ScheduleNoReset runs all forwards then all backwards (reverse order)
	// with no allocator interference: pure kernel parity.
	ScheduleNoReset Schedule = "no-reset"
	// ScheduleResetBetween additionally invokes the candidate side's Reset
	// hook between the forward phase and the backward phase -- the Wolf
	// per-sample-ResetPool hazard. Ops honoring the save-for-backward
	// contract (or recomputing from live inputs) must still be correct;
	// raw struct-field caches of arena-backed intermediates are flagged.
	ScheduleResetBetween Schedule = "reset-between-fwd-bwd"
)

// Schedules lists every variant a full parity run covers.
func Schedules() []Schedule {
	return []Schedule{ScheduleNoReset, ScheduleResetBetween}
}

// Side is one engine under test.
type Side struct {
	// Name labels the side in reports ("cpu-f32", "gpu-f32", "cpu-f32+arena").
	Name string
	// Engine executes the ops.
	Engine compute.Engine[float32]
	// Reset, when non-nil, is called at the schedule's reset point
	// (ScheduleResetBetween only): the GPU engine's ResetPool, or a
	// host-backed test arena's Reset. Nil for plain CPU references.
	Reset func()
}

// pinningSaver implements graph.Saver[float32] for nodes orchestrated
// outside a Graph: it pins arena-backed storage on save (exactly like
// Graph.SaveForBackward) and releases every pin when the runner finishes
// the node's Backward (exactly like Graph.releaseSaved).
type pinningSaver struct {
	mu     sync.Mutex
	pinned []tensor.PinnableStorage
}

func (s *pinningSaver) SaveForBackward(ts ...*t32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, t := range ts {
		if t == nil {
			continue
		}
		if p, ok := t.GetStorage().(tensor.PinnableStorage); ok && p.PinForBackward() {
			s.pinned = append(s.pinned, p)
		}
	}
}

func (s *pinningSaver) release() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, p := range s.pinned {
		p.UnpinForBackward()
	}
	s.pinned = nil
}

var _ graph.Saver[float32] = (*pinningSaver)(nil)

// namedGrad is one parameter gradient snapshot.
type namedGrad struct {
	name string
	vals []float64
}

// sideRun holds one op's snapshots from one side. All tensor values are
// copied to float64 host slices IMMEDIATELY after the producing call, before
// any later op or reset can recycle the underlying buffers -- snapshots, not
// live views, are compared.
type sideRun struct {
	fwd        []float64
	inputGrads [][]float64
	paramGrads []namedGrad
	err        error // first Forward/Backward error, if any
}

// liveOp is one op's in-flight state during a schedule run.
type liveOp struct {
	op       Op
	node     graph.Node[float32]
	saver    *pinningSaver
	inputs   []*t32
	upstream *t32
	run      sideRun
}

// runSchedule executes ops on side under the given schedule:
//
//	phase F: op[0].Forward, op[1].Forward, ..., op[n-1].Forward
//	         (ScheduleResetBetween: side.Reset())
//	phase B: op[n-1].Backward, ..., op[1].Backward, op[0].Backward
//
// A node erroring in Forward skips its Backward; the error is recorded on
// that op's run and the schedule continues, so one broken op cannot mask
// parity results for the rest of the set.
func runSchedule(ctx context.Context, ops []Op, side Side, sched Schedule) ([]sideRun, error) {
	live := make([]*liveOp, len(ops))

	// Phase F: interleaved forwards.
	for i, op := range ops {
		lo := &liveOp{op: op}
		live[i] = lo

		node, err := op.Make(side.Engine)
		if err != nil {
			return nil, fmt.Errorf("parity: building op %q on side %q: %w", op.Name, side.Name, err)
		}
		lo.node = node
		// Wire the save-for-backward contract exactly as graph.Builder does.
		if sa, ok := node.(graph.SaverAware[float32]); ok {
			lo.saver = &pinningSaver{}
			sa.SetSaver(lo.saver)
		}

		inputs, err := op.sampleInputs()
		if err != nil {
			return nil, err
		}
		lo.inputs = inputs

		y, err := node.Forward(ctx, inputs...)
		if err != nil {
			lo.run.err = fmt.Errorf("Forward: %w", err)
			continue
		}
		lo.run.fwd = snapshot(y)
		lo.upstream, err = oracle.SeededUpstream(y.Shape(), op.Seed)
		if err != nil {
			return nil, fmt.Errorf("parity: upstream for op %q: %w", op.Name, err)
		}
	}

	// The schedule's reset point: the hazard under test.
	if sched == ScheduleResetBetween && side.Reset != nil {
		side.Reset()
	}

	// Phase B: backwards in reverse order. Live inputs are re-passed (the
	// graph memo convention); each node's saved set is released after its
	// Backward returns, success or error (graph.releaseSaved semantics).
	for i := len(live) - 1; i >= 0; i-- {
		lo := live[i]
		if lo.run.err != nil {
			continue
		}
		grads, err := lo.node.Backward(ctx, types.FullBackprop, lo.upstream, lo.inputs...)
		if err == nil && len(grads) != len(lo.inputs) {
			err = fmt.Errorf("returned %d gradients for %d inputs", len(grads), len(lo.inputs))
		}
		if err != nil {
			lo.run.err = fmt.Errorf("Backward: %w", err)
		} else {
			lo.run.inputGrads = make([][]float64, len(grads))
			for k, g := range grads {
				if g == nil {
					lo.run.err = fmt.Errorf("Backward: nil gradient for input %d", k)
					break
				}
				lo.run.inputGrads[k] = snapshot(g)
			}
			for _, p := range lo.node.Parameters() {
				if p.Gradient == nil {
					lo.run.err = fmt.Errorf("Backward: parameter %q has nil gradient", p.Name)
					break
				}
				lo.run.paramGrads = append(lo.run.paramGrads, namedGrad{name: p.Name, vals: snapshot(p.Gradient)})
			}
		}
		if lo.saver != nil {
			lo.saver.release()
		}
	}

	runs := make([]sideRun, len(live))
	for i, lo := range live {
		runs[i] = lo.run
	}
	return runs, nil
}

// snapshot copies a tensor's values to a float64 host slice. For GPU-backed
// tensors Data() performs the D2H copy; the returned slice is detached from
// any pool-managed buffer.
func snapshot(t *t32) []float64 {
	data := t.Data()
	out := make([]float64, len(data))
	for i, v := range data {
		out[i] = float64(v)
	}
	return out
}

// Run executes ops on the reference and the candidate side under one
// schedule and diffs them per op. The error return covers harness failures
// (an op that cannot even be constructed); per-op numerical failures and
// Forward/Backward errors are recorded in the report instead.
func Run(ctx context.Context, ops []Op, reference, candidate Side, sched Schedule) (*Report, error) {
	refRuns, err := runSchedule(ctx, ops, reference, sched)
	if err != nil {
		return nil, err
	}
	candRuns, err := runSchedule(ctx, ops, candidate, sched)
	if err != nil {
		return nil, err
	}

	rep := &Report{
		Reference: reference.Name,
		Candidate: candidate.Name,
		Schedule:  string(sched),
	}
	for i, op := range ops {
		rep.Results = append(rep.Results, compareOp(op, sched, refRuns[i], candRuns[i]))
	}
	for _, r := range rep.Results {
		switch {
		case r.Error != "":
			rep.Errored++
		case r.Pass:
			rep.Passed++
		default:
			rep.Failed++
		}
	}
	rep.Pass = rep.Failed == 0 && rep.Errored == 0
	return rep, nil
}

// compareOp diffs one op's two side runs within its tolerance.
func compareOp(op Op, sched Schedule, ref, cand sideRun) OpResult {
	res := OpResult{Op: op.Name, Schedule: string(sched), Tolerance: op.tolerance()}
	if ref.err != nil {
		res.Error = fmt.Sprintf("reference: %v", ref.err)
		return res
	}
	if cand.err != nil {
		res.Error = fmt.Sprintf("candidate: %v", cand.err)
		return res
	}
	tol := res.Tolerance

	if len(ref.fwd) != len(cand.fwd) {
		res.Error = fmt.Sprintf("forward size mismatch: reference %d vs candidate %d", len(ref.fwd), len(cand.fwd))
		return res
	}
	fwd := oracle.Diff(cand.fwd, ref.fwd, tol.FwdAtol, tol.FwdRtol)
	res.Forward = &fwd

	if len(ref.inputGrads) != len(cand.inputGrads) {
		res.Error = fmt.Sprintf("gradient count mismatch: reference %d vs candidate %d", len(ref.inputGrads), len(cand.inputGrads))
		return res
	}
	for k := range ref.inputGrads {
		if len(ref.inputGrads[k]) != len(cand.inputGrads[k]) {
			res.Error = fmt.Sprintf("grad_input_%d size mismatch: reference %d vs candidate %d",
				k, len(ref.inputGrads[k]), len(cand.inputGrads[k]))
			return res
		}
		res.InputGrads = append(res.InputGrads, oracle.Diff(cand.inputGrads[k], ref.inputGrads[k], tol.GradAtol, tol.GradRtol))
	}

	if len(ref.paramGrads) != len(cand.paramGrads) {
		res.Error = fmt.Sprintf("param gradient count mismatch: reference %d vs candidate %d",
			len(ref.paramGrads), len(cand.paramGrads))
		return res
	}
	for k := range ref.paramGrads {
		r, c := ref.paramGrads[k], cand.paramGrads[k]
		if r.name != c.name || len(r.vals) != len(c.vals) {
			res.Error = fmt.Sprintf("param gradient mismatch at %d: reference %s[%d] vs candidate %s[%d]",
				k, r.name, len(r.vals), c.name, len(c.vals))
			return res
		}
		if res.ParamGrads == nil {
			res.ParamGrads = map[string]oracle.DiffStats{}
		}
		res.ParamGrads[r.name] = oracle.Diff(c.vals, r.vals, tol.GradAtol, tol.GradRtol)
	}

	res.Pass = fwd.Pass
	for _, d := range res.InputGrads {
		res.Pass = res.Pass && d.Pass
	}
	for _, d := range res.ParamGrads {
		res.Pass = res.Pass && d.Pass
	}
	return res
}

package parity

import (
	"context"
	"encoding/json"
	"math"
	"path/filepath"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/testing/gradcheck"
)

func cpuSide(name string) Side {
	return Side{Name: name, Engine: compute.NewCPUEngine[float32](numeric.Float32Ops{})}
}

// enablePoison flips ZTENSOR_ARENA_POISON semantics on for the test and
// installs the host-memory poison fill (the host-backed arenas here are not
// CUDA allocations), mirroring the WolfHazard tests in graph/.
func enablePoison(t *testing.T) {
	t.Helper()
	restore := cuda.SetArenaPoisonEnabledForTesting(true)
	t.Cleanup(restore)
	cuda.SetArenaPoisonFill(cuda.HostPoisonFillForTesting)
	t.Cleanup(func() { cuda.SetArenaPoisonFill(nil) })
}

// stressSide builds the CI candidate: a CPU engine whose op results are
// relocated into a host-backed arena with GPU lifetime semantics.
func stressSide(t *testing.T) (Side, *StressEngine) {
	t.Helper()
	eng := NewStressEngine(compute.NewCPUEngine[float32](numeric.Float32Ops{}), 1<<20)
	return Side{Name: "cpu-f32+host-arena", Engine: eng, Reset: eng.ResetArena}, eng
}

// TestRegistryOps_MirrorsGradcheckRegistry keeps the parity op inventory in
// lockstep with the gradcheck OpInfo registry (the shared ADR 091 op list).
func TestRegistryOps_MirrorsGradcheckRegistry(t *testing.T) {
	infos := gradcheck.Registry()
	ops := RegistryOps()
	if len(ops) != len(infos) {
		t.Fatalf("RegistryOps has %d entries, gradcheck.Registry has %d", len(ops), len(infos))
	}
	for i := range infos {
		if ops[i].Name != infos[i].Name {
			t.Fatalf("op %d: parity %q vs gradcheck %q", i, ops[i].Name, infos[i].Name)
		}
	}
}

// TestRun_CPUvsCPU_ExactParity: two plain CPU engines must agree exactly
// under both schedules (the reset hook is nil on both sides) -- this proves
// the runner's determinism (inputs, upstreams, interleaving) and the diff
// plumbing at zero tolerance margin.
func TestRun_CPUvsCPU_ExactParity(t *testing.T) {
	for _, sched := range Schedules() {
		rep, err := Run(context.Background(), RegistryOps(), cpuSide("cpu-ref"), cpuSide("cpu-cand"), sched)
		if err != nil {
			t.Fatalf("schedule %s: Run: %v", sched, err)
		}
		if !rep.Pass {
			t.Fatalf("schedule %s: report failed:\n%s", sched, dumpFailures(t, rep))
		}
		for _, r := range rep.Results {
			if r.Forward.MaxAbs != 0 {
				t.Errorf("schedule %s: op %s forward max_abs = %g, want exact 0 for identical engines",
					sched, r.Op, r.Forward.MaxAbs)
			}
			for k, d := range r.InputGrads {
				if d.MaxAbs != 0 {
					t.Errorf("schedule %s: op %s grad_input_%d max_abs = %g, want exact 0",
						sched, r.Op, k, d.MaxAbs)
				}
			}
		}
	}
}

// TestRun_HostArenaStress_RegistryGreen is the CI lifetime gate: the
// registry ops -- which honor the save-for-backward contract (gradcheck
// opNode saves its cached output; layerNormNode saves xhat+inv) or
// recompute from live inputs -- must survive the reset-between-fwd-and-bwd
// schedule against a poisoned host-backed arena. A node that regresses to a
// raw struct-field cache turns this red (see the fixture test).
func TestRun_HostArenaStress_RegistryGreen(t *testing.T) {
	enablePoison(t)
	for _, sched := range Schedules() {
		side, eng := stressSide(t)
		rep, err := Run(context.Background(), RegistryOps(), cpuSide("cpu-f32"), side, sched)
		if err != nil {
			t.Fatalf("schedule %s: Run: %v", sched, err)
		}
		if !rep.Pass {
			t.Fatalf("schedule %s: report failed:\n%s", sched, dumpFailures(t, rep))
		}
		// Diffs must also be exact: the stress engine delegates to the same
		// CPU kernels, so any nonzero diff means a lifetime bug, not noise.
		for _, r := range rep.Results {
			if r.Forward.MaxAbs != 0 {
				t.Errorf("schedule %s: op %s forward max_abs = %g, want 0", sched, r.Op, r.Forward.MaxAbs)
			}
		}
		// The contract genuinely engaged: nodes saved arena-backed
		// intermediates and the runner pinned them...
		if got := eng.Arena().PinnedHighWaterBytes(); got == 0 {
			t.Errorf("schedule %s: PinnedHighWaterBytes = 0, want > 0 (save-for-backward never pinned)", sched)
		}
		// ...and released every pin after each node's Backward
		// (graph.releaseSaved semantics).
		if got := eng.Arena().PinnedBytes(); got != 0 {
			t.Errorf("schedule %s: PinnedBytes after run = %d, want 0", sched, got)
		}
	}
}

// TestRun_RedProof_RawCacheFlagged is the mandatory red-proof: an op that
// caches a forward intermediate in a struct field WITHOUT SaveForBackward
// (the pre-fix LayerNorm shape, zerfoo#842) MUST be flagged under
// ScheduleResetBetween with poison -- a NaN diff attributed to that op and
// schedule -- while its contract-honoring twin passes green, and BOTH pass
// under ScheduleNoReset (single-op-style runs cannot see the bug).
func TestRun_RedProof_RawCacheFlagged(t *testing.T) {
	enablePoison(t)

	// Schedule (a): no reset -- the raw cache is never recycled, both green.
	side, _ := stressSide(t)
	rep, err := Run(context.Background(), FixtureOps(), cpuSide("cpu-f32"), side, ScheduleNoReset)
	if err != nil {
		t.Fatalf("no-reset: Run: %v", err)
	}
	if !rep.Pass {
		t.Fatalf("no-reset: fixtures must both pass (the bug is invisible without the reset schedule):\n%s",
			dumpFailures(t, rep))
	}

	// Schedule (b): the Wolf hazard -- raw cache red, contract twin green.
	side, _ = stressSide(t)
	rep, err = Run(context.Background(), FixtureOps(), cpuSide("cpu-f32"), side, ScheduleResetBetween)
	if err != nil {
		t.Fatalf("reset-between: Run: %v", err)
	}
	if rep.Pass {
		t.Fatal("reset-between: report passed, want the raw-cache fixture flagged red")
	}

	raw := rep.Result(FixtureRawCache)
	if raw == nil {
		t.Fatal("no result for the raw-cache fixture")
	}
	if raw.Pass {
		t.Fatal("raw-cache fixture passed under reset-between-fwd-bwd, want red")
	}
	if raw.Schedule != string(ScheduleResetBetween) {
		t.Errorf("raw fixture attributed to schedule %q, want %q", raw.Schedule, ScheduleResetBetween)
	}
	if raw.Error != "" {
		t.Fatalf("raw fixture errored (%s), want a NaN gradient diff", raw.Error)
	}
	// The forward outputs were snapshotted before the reset: forward parity
	// holds; the corruption is in the backward read of the poisoned cache.
	if !raw.Forward.Pass {
		t.Errorf("raw fixture forward diff failed (max_abs=%g), corruption should be backward-only", raw.Forward.MaxAbs)
	}
	if len(raw.InputGrads) != 1 || raw.InputGrads[0].Pass {
		t.Fatalf("raw fixture input gradient not flagged: %+v", raw.InputGrads)
	}
	if !math.IsInf(raw.InputGrads[0].MaxAbs, 1) {
		t.Errorf("raw fixture grad max_abs = %g, want +Inf (NaN poison sentinel detected)", raw.InputGrads[0].MaxAbs)
	}

	good := rep.Result(FixtureContractCache)
	if good == nil || !good.Pass {
		t.Fatalf("contract-honoring twin must stay green under reset-between-fwd-bwd: %+v", good)
	}
	if rep.Failed != 1 || rep.Passed != 1 {
		t.Errorf("totals: passed=%d failed=%d errored=%d, want 1/1/0", rep.Passed, rep.Failed, rep.Errored)
	}
}

// saverStripped hides a node's SaverAware implementation from the runner:
// the embedded interface promotes graph.Node methods only, so the runner's
// type assertion fails and no Saver is wired -- simulating a registry op
// that never adopted the save-for-backward contract.
type saverStripped struct {
	graph.Node[float32]
}

// TestRun_RedProof_RealOpWithoutContractFlagged: the harness's sensitivity
// is not an artifact of the synthetic fixture. A REAL registry op (Exp,
// whose Backward consumes its cached forward output) with the contract
// stripped is flagged under the reset schedule exactly like the pre-fix
// LayerNorm would have been.
func TestRun_RedProof_RealOpWithoutContractFlagged(t *testing.T) {
	enablePoison(t)
	op := Op{
		Name: "Exp-no-contract",
		Make: func(e compute.Engine[float32]) (graph.Node[float32], error) {
			n, err := gradcheck.NewRegistryNode[float32]("Exp", e)
			if err != nil {
				return nil, err
			}
			return &saverStripped{Node: n}, nil
		},
		InputShapes: [][]int{{2, 3}},
		Seed:        10,
	}
	side, _ := stressSide(t)
	rep, err := Run(context.Background(), []Op{op}, cpuSide("cpu-f32"), side, ScheduleResetBetween)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	r := rep.Result("Exp-no-contract")
	if r == nil || r.Pass || r.Error != "" {
		t.Fatalf("contract-stripped Exp must fail with a gradient diff, got %+v", r)
	}
	if len(r.InputGrads) != 1 || !math.IsInf(r.InputGrads[0].MaxAbs, 1) {
		t.Fatalf("want +Inf grad max_abs from the poisoned cached output, got %+v", r.InputGrads)
	}
}

// TestReport_JSONRoundTrip pins the report wire format (the devlog artifact,
// oracle-report style).
func TestReport_JSONRoundTrip(t *testing.T) {
	enablePoison(t)
	side, _ := stressSide(t)
	rep, err := Run(context.Background(), append(RegistryOps(), FixtureOps()...),
		cpuSide("cpu-f32"), side, ScheduleResetBetween)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	path := filepath.Join(t.TempDir(), "report.json")
	if err := rep.WriteJSON(path); err != nil {
		t.Fatalf("WriteJSON: %v", err)
	}
	var back Report
	raw, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Schedule != string(ScheduleResetBetween) || back.Candidate != side.Name {
		t.Fatalf("round trip lost run identity: %+v", back)
	}
	if len(back.Results) != len(rep.Results) {
		t.Fatalf("round trip lost results: %d vs %d", len(back.Results), len(rep.Results))
	}
	if back.Passed+back.Failed+back.Errored != len(back.Results) {
		t.Fatalf("totals %d+%d+%d do not cover %d results",
			back.Passed, back.Failed, back.Errored, len(back.Results))
	}
	// The raw-cache fixture's Inf max_abs must survive JSON (encoded by the
	// DiffStats contract as a float64; spot-check it unmarshals as failing).
	if r := back.Result(FixtureRawCache); r == nil || r.Pass {
		t.Fatalf("raw fixture verdict lost in round trip: %+v", r)
	}
}

func dumpFailures(t *testing.T, rep *Report) string {
	t.Helper()
	out := ""
	for _, r := range rep.Results {
		if !r.Pass {
			out += r.String() + "\n"
		}
	}
	return out
}

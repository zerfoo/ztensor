package parity

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
)

// gpuParityArenaBytes is the deliberately SMALL GPU arena for the stress
// schedules: tiny enough that the backward phase reuses the exact region the
// forward phase wrote (after the reset rewinds the bump offset), far below
// the production minimum of 1 GB. The SetArenaBytesForTesting hook exists
// precisely because ZERFOO_ARENA_SIZE_GB cannot express this.
const gpuParityArenaBytes = 64 << 20

// TestParity_GPUvsCPU_ArenaStressSchedules_GPU is the real CPU-f32 vs
// GPU-f32 comparison (plan T1.2 acceptance). It skips cleanly when CUDA is
// unavailable (ordinary CI); on the GB10 it runs the full gradcheck registry
// under both schedules with poison-on-reset enabled and a small arena, and
// writes one oracle-style JSON report per schedule.
//
// Run on the DGX via the Spark pod in scripts/parity/ -- never interactive
// ssh. Set ZTENSOR_PARITY_REPORT_DIR to keep the JSON artifacts.
func TestParity_GPUvsCPU_ArenaStressSchedules_GPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	// Poison-on-reset makes any use of a recycled arena buffer a
	// deterministic NaN at the corruption site instead of a silent stale
	// read; enable before engine construction so the engine logs it and the
	// device fill kernel path is exercised.
	restorePoison := cuda.SetArenaPoisonEnabledForTesting(true)
	defer restorePoison()
	restoreArena := compute.SetArenaBytesForTesting(gpuParityArenaBytes)
	defer restoreArena()

	gpuEng, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	ref := cpuSide("cpu-f32")
	cand := Side{Name: "gpu-f32", Engine: gpuEng, Reset: gpuEng.ResetPool}

	reportDir := os.Getenv("ZTENSOR_PARITY_REPORT_DIR")
	if reportDir == "" {
		reportDir = t.TempDir()
	}

	for _, sched := range Schedules() {
		t.Run(string(sched), func(t *testing.T) {
			rep, err := Run(context.Background(), RegistryOps(), ref, cand, sched)
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
			for _, r := range rep.Results {
				t.Log(r.String())
			}
			path := filepath.Join(reportDir, fmt.Sprintf("parity-gpu-%s.json", sched))
			if err := rep.WriteJSON(path); err != nil {
				t.Errorf("WriteJSON: %v", err)
			} else {
				t.Logf("report: %s", path)
			}
			if !rep.Pass {
				t.Errorf("schedule %s: %d failed, %d errored of %d ops",
					sched, rep.Failed, rep.Errored, len(rep.Results))
			}
		})
	}
}

// TestParity_GPURedProof_GPU re-runs the red-proof pair against the REAL
// CUDA arena: the raw-cache fixture must be flagged on actual GPU memory
// (proving the GB10 run has the same sensitivity as CI), the contract twin
// must pass. Skips without CUDA.
func TestParity_GPURedProof_GPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	restorePoison := cuda.SetArenaPoisonEnabledForTesting(true)
	defer restorePoison()
	restoreArena := compute.SetArenaBytesForTesting(gpuParityArenaBytes)
	defer restoreArena()

	gpuEng, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	cand := Side{Name: "gpu-f32", Engine: gpuEng, Reset: gpuEng.ResetPool}
	rep, err := Run(context.Background(), FixtureOps(), cpuSide("cpu-f32"), cand, ScheduleResetBetween)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	for _, r := range rep.Results {
		t.Log(r.String())
	}
	raw := rep.Result(FixtureRawCache)
	if raw == nil || raw.Pass {
		t.Errorf("raw-cache fixture must be flagged red on the CUDA arena, got %+v", raw)
	}
	good := rep.Result(FixtureContractCache)
	if good == nil || !good.Pass {
		t.Errorf("contract twin must pass on the CUDA arena, got %+v", good)
	}
}

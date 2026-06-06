package compute

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestWedge106Context reproduces the PRODUCTION CONTEXT around the #106 wedge,
// not just the bare 213k upload (which 3 fresh-engine variants proved does NOT
// wedge). It mirrors the CrossAsset GPU sequence:
//
//	PHASE1  UploadWeights(model weights)        -- gpu_train.go:58
//	PHASE2  UploadWeights(213k samples + params)-- crossasset.go:529 (the T4.2 upload)
//	PHASE3  first-compute MatMuls               -- the epoch loop's first GPU ops
//
// The T4.2 print in the production code sits BEFORE the upload call, so the
// observed hang ("never returns") could be in PHASE2 OR in PHASE3 (first kernel
// launch / capture) -- this test exercises both. Phase markers are fsync'd to
// WEDGE_LOG; the wrapping pod encodes the outcome in its exit code and a true
// wedge leaves the pod stuck (the last logged phase localizes it).
//
// OFF by default. Run on a sacrificial GB10 pod:
//
//	ZTENSOR_WEDGE_REPRO=1 WEDGE_LOG=/p/ctx.log go test ./compute/ -run TestWedge106Context -v -timeout 0
//
// Env knobs: WEDGE_N (samples, default 213304), WEDGE_FPS (features, 193),
// WEDGE_DM (model dim, 64), WEDGE_NSRC (sources, 12), WEDGE_NLAYERS (4),
// WEDGE_COMPUTE_ITERS (first-compute MatMuls, 200).
func TestWedge106Context(t *testing.T) {
	if os.Getenv("ZTENSOR_WEDGE_REPRO") == "" {
		t.Skip("set ZTENSOR_WEDGE_REPRO=1 to run the #106 context repro (deliberately tries to wedge the GB10 driver)")
	}
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	logPath := os.Getenv("WEDGE_LOG")
	logf := func(format string, a ...any) {
		msg := fmt.Sprintf(format, a...)
		t.Log(msg)
		if logPath == "" {
			return
		}
		f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return
		}
		fmt.Fprintf(f, "%s %s\n", time.Now().UTC().Format(time.RFC3339Nano), msg)
		_ = f.Sync()
		_ = f.Close()
	}
	envInt := func(k string, def int) int {
		if v := os.Getenv(k); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > 0 {
				return n
			}
		}
		return def
	}

	nSamples := envInt("WEDGE_N", 213304)
	fps := envInt("WEDGE_FPS", 193)
	dm := envInt("WEDGE_DM", 64)
	nSrc := envInt("WEDGE_NSRC", 12)
	nLayers := envInt("WEDGE_NLAYERS", 4)
	computeIters := envInt("WEDGE_COMPUTE_ITERS", 200)
	ffnDim := dm * 4
	ctx := context.Background()

	ones := func(n int) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = 0.01
		}
		return d
	}
	mk := func(shape ...int) *tensor.TensorNumeric[float32] {
		n := 1
		for _, s := range shape {
			n *= s
		}
		tt, err := tensor.New[float32](shape, ones(n))
		if err != nil {
			t.Fatalf("tensor.New(%v): %v", shape, err)
		}
		return tt
	}

	ops := numeric.Float32Ops{}
	eng, err := NewGPUEngine[float32](ops)
	if err != nil {
		logf("NewGPUEngine err: %v", err)
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()
	logf("engine ready managedMem=%v arenaUsed=%d", eng.IsManagedMemory(), eng.ArenaUsedBytes())

	// PHASE1: model weights (input projections + layers + head), CrossAsset shapes.
	logf("PHASE1 upload-weights BEGIN nSrc=%d nLayers=%d dm=%d fps=%d", nSrc, nLayers, dm, fps)
	var inputW0 *tensor.TensorNumeric[float32]
	weights := make([]*tensor.TensorNumeric[float32], 0, nSrc*2+nLayers*12+2)
	for s := 0; s < nSrc; s++ {
		w := mk(fps, dm)
		if s == 0 {
			inputW0 = w
		}
		weights = append(weights, w, mk(dm))
	}
	for l := 0; l < nLayers; l++ {
		weights = append(weights,
			mk(dm, dm), mk(dm, dm), mk(dm, dm), mk(dm, dm), // qW kW vW outW
			mk(dm), mk(dm), // lnGamma lnBeta
			mk(dm, ffnDim), mk(ffnDim), mk(ffnDim, dm), mk(dm), // ffnW1 ffnB1 ffnW2 ffnB2
			mk(dm), mk(dm)) // ffnGamma ffnBeta
	}
	weights = append(weights, mk(dm, 3), mk(3)) // head
	if err := eng.UploadWeights(weights); err != nil {
		logf("PHASE1 UploadWeights err: %v", err)
		t.Fatalf("PHASE1 UploadWeights: %v", err)
	}
	logf("PHASE1 upload-weights DONE (%d weights) arenaUsed=%d", len(weights), eng.ArenaUsedBytes())

	// PHASE2: the T4.2 upload -- 213k sample tensors [1,fps] + ~50 params.
	logf("PHASE2 upload-samples BEGIN n=%d fps=%d", nSamples, fps)
	samples := make([]*tensor.TensorNumeric[float32], 0, nSamples+50)
	for i := 0; i < nSamples; i++ {
		d := make([]float32, fps)
		d[0] = float32(i)
		tt, terr := tensor.New[float32]([]int{1, fps}, d)
		if terr != nil {
			t.Fatalf("sample %d: %v", i, terr)
		}
		samples = append(samples, tt)
	}
	for p := 0; p < 50; p++ {
		samples = append(samples, mk(dm, dm))
	}
	logf("PHASE2 samples built (%d); UploadWeights BEGIN", len(samples))
	start := time.Now()
	if err := eng.UploadWeights(samples); err != nil {
		logf("PHASE2 UploadWeights err after %s: %v", time.Since(start), err)
		t.Fatalf("PHASE2 UploadWeights: %v", err)
	}
	logf("PHASE2 upload-samples DONE after %s; bulkBuffers=%d arenaUsed=%d",
		time.Since(start), len(eng.bulkUploadBuffers), eng.ArenaUsedBytes())

	// PHASE3: first-compute. Run MatMuls using the uploaded weights to trigger
	// arena allocation / kernel launches / any capture path -- the epoch loop's
	// first GPU work, which is the other candidate hang site.
	logf("PHASE3 compute BEGIN iters=%d", computeIters)
	x := samples[0] // [1,fps] (now GPU-resident)
	for k := 0; k < computeIters; k++ {
		out, merr := eng.MatMul(ctx, x, inputW0) // [1,fps] x [fps,dm] = [1,dm]
		if merr != nil {
			logf("PHASE3 MatMul[%d] err: %v", k, merr)
			t.Fatalf("PHASE3 MatMul: %v", merr)
		}
		_ = out
		if k%50 == 0 {
			logf("PHASE3 compute iter=%d arenaUsed=%d", k, eng.ArenaUsedBytes())
		}
	}
	logf("PHASE3 compute DONE")
	logf("ALL PHASES COMPLETE (NO WEDGE)")
}

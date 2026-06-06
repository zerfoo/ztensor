package compute

import (
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestWedge106Repro is the pure-ztensor reproduction for zerfoo/ztensor#106:
// uploading a CrossAsset-scale weight set (default 213,304 float32 tensors) via
// GPUEngine.UploadWeights wedges the GB10 (sm_121) CUDA driver in an
// uninterruptible D-state even with the v1.8.1 chunking in place.
//
// It is OFF by default (it deliberately tries to wedge the driver, which can
// leave an unkillable container). Run only on a sacrificial GB10 pod with:
//
//	ZTENSOR_WEDGE_REPRO=1 WEDGE_LOG=/path/on/hostpath/repro.log \
//	  go test ./compute/ -run TestWedge106Repro -v -timeout 0
//
// Tunables (env): WEDGE_N (tensor count), WEDGE_ELEMS (float32s per tensor).
//
// Phase markers are appended to WEDGE_LOG and fsync'd after every write, so when
// the process wedges the file's last line names the exact phase reached (e.g.
// "UploadWeights BEGIN" with no following "RETURNED" => wedge inside the
// upload). The out-of-band dstate-watchdog.sh captures the kernel frame.
func TestWedge106Repro(t *testing.T) {
	if os.Getenv("ZTENSOR_WEDGE_REPRO") == "" {
		t.Skip("set ZTENSOR_WEDGE_REPRO=1 to run the #106 wedge repro (deliberately wedges the GB10 driver)")
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

	nTensors := 213304
	if v := os.Getenv("WEDGE_N"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			nTensors = n
		}
	}
	elemsPer := 2048 // 8 KiB/tensor; 213304 x 8 KiB ~= 1.7 GiB total (multi-GB, matches the repro scale)
	if v := os.Getenv("WEDGE_ELEMS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			elemsPer = n
		}
	}

	logf("repro start pid=%d: building %d f32 tensors x %d elems (~%.2f GiB)",
		os.Getpid(), nTensors, elemsPer, float64(nTensors)*float64(elemsPer)*4/(1<<30))

	tensors := make([]*tensor.TensorNumeric[float32], nTensors)
	for i := range tensors {
		data := make([]float32, elemsPer)
		data[0] = float32(i) // unique sentinel; defeats any zero-page dedup
		tt, err := tensor.New[float32]([]int{elemsPer}, data)
		if err != nil {
			logf("tensor.New[%d] err: %v", i, err)
			t.Fatalf("tensor.New: %v", err)
		}
		tensors[i] = tt
	}
	logf("tensors built; constructing engine")

	ops := numeric.Float32Ops{}
	eng, err := NewGPUEngine[float32](ops)
	if err != nil {
		logf("NewGPUEngine err: %v", err)
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	logf("engine ready managedMem=%v chunkBytes=%d chunkTensors=%d; UploadWeights BEGIN",
		eng.IsManagedMemory(), bulkUploadF32MaxChunkBytes, bulkUploadF32MaxChunkTensors)
	start := time.Now()
	if err := eng.UploadWeights(tensors); err != nil {
		logf("UploadWeights err after %s: %v", time.Since(start), err)
		t.Fatalf("UploadWeights: %v", err)
	}
	logf("UploadWeights RETURNED ok after %s (NO WEDGE); bulkBuffers=%d",
		time.Since(start), len(eng.bulkUploadBuffers))
}

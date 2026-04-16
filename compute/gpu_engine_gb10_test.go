//go:build dgxgb10

package compute

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/zerfoo/ztensor/tensor"
)

// TestCUDAGraph_MultiTensorUpload_GB10 reproduces the GB10 hang where a
// capture region starts, an allocation-during-capture happens, and
// StreamEndCapture deadlocks. It is gated by //go:build dgxgb10 so it
// only runs on the DGX Spark host; the DGX runner is expected to pass
// -tags dgxgb10.
//
// The test accepts three outcomes so pre-fix and post-fix states are
// both observable:
//
//  1. EndCapture returns a valid graph: E2 fix is in place. The test
//     passes.
//  2. BeginCapture or EndCapture returns ErrCaptureIncompatibleAllocation
//     (or any wrapping of it): the probe from T1.2 caught the unsafe
//     allocation synchronously. The test records this and passes.
//  3. The capture body does not complete inside a 30 second timeout:
//     the hang is still present. The test calls t.Fatal. This is the
//     signal that the fix regressed (or is not yet in place).
//
// Hangs manifest as a deadlock inside StreamEndCapture on GB10 with
// allocations issued during capture, so the 30s guard is the only
// reliable way to surface the bug without hanging the whole test
// binary.
func TestCUDAGraph_MultiTensorUpload_GB10(t *testing.T) {
	eng := newTestGPUEngine(t)

	uploadTensors := buildGB10StressTensors(t)
	if err := eng.UploadWeights(uploadTensors); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	// Pair of tensors used inside the capture region for MatMul.
	// 256x1024 * 1024x256 matches a tensor uploaded above and exercises
	// the dense float32 kernel that triggers the hang on GB10.
	aData := make([]float32, 256*1024)
	for i := range aData {
		aData[i] = float32(i%7) * 0.125
	}
	bData := make([]float32, 1024*256)
	for i := range bData {
		bData[i] = float32(i%5) * 0.0625
	}
	a, err := tensor.New[float32]([]int{256, 1024}, aData)
	if err != nil {
		t.Fatalf("tensor.New A: %v", err)
	}
	b, err := tensor.New[float32]([]int{1024, 256}, bData)
	if err != nil {
		t.Fatalf("tensor.New B: %v", err)
	}
	if err := eng.UploadWeights([]*tensor.TensorNumeric[float32]{a, b}); err != nil {
		t.Fatalf("UploadWeights(matmul operands): %v", err)
	}

	// 30 second watchdog: if the capture lifecycle does not complete,
	// the goroutine is leaked but the test fails the offending run so
	// the CI job surfaces the bug instead of spinning forever.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	type captureResult struct {
		handle GraphHandle
		err    error
		// phase identifies where the failure originated so the log
		// distinguishes BeginCapture errors (T1.2 probe) from
		// EndCapture errors (post-fix graph instantiation failures).
		phase string
	}
	done := make(chan captureResult, 1)

	go func() {
		if err := eng.BeginCapture(); err != nil {
			done <- captureResult{err: err, phase: "BeginCapture"}
			return
		}
		// Run a MatMul inside the capture region. On the pre-fix path
		// this is the op whose cudaMallocAsync call deadlocks
		// StreamEndCapture downstream.
		if _, err := eng.MatMul(context.Background(), a, b); err != nil {
			// If MatMul itself fails synchronously we still need to
			// clean up the capture state before surfacing the error.
			_, endErr := eng.EndCapture()
			if endErr != nil {
				err = fmt.Errorf("%w (EndCapture cleanup: %v)", err, endErr)
			}
			done <- captureResult{err: err, phase: "MatMul"}
			return
		}
		handle, err := eng.EndCapture()
		done <- captureResult{handle: handle, err: err, phase: "EndCapture"}
	}()

	select {
	case <-ctx.Done():
		t.Fatal("hang detected -- capture lifecycle did not complete within 30s")
	case res := <-done:
		// Ensure any captured graph is released even if the test fails
		// later in its assertions.
		t.Cleanup(func() {
			if res.err == nil {
				_ = eng.DestroyGraph(res.handle)
			}
		})

		if res.err == nil {
			t.Logf("capture completed cleanly in phase=%s; fix is in place", res.phase)
			return
		}
		if errors.Is(res.err, ErrCaptureIncompatibleAllocation) {
			t.Logf("observed ErrCaptureIncompatibleAllocation in phase=%s (expected pre-fix outcome): %v", res.phase, res.err)
			return
		}
		t.Fatalf("unexpected capture error in phase=%s: %v", res.phase, res.err)
	}
}

// buildGB10StressTensors constructs >=50 float32 tensors spanning a mix
// of shapes that matches the production upload pattern that triggers the
// hang: several row-major matrices, a 256x1024 dense matrix, and a
// handful of long 1-D vectors. Each tensor is populated with a cheap
// deterministic pattern so MatMul inside the capture region produces
// non-zero work.
func buildGB10StressTensors(t *testing.T) []*tensor.TensorNumeric[float32] {
	t.Helper()

	// 50 varied tensors. The 256x1024 matrix is mandatory because it is
	// the shape that reproduces on GB10; the remainder is spread across
	// smaller shapes to force the allocator to touch multiple size
	// buckets in the pool.
	shapes := [][]int{
		{256, 1024},
		{64, 64}, {64, 64}, {64, 64}, {64, 64},
		{128, 256}, {128, 256}, {128, 256}, {128, 256},
		{1024},
		{512, 128}, {512, 128},
		{32, 32}, {32, 32}, {32, 32}, {32, 32}, {32, 32},
		{256}, {256}, {256},
		{128, 128}, {128, 128}, {128, 128}, {128, 128},
		{16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 16},
		{512},
		{64, 128}, {64, 128},
		{8, 8}, {8, 8}, {8, 8}, {8, 8}, {8, 8}, {8, 8}, {8, 8}, {8, 8},
		{2048},
		{96, 96}, {96, 96}, {96, 96},
		{4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 4},
		{1024, 64},
	}
	if len(shapes) < 50 {
		t.Fatalf("shape list too short: %d", len(shapes))
	}

	out := make([]*tensor.TensorNumeric[float32], 0, len(shapes))
	for i, shape := range shapes {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for j := range data {
			// Mix the tensor index into the value to avoid identical
			// payloads being deduped by any future cache layer.
			data[j] = float32((i+1)*(j+1)%131) * 0.03125
		}
		tn, err := tensor.New[float32](shape, data)
		if err != nil {
			t.Fatalf("tensor.New shape=%v: %v", shape, err)
		}
		out = append(out, tn)
	}
	return out
}

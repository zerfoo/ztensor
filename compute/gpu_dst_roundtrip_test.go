package compute

// gpu_dst_roundtrip_test.go reproduces the dst-output routing bug tracked in
// github.com/zerfoo/ztensor#79. These tests launch simple GPU binary ops with
// CPU-resident inputs, then read the destination tensor via .Data() and
// assert the expected values. They run only when CUDA is available and are
// designed to be executed on the DGX GB10 (unified memory, aarch64) where
// PatchTST training was observed to freeze with all-zero gradients.

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/tensor"
)

// TestGPUEngine_Add_DstRoundTrip_OutOfPlace exercises the simplest possible
// path: two CPU-resident inputs, a fresh CPU-resident dst, engine.Add, then
// dst.Data() must return the elementwise sum. A failure here proves the bug
// is reproducible at the ztensor level without zerfoo.
func TestGPUEngine_Add_DstRoundTrip_OutOfPlace(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	const n = 8
	aData := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	bData := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	want := []float32{11, 22, 33, 44, 55, 66, 77, 88}

	a, err := tensor.New[float32]([]int{n}, aData)
	if err != nil {
		t.Fatalf("tensor.New a: %v", err)
	}
	b, err := tensor.New[float32]([]int{n}, bData)
	if err != nil {
		t.Fatalf("tensor.New b: %v", err)
	}
	dst, err := tensor.New[float32]([]int{n}, make([]float32, n))
	if err != nil {
		t.Fatalf("tensor.New dst: %v", err)
	}

	if _, err := eng.Add(ctx, a, b, dst); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Explicit sync to remove any async/stream race from the picture.
	if err := eng.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	got := dst.Data()
	if len(got) != n {
		t.Fatalf("Data len = %d, want %d", len(got), n)
	}
	for i := 0; i < n; i++ {
		if got[i] != want[i] {
			t.Fatalf("dst[%d] = %v, want %v (full got=%v)", i, got[i], want[i], got)
		}
	}
}

// TestGPUEngine_Add_DstRoundTrip_InPlace exercises dst == a, which is how
// PatchTST's gradient accumulation path uses Add (grads[i] += dX[i]).
// If the bug is triggered by in-place aliasing, this test will fail while
// the out-of-place test passes.
func TestGPUEngine_Add_DstRoundTrip_InPlace(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	const n = 8
	aData := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	bData := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	want := []float32{11, 22, 33, 44, 55, 66, 77, 88}

	a, err := tensor.New[float32]([]int{n}, aData)
	if err != nil {
		t.Fatalf("tensor.New a: %v", err)
	}
	b, err := tensor.New[float32]([]int{n}, bData)
	if err != nil {
		t.Fatalf("tensor.New b: %v", err)
	}

	if _, err := eng.Add(ctx, a, b, a); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if err := eng.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	got := a.Data()
	for i := 0; i < n; i++ {
		if got[i] != want[i] {
			t.Fatalf("a[%d] = %v, want %v (full got=%v)", i, got[i], want[i], got)
		}
	}
}

// TestGPUEngine_Add_DstRoundTrip_RepeatedInPlace simulates the backward-pass
// pattern: grad accumulates across multiple batches via in-place Add. After
// each call the storage flips from CPUStorage to GPUStorage, so subsequent
// reads go through GPUStorage.Slice() which was the observed failure path.
func TestGPUEngine_Add_DstRoundTrip_RepeatedInPlace(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	const n = 16
	acc, err := tensor.New[float32]([]int{n}, make([]float32, n))
	if err != nil {
		t.Fatalf("tensor.New acc: %v", err)
	}
	delta, err := tensor.New[float32]([]int{n}, func() []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = 1
		}
		return d
	}())
	if err != nil {
		t.Fatalf("tensor.New delta: %v", err)
	}

	const iters = 5
	for k := 0; k < iters; k++ {
		if _, err := eng.Add(ctx, acc, delta, acc); err != nil {
			t.Fatalf("Add iter %d: %v", k, err)
		}
		if err := eng.Sync(); err != nil {
			t.Fatalf("Sync iter %d: %v", k, err)
		}
		got := acc.Data()
		wantVal := float32(k + 1)
		for i := 0; i < n; i++ {
			if got[i] != wantVal {
				t.Fatalf("iter %d: acc[%d] = %v, want %v (full=%v)", k, i, got[i], wantVal, got)
			}
		}
	}
}

// TestGPUEngine_Add_DstRoundTrip_NoExplicitSync is the same as OutOfPlace but
// does NOT call eng.Sync() before reading. If this test fails while the
// explicit-sync variant passes, the root cause is that GPUStorage.Slice()
// uses a blocking Memcpy that is not ordered with respect to the engine's
// custom CUDA stream, and the fix is to sync the stream inside Slice().
func TestGPUEngine_Add_DstRoundTrip_NoExplicitSync(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	const n = 8
	a, err := tensor.New[float32]([]int{n}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	if err != nil {
		t.Fatalf("tensor.New a: %v", err)
	}
	b, err := tensor.New[float32]([]int{n}, []float32{10, 20, 30, 40, 50, 60, 70, 80})
	if err != nil {
		t.Fatalf("tensor.New b: %v", err)
	}
	dst, err := tensor.New[float32]([]int{n}, make([]float32, n))
	if err != nil {
		t.Fatalf("tensor.New dst: %v", err)
	}

	if _, err := eng.Add(ctx, a, b, dst); err != nil {
		t.Fatalf("Add: %v", err)
	}

	want := []float32{11, 22, 33, 44, 55, 66, 77, 88}
	got := dst.Data()
	for i := 0; i < n; i++ {
		if got[i] != want[i] {
			t.Fatalf("dst[%d] = %v, want %v (full got=%v)", i, got[i], want[i], got)
		}
	}
}

// TestGPUEngine_PatchTSTBackward_DstRoundTrip is a verbatim port of the
// op sequence from zerfoo timeseries/patchtst_gpu_train.go lines 1022-1031,
// which is where gradient accumulation for the patch-embedding weight freezes
// at zero on DGX GB10. It mirrors the exact pattern:
//
//	Transpose(patches -> patchesT)
//	Zero(dPEW)                              // pre-allocated CPU-wrapper tensor
//	MatMul(patchesT, dX, dPEW)              // fresh write via GPU kernel
//	Add(gradW, dPEW, gradW)                 // in-place accumulate into CPU-wrapper
//	gradW.Data()                            // AdamW-side read
//
// gradW is pre-seeded with nonzero values that simulate a prior batch's
// accumulated gradient. After the sequence, gradW.Data() must equal
// seed + (patchesT @ dX). On the broken path, it reads all-zero.
//
// The simpler tests above chain only a single op and did NOT reproduce on
// DGX. This test chains Transpose -> Zero -> MatMul -> Add with storage-flip
// between kernel outputs and CPU-allocated wrappers, which is the minimum
// surface that reproduces the training freeze.
func TestGPUEngine_PatchTSTBackward_DstRoundTrip(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	// Shapes taken from a small PatchTST config:
	//   patches:  [totalRows, patchLen]  = [4, 3]
	//   patchesT: [patchLen, totalRows]  = [3, 4]
	//   dX:       [totalRows, dModel]    = [4, 2]
	//   dPEW:     [patchLen, dModel]     = [3, 2]
	//   gradW:    [patchLen, dModel]     = [3, 2]
	const totalRows, patchLen, dModel = 4, 3, 2

	patchesData := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}
	dXData := []float32{
		1, 1,
		2, 2,
		3, 3,
		4, 4,
	}
	// gradW pre-seeded with known nonzero values (simulating prior-batch grads
	// that AdamW already consumed on a previous step). After the accumulate,
	// these must still be present in gradW.Data() on top of the new contribution.
	gradWSeed := []float32{
		0.5, 0.5,
		0.5, 0.5,
		0.5, 0.5,
	}

	patches, err := tensor.New[float32]([]int{totalRows, patchLen}, patchesData)
	if err != nil {
		t.Fatalf("tensor.New patches: %v", err)
	}
	patchesT, err := tensor.New[float32]([]int{patchLen, totalRows}, make([]float32, patchLen*totalRows))
	if err != nil {
		t.Fatalf("tensor.New patchesT: %v", err)
	}
	dX, err := tensor.New[float32]([]int{totalRows, dModel}, dXData)
	if err != nil {
		t.Fatalf("tensor.New dX: %v", err)
	}
	dPEW, err := tensor.New[float32]([]int{patchLen, dModel}, make([]float32, patchLen*dModel))
	if err != nil {
		t.Fatalf("tensor.New dPEW: %v", err)
	}
	gradW, err := tensor.New[float32]([]int{patchLen, dModel}, append([]float32(nil), gradWSeed...))
	if err != nil {
		t.Fatalf("tensor.New gradW: %v", err)
	}

	// Step 1: Transpose patches into patchesT.
	if _, err := eng.Transpose(ctx, patches, []int{1, 0}, patchesT); err != nil {
		t.Fatalf("Transpose: %v", err)
	}
	// Step 2: Zero the pre-allocated dPEW buffer.
	if err := eng.Zero(ctx, dPEW); err != nil {
		t.Fatalf("Zero dPEW: %v", err)
	}
	// Step 3: MatMul(patchesT, dX, dPEW) -- writes into pre-allocated dPEW.
	if _, err := eng.MatMul(ctx, patchesT, dX, dPEW); err != nil {
		t.Fatalf("MatMul: %v", err)
	}
	// Step 4: In-place accumulate gradW += dPEW.
	if _, err := eng.Add(ctx, gradW, dPEW, gradW); err != nil {
		t.Fatalf("Add accumulate: %v", err)
	}
	// Force stream ordering before the D2H read, to rule out the race.
	if err := eng.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	// Expected: gradW[i,j] = seed[i,j] + sum_k patchesT[i,k] * dX[k,j]
	// patchesT is the transpose of patches, so patchesT[i,k] = patches[k,i].
	want := make([]float32, patchLen*dModel)
	copy(want, gradWSeed)
	for i := 0; i < patchLen; i++ {
		for j := 0; j < dModel; j++ {
			var acc float32
			for k := 0; k < totalRows; k++ {
				acc += patchesData[k*patchLen+i] * dXData[k*dModel+j]
			}
			want[i*dModel+j] += acc
		}
	}

	got := gradW.Data()
	if len(got) != len(want) {
		t.Fatalf("gradW.Data len = %d, want %d", len(got), len(want))
	}
	allZero := true
	for _, v := range got {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Fatalf("gradW.Data() is all-zero -- Issue #79 REPRODUCED. want=%v", want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("gradW[%d] = %v, want %v (full got=%v, want=%v)", i, got[i], want[i], got, want)
		}
	}
}

// runPatchTSTBackwardRepro runs the PatchTST patch-embedding backward op
// sequence (Transpose -> Zero -> MatMul -> in-place Add) for a given shape
// configuration and a given number of iterations. gradW is pre-seeded so a
// silent-zero readback is detectable. After each iteration the function
// asserts gradW.Data() is not all-zero and that a sample of positions
// matches the analytical expected value within tolerance.
func runPatchTSTBackwardRepro(t *testing.T, eng *GPUEngine[float32], totalRows, patchLen, dModel, iters int) {
	t.Helper()
	ctx := context.Background()

	patchesData := make([]float32, totalRows*patchLen)
	for i := range patchesData {
		patchesData[i] = float32((i%7)+1) * 0.01
	}
	dXData := make([]float32, totalRows*dModel)
	for i := range dXData {
		dXData[i] = float32((i%5)+1) * 0.01
	}
	gradWSeed := make([]float32, patchLen*dModel)
	for i := range gradWSeed {
		gradWSeed[i] = float32(i+1) * 0.001
	}

	patches, err := tensor.New[float32]([]int{totalRows, patchLen}, patchesData)
	if err != nil {
		t.Fatalf("tensor.New patches: %v", err)
	}
	patchesT, err := tensor.New[float32]([]int{patchLen, totalRows}, make([]float32, patchLen*totalRows))
	if err != nil {
		t.Fatalf("tensor.New patchesT: %v", err)
	}
	dX, err := tensor.New[float32]([]int{totalRows, dModel}, dXData)
	if err != nil {
		t.Fatalf("tensor.New dX: %v", err)
	}
	dPEW, err := tensor.New[float32]([]int{patchLen, dModel}, make([]float32, patchLen*dModel))
	if err != nil {
		t.Fatalf("tensor.New dPEW: %v", err)
	}
	gradW, err := tensor.New[float32]([]int{patchLen, dModel}, append([]float32(nil), gradWSeed...))
	if err != nil {
		t.Fatalf("tensor.New gradW: %v", err)
	}

	// Single-batch analytical contribution: patchesT @ dX.
	contrib := make([]float32, patchLen*dModel)
	for i := 0; i < patchLen; i++ {
		for j := 0; j < dModel; j++ {
			var acc float32
			for k := 0; k < totalRows; k++ {
				acc += patchesData[k*patchLen+i] * dXData[k*dModel+j]
			}
			contrib[i*dModel+j] = acc
		}
	}

	for iter := 0; iter < iters; iter++ {
		if _, err := eng.Transpose(ctx, patches, []int{1, 0}, patchesT); err != nil {
			t.Fatalf("iter %d Transpose: %v", iter, err)
		}
		if err := eng.Zero(ctx, dPEW); err != nil {
			t.Fatalf("iter %d Zero: %v", iter, err)
		}
		if _, err := eng.MatMul(ctx, patchesT, dX, dPEW); err != nil {
			t.Fatalf("iter %d MatMul: %v", iter, err)
		}
		if _, err := eng.Add(ctx, gradW, dPEW, gradW); err != nil {
			t.Fatalf("iter %d Add: %v", iter, err)
		}
		if err := eng.Sync(); err != nil {
			t.Fatalf("iter %d Sync: %v", iter, err)
		}

		got := gradW.Data()
		if len(got) != patchLen*dModel {
			t.Fatalf("iter %d: gradW.Data len = %d, want %d", iter, len(got), patchLen*dModel)
		}
		allZero := true
		for _, v := range got {
			if v != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Fatalf("iter %d: gradW.Data() is ALL-ZERO -- Issue #79 REPRODUCED at shape totalRows=%d patchLen=%d dModel=%d",
				iter, totalRows, patchLen, dModel)
		}
		nIter := float32(iter + 1)
		for _, idx := range []int{0, 1, patchLen*dModel/2, patchLen*dModel - 1} {
			w := gradWSeed[idx] + nIter*contrib[idx]
			cur := got[idx]
			tol := float32(1e-2) * (1 + float32(iter))
			diff := cur - w
			if diff < 0 {
				diff = -diff
			}
			if diff > tol {
				t.Fatalf("iter %d gradW[%d] = %v, want %v (tol=%v, diff=%v)",
					iter, idx, cur, w, tol, diff)
			}
		}
	}
}

// TestGPUEngine_PatchTSTBackward_RealisticShapes matches the default
// cmd/bench_train/main.go config (PatchLength=8, Stride=4, DModel=64,
// BatchSize=64, Channels=5, NumPatches=5 -> totalRows=1600) and runs
// 20 backward accumulations to exercise arena/alias state that may only
// manifest after many GPU-kernel output routings into the same wrapper.
func TestGPUEngine_PatchTSTBackward_RealisticShapes(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	eng := newTestGPUEngine(t)
	runPatchTSTBackwardRepro(t, eng, 1600, 8, 64, 20)
}

// TestGPUEngine_PatchTSTBackward_LargerBatch pushes totalRows to 3200 to
// match bench-spark -samples 5000 -channels 10 style configurations.
func TestGPUEngine_PatchTSTBackward_LargerBatch(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	eng := newTestGPUEngine(t)
	runPatchTSTBackwardRepro(t, eng, 3200, 8, 64, 20)
}

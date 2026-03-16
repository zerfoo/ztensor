package compute

import (
	"context"
	"reflect"
	"runtime"
	"sync/atomic"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// ---------------------------------------------------------------------------
// parallelForCtx coverage
// ---------------------------------------------------------------------------

func TestParallelForCtx_AlreadyCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	called := false
	err := parallelForCtx(ctx, 10, func(_, _ int) { called = true })
	if err == nil {
		t.Fatal("expected context error, got nil")
	}
	if called {
		t.Fatal("fn should not have been called for a pre-canceled context")
	}
}

func TestParallelForCtx_CancelledMidLoop(t *testing.T) {
	// To hit the mid-loop cancellation (line 128), the context must
	// become Done after the initial check (line 95) but during a loop
	// iteration (line 125).
	//
	// The loop dispatches goroutines without waiting, so we need the
	// cancel to happen fast enough for a subsequent iteration to see it.
	//
	// Strategy: use a very large total so there are many loop iterations,
	// and cancel from a goroutine spawned by fn on the first chunk.
	// The loop runs in the main goroutine; once we call cancel(), the
	// channel returned by ctx.Done() is closed, so a subsequent select
	// in the loop will see it. We use runtime.Gosched() to encourage
	// scheduling.
	const total = 32768 * 128 // many chunks -> many loop iterations
	ctx, cancel := context.WithCancel(context.Background())

	var chunks int64
	err := parallelForCtx(ctx, total, func(_, _ int) {
		n := atomic.AddInt64(&chunks, 1)
		if n == 1 {
			cancel()
			// Yield to give the main goroutine (loop) a chance to
			// check ctx.Done() on its next iteration.
			runtime.Gosched()
		}
	})

	// The test is inherently non-deterministic. We accept either outcome
	// but verify consistency: if error is returned, not all chunks ran.
	if err != nil {
		if chunks == int64(total/32768*128) {
			t.Fatal("error returned but all chunks were dispatched")
		}
	}
	// Even if err is nil, the cancel() call in fn still exercises the
	// select path in many scheduler orderings.
}

func TestParallelForCtx_SmallTotal(t *testing.T) {
	// total <= minPerG: runs synchronously, no goroutines.
	var called bool
	err := parallelForCtx(context.Background(), 100, func(start, end int) {
		if start != 0 || end != 100 {
			t.Fatalf("expected (0, 100), got (%d, %d)", start, end)
		}
		called = true
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Fatal("fn was not called")
	}
}

func TestParallelForCtx_LargeTotal_NoCancellation(t *testing.T) {
	const total = 32768 * 4
	var processed int64
	err := parallelForCtx(context.Background(), total, func(start, end int) {
		atomic.AddInt64(&processed, int64(end-start))
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if processed != int64(total) {
		t.Fatalf("expected all %d elements processed, got %d", total, processed)
	}
}

// ---------------------------------------------------------------------------
// MatMul context cancellation
// ---------------------------------------------------------------------------

func TestMatMul_ContextCancelled(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	b, _ := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})

	_, err := engine.MatMul(ctx, a, b)
	if err == nil {
		t.Fatal("expected context error, got nil")
	}
}

// ---------------------------------------------------------------------------
// TestableMatMul: nil result and invalid shapes
// ---------------------------------------------------------------------------

func TestTestableMatMul_NilResult(t *testing.T) {
	engine := NewTestableEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	b, _ := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})

	err := engine.TestableMatMul(ctx, a, b, nil)
	if err == nil {
		t.Fatal("expected error for nil result, got nil")
	}
}

func TestTestableMatMul_InvalidShapes(t *testing.T) {
	engine := NewTestableEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	tests := []struct {
		name   string
		aShape []int
		aData  []float32
		bShape []int
		bData  []float32
	}{
		{
			name:   "1D tensors",
			aShape: []int{4},
			aData:  []float32{1, 2, 3, 4},
			bShape: []int{4},
			bData:  []float32{5, 6, 7, 8},
		},
		{
			name:   "incompatible inner dims",
			aShape: []int{2, 3},
			aData:  []float32{1, 2, 3, 4, 5, 6},
			bShape: []int{2, 2},
			bData:  []float32{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, _ := tensor.New[float32](tt.aShape, tt.aData)
			b, _ := tensor.New[float32](tt.bShape, tt.bData)
			result, _ := tensor.New[float32]([]int{2, 2}, make([]float32, 4))
			fr := NewFailableTensor(result)

			err := engine.TestableMatMul(ctx, a, b, fr)
			if err == nil {
				t.Fatal("expected error for invalid shapes, got nil")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestableTranspose: nil result and non-2D tensor
// ---------------------------------------------------------------------------

func TestTestableTranspose_NilResult(t *testing.T) {
	engine := NewTestableEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

	err := engine.TestableTranspose(ctx, a, nil)
	if err == nil {
		t.Fatal("expected error for nil result, got nil")
	}
}

func TestTestableTranspose_Non2D(t *testing.T) {
	engine := NewTestableEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 3, 4}, make([]float32, 24))
	result, _ := tensor.New[float32]([]int{4, 3}, make([]float32, 12))
	fr := NewFailableTensor(result)

	err := engine.TestableTranspose(ctx, a, fr)
	if err == nil {
		t.Fatal("expected error for non-2D tensor, got nil")
	}
}

// ---------------------------------------------------------------------------
// equalSlices: different lengths
// ---------------------------------------------------------------------------

func TestEqualSlices_DifferentLengths(t *testing.T) {
	if equalSlices([]int{1, 2}, []int{1, 2, 3}) {
		t.Fatal("expected false for slices of different lengths")
	}
	if equalSlices([]int{}, []int{1}) {
		t.Fatal("expected false for empty vs non-empty")
	}
	// same length, different values (already covered, but confirm)
	if equalSlices([]int{1, 2}, []int{1, 3}) {
		t.Fatal("expected false for different values")
	}
	// identical
	if !equalSlices([]int{1, 2}, []int{1, 2}) {
		t.Fatal("expected true for equal slices")
	}
}

// ---------------------------------------------------------------------------
// MatMul with float64 (covers the float64 case branch in MatMul)
// ---------------------------------------------------------------------------

func TestMatMul_Float64(t *testing.T) {
	engine := NewCPUEngine[float64](numeric.Float64Ops{})
	ctx := context.Background()

	a, _ := tensor.New[float64]([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	b, _ := tensor.New[float64]([]int{3, 2}, []float64{7, 8, 9, 10, 11, 12})

	result, err := engine.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
	// [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
	expected := []float64{58, 64, 139, 154}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestMatMul_Float64_Batched(t *testing.T) {
	engine := NewCPUEngine[float64](numeric.Float64Ops{})
	ctx := context.Background()

	// Batched: same-rank 3D tensors -> covers lines 832-846
	a, _ := tensor.New[float64]([]int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	b, _ := tensor.New[float64]([]int{2, 2, 2}, []float64{1, 0, 0, 1, 2, 0, 0, 2})

	result, err := engine.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// batch 0: [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
	// batch 1: [[5,6],[7,8]] @ [[2,0],[0,2]] = [[10,12],[14,16]]
	expected := []float64{1, 2, 3, 4, 10, 12, 14, 16}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

// ---------------------------------------------------------------------------
// makeStrides and expandShapeStrides edge cases
// ---------------------------------------------------------------------------

func TestMakeStrides_EmptyShape(t *testing.T) {
	strides := makeStrides([]int{})
	if len(strides) != 0 {
		t.Fatalf("expected empty strides, got %v", strides)
	}
}

func TestExpandShapeStrides_PadNegative(t *testing.T) {
	// rank < len(shape) -> pad is 0 (pad = rank - len(shape) < 0 -> clamped to 0)
	shape := []int{2, 3, 4}
	strides := makeStrides(shape)
	es, est := expandShapeStrides(shape, strides, 2)
	// When rank < len(shape), pad=0, only copies first `rank` elements
	if len(es) != 2 || len(est) != 2 {
		t.Fatalf("expected length 2, got %d, %d", len(es), len(est))
	}
}

// ---------------------------------------------------------------------------
// Sum with negative axis
// ---------------------------------------------------------------------------

func TestSum_NegativeAxis_SumsAll(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Negative axis means "sum all elements" in this implementation.
	a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

	result, err := engine.Sum(ctx, a, -1, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	data := result.Data()
	if len(data) != 1 || data[0] != 21 {
		t.Errorf("expected [21], got %v", data)
	}
}

// ---------------------------------------------------------------------------
// Reshape: inferred dimension not divisible
// ---------------------------------------------------------------------------

func TestReshape_InferredDimNotDivisible(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

	// 6 elements, try to reshape to [-1, 4] -> 6/4 is not integer
	_, err := engine.Reshape(ctx, a, []int{-1, 4})
	if err == nil {
		t.Fatal("expected error for non-divisible inferred dimension")
	}
}

// ---------------------------------------------------------------------------
// Zeros with invalid shape (covers tensor.New error in Zeros)
// ---------------------------------------------------------------------------

func TestZeros_InvalidShape(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})

	// Negative dimension should cause tensor.New to fail
	err := engine.Zeros(ctx, a, []int{-1})
	if err == nil {
		t.Fatal("expected error for invalid shape in Zeros")
	}
}

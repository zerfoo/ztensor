package cuda

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

// enableArenaPoisonForTest turns the poison mode on for one test and restores
// the process default afterwards. The flag is normally read once from
// ZTENSOR_ARENA_POISON at init; tests flip the variable directly.
func enableArenaPoisonForTest(t *testing.T, enabled bool) {
	t.Helper()
	orig := arenaPoisonEnabled
	arenaPoisonEnabled = enabled
	t.Cleanup(func() { arenaPoisonEnabled = orig })
}

// swapHostPoisonFillForTest replaces the device fill with one that writes the
// production byte pattern (fillHostPoison) directly into host memory -- valid
// here because the test arenas are backed by host buffers. Returns a counter
// of fill invocations.
func swapHostPoisonFillForTest(t *testing.T) *int {
	t.Helper()
	calls := 0
	orig := arenaPoisonFillFn
	arenaPoisonFillFn = func(ptr unsafe.Pointer, byteLen int) error {
		calls++
		fillHostPoison(unsafe.Slice((*byte)(ptr), byteLen))
		return nil
	}
	t.Cleanup(func() { arenaPoisonFillFn = orig })
	return &calls
}

// swapPoisonWarnForTest captures poison-mode warnings.
func swapPoisonWarnForTest(t *testing.T) *[]string {
	t.Helper()
	var warns []string
	orig := arenaPoisonWarnFn
	arenaPoisonWarnFn = func(format string, args ...any) {
		warns = append(warns, fmt.Sprintf(format, args...))
	}
	t.Cleanup(func() { arenaPoisonWarnFn = orig })
	return &warns
}

// newHostArena builds an ArenaPool backed by a real host buffer so pointers
// returned by Alloc are dereferenceable without a GPU (same pattern as
// TestArenaDiagnostics_Fields).
func newHostArena(t *testing.T, size int) *ArenaPool {
	t.Helper()
	return NewHostBackedArenaForTesting(make([]byte, size))
}

// TestArenaPoisonWord_Pattern pins down the sentinel: the repeated 32-bit
// word must decode to NaN for both f32 and aligned f64 reads, and to a
// recognizable sentinel for integer reads (ADR 006 decision 4).
func TestArenaPoisonWord_Pattern(t *testing.T) {
	if f := math.Float32frombits(ArenaPoisonWord); !math.IsNaN(float64(f)) {
		t.Errorf("f32 0x%08X is not NaN", ArenaPoisonWord)
	}
	q := uint64(ArenaPoisonWord)<<32 | uint64(ArenaPoisonWord)
	if d := math.Float64frombits(q); !math.IsNaN(d) {
		t.Errorf("f64 0x%016X (pattern repeated) is not NaN", q)
	}

	// The byte fill must reproduce the word on a little-endian decode for
	// every 4-byte-aligned phase.
	b := make([]byte, 32)
	fillHostPoison(b)
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(b)/4)
	for i, w := range words {
		if w != ArenaPoisonWord {
			t.Fatalf("word %d: got 0x%08X, want 0x%08X", i, w, ArenaPoisonWord)
		}
	}
	f64s := unsafe.Slice((*float64)(unsafe.Pointer(&b[0])), len(b)/8)
	for i, d := range f64s {
		if !math.IsNaN(d) {
			t.Fatalf("f64 read %d of poisoned bytes is not NaN: %v", i, d)
		}
	}
}

// TestArenaPoison_CachedBufferAfterReset is the T1.4 demo/regression test for
// the shipped bug class (zerfoo#842, zerfoo#845, Wolf QK-norm): a fake node
// caches a tensor backed by arena memory during Forward, the pool is reset
// between steps, and Backward reads the cached pointer.
//
// Under ZTENSOR_ARENA_POISON=1 semantics the stale read must see NaN at the
// corruption site; without poison it silently sees the old (clean) values --
// exactly the delayed-corruption behavior the mode exists to expose.
func TestArenaPoison_CachedBufferAfterReset(t *testing.T) {
	const numFloats = 256
	byteSize := numFloats * 4

	run := func(t *testing.T, poison bool) []float32 {
		t.Helper()
		enableArenaPoisonForTest(t, poison)
		swapHostPoisonFillForTest(t)

		a := newHostArena(t, 4096)
		ptr, err := a.Alloc(0, byteSize)
		if err != nil {
			t.Fatalf("Alloc: %v", err)
		}
		// "Forward": the fake node writes an intermediate and caches the
		// tensor in a struct field (here: keeps the slice across Reset).
		cached := unsafe.Slice((*float32)(ptr), numFloats)
		for i := range cached {
			cached[i] = 1.5
		}
		// Step boundary: the training loop resets the pool.
		a.Reset()
		// "Backward": the node reads its cached intermediate.
		return cached
	}

	t.Run("poison=1 stale read explodes as NaN", func(t *testing.T) {
		cached := run(t, true)
		for i, v := range cached {
			if !math.IsNaN(float64(v)) {
				t.Fatalf("cached[%d] after Reset = %v, want NaN", i, v)
			}
		}
		if bits := math.Float32bits(cached[0]); bits != ArenaPoisonWord {
			t.Fatalf("cached[0] bits = 0x%08X, want 0x%08X", bits, ArenaPoisonWord)
		}
	})

	t.Run("poison off stale read sees clean values", func(t *testing.T) {
		cached := run(t, false)
		for i, v := range cached {
			if v != 1.5 {
				t.Fatalf("cached[%d] after Reset = %v, want 1.5 (untouched)", i, v)
			}
		}
	})
}

// TestArenaPoison_FreeArenaPoisonsBlock verifies the intra-step reuse path:
// a block is poisoned the moment it enters the free-list, and the poison is
// still present when the block is handed back out by Alloc.
func TestArenaPoison_FreeArenaPoisonsBlock(t *testing.T) {
	enableArenaPoisonForTest(t, true)
	swapHostPoisonFillForTest(t)

	a := newHostArena(t, 4096)
	ptr, err := a.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	vals := unsafe.Slice((*float32)(ptr), 256)
	for i := range vals {
		vals[i] = 2.0
	}

	// Free routes arena pointers to FreeArena: the block must be poisoned
	// immediately, before any reuse.
	a.Free(0, ptr, 1024)
	for i, v := range vals {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("vals[%d] after Free = %v, want NaN", i, v)
		}
	}

	// Reuse hands the same (still poisoned) block back out.
	ptr2, err := a.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc (reuse): %v", err)
	}
	if ptr2 != ptr {
		t.Fatalf("expected free-list reuse of the same block, got %p want %p", ptr2, ptr)
	}
	if v := *(*float32)(ptr2); !math.IsNaN(float64(v)) {
		t.Fatalf("reused block first word = %v, want NaN", v)
	}
}

// TestArenaPoison_ResetRespectsResetFloor verifies that buffers below the
// reset floor (weights, optimizer state, captured-graph buffers) are never
// poisoned; only the reclaimed span above the floor is filled.
func TestArenaPoison_ResetRespectsResetFloor(t *testing.T) {
	enableArenaPoisonForTest(t, true)
	swapHostPoisonFillForTest(t)

	a := newHostArena(t, 4096)
	persistentPtr, err := a.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc persistent: %v", err)
	}
	persistent := unsafe.Slice((*float32)(persistentPtr), 64)
	for i := range persistent {
		persistent[i] = 3.0
	}
	a.SetResetFloor(a.UsedBytes()) // MarkStepBoundary equivalent

	stepPtr, err := a.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc step: %v", err)
	}
	step := unsafe.Slice((*float32)(stepPtr), 64)
	for i := range step {
		step[i] = 4.0
	}

	a.Reset()

	for i, v := range persistent {
		if v != 3.0 {
			t.Fatalf("persistent[%d] = %v, want 3.0 (below reset floor, must not be poisoned)", i, v)
		}
	}
	for i, v := range step {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("step[%d] = %v, want NaN (above reset floor)", i, v)
		}
	}
}

// TestArenaPoison_SkippedDuringCapture verifies the ADR 004/005 interaction:
// while a CUDA graph capture is active, poison fills are skipped with a
// warning instead of issuing capture-unsafe work.
func TestArenaPoison_SkippedDuringCapture(t *testing.T) {
	resetCaptureStateForTest(t)
	enableArenaPoisonForTest(t, true)
	fills := swapHostPoisonFillForTest(t)
	warns := swapPoisonWarnForTest(t)

	a := newHostArena(t, 4096)
	ptr, err := a.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	vals := unsafe.Slice((*float32)(ptr), 256)
	vals[0] = 5.0

	const captureHandle = uintptr(0xCAFE)
	markStreamCapturing(captureHandle)
	defer unmarkStreamCapturing(captureHandle)

	a.Reset()

	if *fills != 0 {
		t.Fatalf("poison fill ran %d times during capture, want 0", *fills)
	}
	if len(*warns) != 1 {
		t.Fatalf("got %d warnings, want exactly 1: %v", len(*warns), *warns)
	}
	if vals[0] != 5.0 {
		t.Fatalf("vals[0] = %v, want 5.0 (fill must be skipped during capture)", vals[0])
	}
}

// TestArenaPoison_ZeroWorkWhenDisabled verifies the off-by-default contract:
// with the mode disabled, no fill is ever attempted on any reuse path.
func TestArenaPoison_ZeroWorkWhenDisabled(t *testing.T) {
	enableArenaPoisonForTest(t, false)
	fills := swapHostPoisonFillForTest(t)

	a := newHostArena(t, 4096)
	ptr, err := a.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	a.Free(0, ptr, 1024)
	if _, err := a.Alloc(0, 512); err != nil {
		t.Fatalf("Alloc (reuse): %v", err)
	}
	a.Reset()

	if *fills != 0 {
		t.Fatalf("poison fill ran %d times with mode disabled, want 0", *fills)
	}
}

// TestSetArenaPoisonFill_NilRestoresDefault pins the SetArenaPoisonFill
// contract used by internal/gpuapi to register the device fill kernel.
func TestSetArenaPoisonFill_NilRestoresDefault(t *testing.T) {
	orig := arenaPoisonFillFn
	t.Cleanup(func() { arenaPoisonFillFn = orig })

	called := false
	SetArenaPoisonFill(func(unsafe.Pointer, int) error {
		called = true
		return nil
	})
	if err := arenaPoisonFillFn(nil, 0); err != nil {
		t.Fatalf("custom fill: %v", err)
	}
	if !called {
		t.Fatal("SetArenaPoisonFill did not install the custom fill")
	}

	SetArenaPoisonFill(nil)
	// Cannot invoke the default without a CUDA runtime; identity check only.
	if arenaPoisonFillFn == nil {
		t.Fatal("SetArenaPoisonFill(nil) must restore the default, not nil")
	}
}

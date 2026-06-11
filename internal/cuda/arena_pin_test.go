package cuda

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

// swapPinWarnForTest captures pin-misuse warnings (refcount underflow).
func swapPinWarnForTest(t *testing.T) *[]string {
	t.Helper()
	var warns []string
	orig := arenaPinWarnFn
	arenaPinWarnFn = func(format string, args ...any) {
		warns = append(warns, fmt.Sprintf(format, args...))
	}
	t.Cleanup(func() { arenaPinWarnFn = orig })
	return &warns
}

func fillF32(ptr unsafe.Pointer, n int, v float32) []float32 {
	s := unsafe.Slice((*float32)(ptr), n)
	for i := range s {
		s[i] = v
	}
	return s
}

// TestArenaPin_PinnedSurvivesReset is the core S2.2.1 guarantee: a pinned
// buffer's contents survive Reset (value intact, not poisoned), while an
// unpinned buffer allocated above it is reclaimed and poisoned.
func TestArenaPin_PinnedSurvivesReset(t *testing.T) {
	restore := SetArenaPoisonEnabledForTesting(true)
	t.Cleanup(restore)
	swapHostPoisonFillForTest(t)

	a := newHostArena(t, 8192)
	pinnedPtr, err := a.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc pinned: %v", err)
	}
	pinned := fillF32(pinnedPtr, 256, 1.5)
	if !a.Pin(pinnedPtr, 1024) {
		t.Fatal("Pin returned false for an arena pointer")
	}

	unpinnedPtr, err := a.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc unpinned: %v", err)
	}
	unpinned := fillF32(unpinnedPtr, 256, 2.5)

	a.Reset()

	for i, v := range pinned {
		if v != 1.5 {
			t.Fatalf("pinned[%d] after Reset = %v, want 1.5 (pinned span must survive)", i, v)
		}
	}
	for i, v := range unpinned {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("unpinned[%d] after Reset = %v, want NaN (reclaimed span must be poisoned)", i, v)
		}
	}
	// The rewind floor was raised to the end of the pinned span, so the
	// pinned bytes cannot be re-issued by the bump allocator.
	if got := a.UsedBytes(); got != 1024 {
		t.Fatalf("UsedBytes after Reset with pin = %d, want 1024 (raised floor)", got)
	}

	// After the last Unpin the next Reset reclaims (and poisons) the span.
	a.Unpin(pinnedPtr)
	a.Reset()
	for i, v := range pinned {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("pinned[%d] after Unpin+Reset = %v, want NaN", i, v)
		}
	}
	if got := a.UsedBytes(); got != 0 {
		t.Fatalf("UsedBytes after Unpin+Reset = %d, want 0", got)
	}
}

// TestArenaPin_RaisedFloorRetainsDeadBytes documents the watermark
// consequence of the raise-the-floor semantics: dead, unpinned bytes BELOW
// a pinned span stay retained (not re-issuable) until the pins release, but
// they are still poisoned at Reset.
func TestArenaPin_RaisedFloorRetainsDeadBytes(t *testing.T) {
	restore := SetArenaPoisonEnabledForTesting(true)
	t.Cleanup(restore)
	swapHostPoisonFillForTest(t)

	a := newHostArena(t, 8192)
	deadPtr, err := a.Alloc(0, 1024) // [0, 1024): dead after Reset
	if err != nil {
		t.Fatalf("Alloc dead: %v", err)
	}
	dead := fillF32(deadPtr, 256, 3.0)
	pinnedPtr, err := a.Alloc(0, 1024) // [1024, 2048): pinned
	if err != nil {
		t.Fatalf("Alloc pinned: %v", err)
	}
	pinned := fillF32(pinnedPtr, 256, 4.0)
	a.Pin(pinnedPtr, 1024)

	a.Reset()

	// Floor raised past the pinned span retains the dead bytes below it...
	if got := a.UsedBytes(); got != 2048 {
		t.Fatalf("UsedBytes = %d, want 2048 (floor raised to end of pinned span)", got)
	}
	// ...but they are dead by contract, so they are poisoned anyway.
	for i, v := range dead {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("dead[%d] = %v, want NaN (dead bytes below raised floor are poisoned)", i, v)
		}
	}
	for i, v := range pinned {
		if v != 4.0 {
			t.Fatalf("pinned[%d] = %v, want 4.0", i, v)
		}
	}

	a.Unpin(pinnedPtr)
	a.Reset()
	if got := a.UsedBytes(); got != 0 {
		t.Fatalf("UsedBytes after Unpin+Reset = %d, want 0 (retained span reclaimed)", got)
	}
}

// TestArenaPin_FreeArenaDeferredWhilePinned: FreeArena on a pinned buffer
// must not poison it or hand it to the free-list until the last Unpin.
func TestArenaPin_FreeArenaDeferredWhilePinned(t *testing.T) {
	restore := SetArenaPoisonEnabledForTesting(true)
	t.Cleanup(restore)
	swapHostPoisonFillForTest(t)

	a := newHostArena(t, 8192)
	ptr, err := a.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	vals := fillF32(ptr, 256, 5.0)
	a.Pin(ptr, 1024)

	// Free routes arena pointers to FreeArena; the pin defers it.
	a.Free(0, ptr, 1024)
	for i, v := range vals {
		if v != 5.0 {
			t.Fatalf("vals[%d] after deferred Free = %v, want 5.0 (no poison while pinned)", i, v)
		}
	}
	if got := a.FreeListLen(); got != 0 {
		t.Fatalf("free-list length = %d, want 0 (free deferred while pinned)", got)
	}
	// A same-size Alloc must NOT reuse the pinned block.
	other, err := a.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc other: %v", err)
	}
	if other == ptr {
		t.Fatal("Alloc reused a pinned block")
	}

	// Last Unpin applies the deferred free: poison fill + free-list insert.
	a.Unpin(ptr)
	for i, v := range vals {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("vals[%d] after Unpin = %v, want NaN (deferred free poisons on release)", i, v)
		}
	}
	if got := a.FreeListLen(); got != 1 {
		t.Fatalf("free-list length after Unpin = %d, want 1", got)
	}
	reused, err := a.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc reuse: %v", err)
	}
	if reused != ptr {
		t.Fatalf("expected free-list reuse of released block, got %p want %p", reused, ptr)
	}
}

// TestArenaPin_RefcountedPin: each Pin must be balanced by one Unpin; the
// span is only released when the last reference drops.
func TestArenaPin_RefcountedPin(t *testing.T) {
	restore := SetArenaPoisonEnabledForTesting(true)
	t.Cleanup(restore)
	swapHostPoisonFillForTest(t)

	a := newHostArena(t, 4096)
	ptr, err := a.Alloc(0, 512)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	vals := fillF32(ptr, 128, 6.0)
	a.Pin(ptr, 512)
	a.Pin(ptr, 512)

	a.Unpin(ptr) // one reference still held
	a.Reset()
	for i, v := range vals {
		if v != 6.0 {
			t.Fatalf("vals[%d] = %v, want 6.0 (one pin reference still held)", i, v)
		}
	}

	a.Unpin(ptr) // last reference
	a.Reset()
	if v := vals[0]; !math.IsNaN(float64(v)) {
		t.Fatalf("vals[0] after final Unpin+Reset = %v, want NaN", v)
	}
}

// TestArenaPin_UnpinUnderflowWarns: Unpin without a matching Pin logs a
// warning and never panics (production-path guard).
func TestArenaPin_UnpinUnderflowWarns(t *testing.T) {
	warns := swapPinWarnForTest(t)

	a := newHostArena(t, 4096)
	ptr, err := a.Alloc(0, 512)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	a.Unpin(ptr)
	if len(*warns) != 1 {
		t.Fatalf("got %d warnings, want 1: %v", len(*warns), *warns)
	}

	// Balanced pin/unpin then one extra: the extra must warn, not panic.
	a.Pin(ptr, 512)
	a.Unpin(ptr)
	a.Unpin(ptr)
	if len(*warns) != 2 {
		t.Fatalf("got %d warnings, want 2: %v", len(*warns), *warns)
	}
}

// TestArenaPin_NonArenaPointerNoop: pointers outside the arena (fallback /
// CPU memory) are not subject to Reset reclamation, so Pin is a no-op that
// returns false and Unpin is silent.
func TestArenaPin_NonArenaPointerNoop(t *testing.T) {
	warns := swapPinWarnForTest(t)

	a := newHostArena(t, 4096)
	outside := make([]byte, 256)
	if a.Pin(unsafe.Pointer(&outside[0]), 256) {
		t.Fatal("Pin returned true for a non-arena pointer")
	}
	a.Unpin(unsafe.Pointer(&outside[0]))
	if a.Pin(nil, 256) {
		t.Fatal("Pin returned true for nil")
	}
	a.Unpin(nil)
	if len(*warns) != 0 {
		t.Fatalf("non-arena Unpin warned: %v", *warns)
	}
	if got := a.PinnedBytes(); got != 0 {
		t.Fatalf("PinnedBytes = %d, want 0", got)
	}
}

// TestArenaPin_PinnedBytesTracking: PinnedBytes reflects live pins (counting
// each span once regardless of refcount) and PinnedHighWaterBytes records
// the maximum (the monitoring number for the contract's watermark cost).
func TestArenaPin_PinnedBytesTracking(t *testing.T) {
	a := newHostArena(t, 8192)
	p1, _ := a.Alloc(0, 1024)
	p2, _ := a.Alloc(0, 512)

	a.Pin(p1, 1024)
	if got := a.PinnedBytes(); got != 1024 {
		t.Fatalf("PinnedBytes = %d, want 1024", got)
	}
	a.Pin(p1, 1024) // second reference: no double count
	if got := a.PinnedBytes(); got != 1024 {
		t.Fatalf("PinnedBytes after re-pin = %d, want 1024", got)
	}
	a.Pin(p2, 512)
	if got := a.PinnedBytes(); got != 1536 {
		t.Fatalf("PinnedBytes = %d, want 1536", got)
	}

	a.Unpin(p2)
	a.Unpin(p1)
	a.Unpin(p1)
	if got := a.PinnedBytes(); got != 0 {
		t.Fatalf("PinnedBytes after release = %d, want 0", got)
	}
	if got := a.PinnedHighWaterBytes(); got != 1536 {
		t.Fatalf("PinnedHighWaterBytes = %d, want 1536", got)
	}
}

// TestArenaPin_ResetDropsDeferredFrees: a deferred free that survives into a
// Reset is dropped (the rewind subsumes it); the later Unpin must not insert
// a stale block into the free-list.
func TestArenaPin_ResetDropsDeferredFrees(t *testing.T) {
	a := newHostArena(t, 8192)
	ptr, err := a.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	a.Pin(ptr, 1024)
	a.FreeArena(ptr, 1024) // deferred: overlaps the pin

	a.Reset() // floor raised to 1024; deferred free dropped

	a.Unpin(ptr)
	if got := a.FreeListLen(); got != 0 {
		t.Fatalf("free-list length after Reset+Unpin = %d, want 0 (deferred free dropped by Reset)", got)
	}
	a.Reset() // pins gone: full rewind reclaims the retained span
	if got := a.UsedBytes(); got != 0 {
		t.Fatalf("UsedBytes = %d, want 0", got)
	}
}

// TestArenaPin_PoisonOffNoFills: with poison mode disabled, pin/unpin and
// deferred frees never attempt a fill.
func TestArenaPin_PoisonOffNoFills(t *testing.T) {
	restore := SetArenaPoisonEnabledForTesting(false)
	t.Cleanup(restore)
	fills := swapHostPoisonFillForTest(t)

	a := newHostArena(t, 4096)
	ptr, _ := a.Alloc(0, 512)
	a.Pin(ptr, 512)
	a.Free(0, ptr, 512)
	a.Reset()
	a.Unpin(ptr)
	a.Reset()
	if *fills != 0 {
		t.Fatalf("poison fill ran %d times with mode disabled, want 0", *fills)
	}
}

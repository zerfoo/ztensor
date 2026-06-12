package cuda

import (
	"math"
	"testing"
	"unsafe"
)

// Bug 11 residual regression: an arena allocation's lifetime cannot extend
// across a Reset. A Free that arrives AFTER the Reset that reclaimed its
// allocation -- the canonical case is a Go GC finalizer on a dead GPUStorage
// firing one collection cycle late (the Wolf training loop's first big GC
// landed at batch 3-4 and freed thousands of pre-Reset pointers) -- must be
// dropped. Honoring it poisons live data of the current epoch and
// double-issues the block through the free-list.

func newEpochTestArena(t *testing.T, elems int) (*ArenaPool, []float32) {
	t.Helper()
	buf := make([]float32, elems)
	a := NewHostBackedArenaForTesting(unsafe.Slice((*byte)(unsafe.Pointer(&buf[0])), elems*4))
	return a, buf
}

func TestFreeAtEpoch_StaleFreeIsDropped(t *testing.T) {
	a, _ := newEpochTestArena(t, 1024)

	// Epoch 0: allocate A.
	e0 := a.Epoch()
	ptrA, err := a.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc A: %v", err)
	}

	// Reset reclaims A wholesale and advances the epoch.
	a.Reset()
	if got := a.Epoch(); got != e0+1 {
		t.Fatalf("Epoch after Reset = %d, want %d", got, e0+1)
	}

	// Epoch 1: B is bump-allocated at the SAME offset A had.
	ptrB, err := a.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc B: %v", err)
	}
	if ptrA != ptrB {
		t.Fatalf("test setup: expected B to reuse A's offset (A=%p B=%p)", ptrA, ptrB)
	}

	// The stale free: A's finalizer fires now, carrying A's allocation epoch.
	a.FreeAtEpoch(0, ptrA, 256, e0)

	if got := a.StaleFrees(); got != 1 {
		t.Fatalf("StaleFrees = %d, want 1", got)
	}
	if got := a.FreeListLen(); got != 0 {
		t.Fatalf("free-list length after stale free = %d, want 0 (block belongs to live B)", got)
	}

	// The next allocation must NOT alias B's live memory.
	ptrC, err := a.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc C: %v", err)
	}
	if ptrC == ptrB {
		t.Fatal("stale free double-issued B's block: C aliases live allocation B")
	}
}

func TestFreeAtEpoch_SameEpochFreeIsHonored(t *testing.T) {
	a, _ := newEpochTestArena(t, 1024)

	ptr, err := a.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	a.FreeAtEpoch(0, ptr, 256, a.Epoch())

	if got := a.StaleFrees(); got != 0 {
		t.Fatalf("StaleFrees = %d, want 0", got)
	}
	if got := a.FreeListLen(); got != 1 {
		t.Fatalf("free-list length = %d, want 1 (same-epoch free enters the free-list)", got)
	}
	// The freed block is reusable.
	ptr2, err := a.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc after free: %v", err)
	}
	if ptr2 != ptr {
		t.Fatalf("expected free-list reuse of the same block (got %p, want %p)", ptr2, ptr)
	}
}

// TestFreeAtEpoch_StaleFreeDoesNotPoisonLiveData is the poison-mode shape of
// the bug: verify8 saw the FORWARD read NaN because a stale FreeArena
// poison-filled memory owned by a live current-epoch tensor.
func TestFreeAtEpoch_StaleFreeDoesNotPoisonLiveData(t *testing.T) {
	restore := SetArenaPoisonEnabledForTesting(true)
	defer restore()
	SetArenaPoisonFill(HostPoisonFillForTesting)
	defer SetArenaPoisonFill(nil)

	a, buf := newEpochTestArena(t, 1024)

	e0 := a.Epoch()
	ptrA, err := a.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc A: %v", err)
	}
	a.Reset()

	// Live tensor B of the new epoch occupies A's bytes.
	ptrB, err := a.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc B: %v", err)
	}
	if ptrA != ptrB {
		t.Fatalf("test setup: expected B to reuse A's offset")
	}
	live := unsafe.Slice((*float32)(ptrB), 64)
	for i := range live {
		live[i] = float32(i + 1)
	}

	// Stale free under poison mode: must NOT NaN-fill B's live data.
	a.FreeAtEpoch(0, ptrA, 256, e0)

	for i := range live {
		if math.IsNaN(float64(buf[i])) || live[i] != float32(i+1) {
			t.Fatalf("stale free poisoned live data at [%d]=%v", i, live[i])
		}
	}
}

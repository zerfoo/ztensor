package cuda

import (
	"testing"
	"unsafe"
)

// swapArenaOverflowLogger replaces the package first-overflow sink for the
// duration of a test and returns a slice that records each call.
func swapArenaOverflowLogger(t *testing.T) *[]string {
	t.Helper()
	var calls []string
	orig := arenaOverflowFn
	arenaOverflowFn = func(_ ArenaDiagnostics, _, _ int, path string) {
		calls = append(calls, path)
	}
	t.Cleanup(func() { arenaOverflowFn = orig })
	return &calls
}

// TestArenaDiagnostics_Fields exercises a bump/reuse/reset sequence on a
// GPU-free arena (base is nil; bump pointers are never dereferenced) and
// asserts the diagnostics snapshot reports the expected counters (issue #118).
func TestArenaDiagnostics_Fields(t *testing.T) {
	// Back the arena with a real host buffer so the bump path's unsafe.Add stays
	// in-bounds under -race (checkptr). The pointers are never dereferenced; only
	// the offset accounting is asserted.
	buf := make([]byte, 4096)
	a := &ArenaPool{
		base:         unsafe.Pointer(&buf[0]),
		capacity:     len(buf),
		fallback:     NewMemPool(),
		fallbackPtrs: make(map[unsafe.Pointer]int),
	}

	// Two bump allocs: 256 and 1024 (aligned). offset advances by 256 + 1024.
	if _, err := a.Alloc(0, 200); err != nil { // aligns to 256
		t.Fatalf("alloc 1: %v", err)
	}
	if _, err := a.Alloc(0, 1000); err != nil { // aligns to 1024
		t.Fatalf("alloc 2: %v", err)
	}

	d := a.Diagnostics()
	if d.CapacityBytes != 4096 {
		t.Errorf("capacity: got %d, want 4096", d.CapacityBytes)
	}
	if d.OffsetBytes != 256+1024 {
		t.Errorf("offset: got %d, want %d", d.OffsetBytes, 256+1024)
	}
	if d.EpochAllocs != 2 {
		t.Errorf("epochAllocs: got %d, want 2", d.EpochAllocs)
	}
	if d.EpochMaxAllocBytes != 1024 {
		t.Errorf("epochMaxAllocBytes: got %d, want 1024", d.EpochMaxAllocBytes)
	}
	if d.Hits != 2 {
		t.Errorf("hits: got %d, want 2", d.Hits)
	}

	// Reset clears per-epoch diagnostics and bumps the reset counter.
	a.Reset()
	d = a.Diagnostics()
	if d.OffsetBytes != 0 {
		t.Errorf("offset after reset: got %d, want 0", d.OffsetBytes)
	}
	if d.EpochAllocs != 0 || d.EpochMaxAllocBytes != 0 {
		t.Errorf("epoch counters after reset: allocs=%d max=%d, want 0/0",
			d.EpochAllocs, d.EpochMaxAllocBytes)
	}
	if d.Resets != 1 {
		t.Errorf("resets: got %d, want 1", d.Resets)
	}
}

// TestArenaFirstOverflowLog_FiresOncePerEpoch verifies the first-overflow
// diagnostic fires exactly once per reset-epoch and re-arms after Reset
// (issue #118).
func TestArenaFirstOverflowLog_FiresOncePerEpoch(t *testing.T) {
	resetCaptureStateForTest(t)
	calls := swapArenaOverflowLogger(t)

	var sentinel byte
	stubArenaAsync(t, unsafe.Pointer(&sentinel), nil)

	// capacity 0 => every alloc overflows; overflow stream set => "async" path.
	a := newExhaustedArenaWithStream(&Stream{})

	for i := range 3 {
		if _, err := a.Alloc(0, 4096); err != nil {
			t.Fatalf("overflow alloc %d: %v", i, err)
		}
	}
	if len(*calls) != 1 {
		t.Fatalf("expected exactly one first-overflow log, got %d", len(*calls))
	}
	if (*calls)[0] != "async" {
		t.Errorf("path: got %q, want \"async\"", (*calls)[0])
	}

	// After Reset the diagnostic re-arms for the next epoch.
	a.Reset()
	if _, err := a.Alloc(0, 4096); err != nil {
		t.Fatalf("post-reset overflow alloc: %v", err)
	}
	if len(*calls) != 2 {
		t.Fatalf("expected the log to re-arm after Reset (2 total), got %d", len(*calls))
	}
}

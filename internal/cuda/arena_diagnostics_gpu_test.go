package cuda

import (
	"testing"
	"unsafe"
)

// TestArenaPool_DiagnosticsAndThreshold_GPU is the on-hardware (GB10) validation
// for issue #118. It (1) sets the stream-ordered pool release threshold (the
// async-overflow hardening) so cudaMallocAsync retains freed blocks instead of
// faulting them back from the OS, (2) overflows a small arena under repeated
// pressure through the async path, and (3) captures and prints the first-overflow
// diagnostic so the report (capacity/offset/epoch allocs/largest alloc) is
// observable on real hardware. A wedge -- the pre-#118 failure mode -- would hang
// before the final stream Synchronize, so reaching the end is the pass condition.
//
// Skips on CPU (no CUDA). Run on the GB10 via Spark under the dstate-watchdog --
// see docs/bench/manifests/issue-118-arena-diagnostics.yaml.
func TestArenaPool_DiagnosticsAndThreshold_GPU(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	const deviceID = 0
	if err := SetDevice(deviceID); err != nil {
		t.Fatalf("SetDevice: %v", err)
	}

	// Harden the async overflow pool: retain up to 512 MiB of freed blocks so
	// repeated overflow allocations are pool hits, not OS page faults (#118).
	if err := SetMemPoolReleaseThreshold(deviceID, 512<<20); err != nil {
		t.Fatalf("SetMemPoolReleaseThreshold: %v", err)
	}

	const arenaCap = 1 << 20 // 1 MiB: small so allocations overflow.
	fallback := NewMemPool()
	arena, err := NewArenaPool(deviceID, arenaCap, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	stream, err := CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()
	arena.SetOverflowStream(stream)

	// Capture the first-overflow diagnostic into the test log (and keep the
	// default stderr line via the engine path elsewhere). This is the report the
	// issue asks for: it says how full the arena was and via how many allocs.
	var firstOverflow ArenaDiagnostics
	var firstPath string
	captured := false
	orig := arenaOverflowFn
	arenaOverflowFn = func(d ArenaDiagnostics, requested, aligned int, path string) {
		if !captured {
			firstOverflow = d
			firstPath = path
			captured = true
			t.Logf("ARENA_FIRST_OVERFLOW path=%s requested=%d aligned=%d "+
				"capacity=%d offset=%d epochAllocs=%d epochMaxAlloc=%d "+
				"hits=%d misses=%d reuses=%d resets=%d freeList=%d",
				path, requested, aligned, d.CapacityBytes, d.OffsetBytes,
				d.EpochAllocs, d.EpochMaxAllocBytes, d.Hits, d.Misses,
				d.Reuses, d.Resets, d.FreeListLen)
		}
	}
	t.Cleanup(func() { arenaOverflowFn = orig })

	sizes := []int{48, 256, 12 << 10, 16 << 10, 1 << 20, 4 << 20}
	const rounds = 200

	type alloc struct {
		ptr  unsafe.Pointer
		size int
	}
	for r := range rounds {
		live := make([]alloc, 0, len(sizes))
		for _, sz := range sizes {
			p, err := arena.Alloc(deviceID, sz)
			if err != nil {
				t.Fatalf("round %d size %d: Alloc failed: %v", r, sz, err)
			}
			if p == nil {
				t.Fatalf("round %d size %d: Alloc returned nil", r, sz)
			}
			live = append(live, alloc{ptr: p, size: sz})
		}
		for _, a := range live {
			arena.Free(deviceID, a.ptr, a.size)
		}
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream Synchronize after %d rounds: %v", rounds, err)
	}
	if !captured {
		t.Fatal("expected the arena to overflow and fire the first-overflow diagnostic")
	}
	if firstPath != "async" {
		t.Errorf("first overflow path: got %q, want \"async\"", firstPath)
	}
	if firstOverflow.CapacityBytes != arenaCap {
		t.Errorf("diagnostic capacity: got %d, want %d", firstOverflow.CapacityBytes, arenaCap)
	}
}

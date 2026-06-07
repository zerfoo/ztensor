package cuda

import (
	"testing"
	"unsafe"
)

// TestArenaPool_OverflowStress_GPU is the on-hardware (GB10) validation for issue
// #115. It sets a real engine stream as the arena overflow stream, then issues
// many allocations that overflow a small arena -- mixing the tiny sizes seen in
// the #115 training frames (48 B .. 16 KB) with larger activation-sized blocks.
// Each overflow goes through cudaMallocAsync on the stream; PRE-FIX this path was
// a synchronous cudaMalloc that page-fault-thrashed GB10 unified memory and froze
// training. With the fix every alloc/free completes and the stream syncs cleanly.
//
// Skips on CPU (no CUDA). Run on the GB10 via Spark under the dstate-watchdog --
// see docs/bench/manifests/issue-115-overflow-stress.yaml.
func TestArenaPool_OverflowStress_GPU(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	const deviceID = 0
	if err := SetDevice(deviceID); err != nil {
		t.Fatalf("SetDevice: %v", err)
	}

	// A small arena so allocations overflow into the stream-ordered fallback.
	const arenaCap = 1 << 20 // 1 MiB
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

	// Sizes mirror the #115 frames (48 B, 12 KB, 16 KB) plus larger activations
	// that force the overflow past the 1 MiB arena.
	sizes := []int{48, 256, 12 << 10, 16 << 10, 1 << 20, 4 << 20}
	const rounds = 100

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

	// Syncing the stream proves every async alloc/free actually completed --
	// a wedge would never reach here (the test would hang, not pass).
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream Synchronize after %d rounds: %v", rounds, err)
	}
}

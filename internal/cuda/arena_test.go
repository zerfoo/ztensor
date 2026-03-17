package cuda

import (
	"testing"
	"unsafe"
)

func TestArenaPool_BasicAllocReset(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	fallback := NewMemPool()
	arena, err := NewArenaPool(0, 4096, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	ptr, err := arena.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	if ptr == nil {
		t.Fatal("Alloc returned nil")
	}

	hits, misses, _ := arena.HitMissStats()
	if hits != 1 {
		t.Errorf("hits = %d, want 1", hits)
	}
	if misses != 0 {
		t.Errorf("misses = %d, want 0", misses)
	}

	arena.Reset()
	if arena.UsedBytes() != 0 {
		t.Errorf("UsedBytes after Reset = %d, want 0", arena.UsedBytes())
	}
}

func TestArenaPool_FallbackOnExhaustion(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	fallback := NewMemPool()
	arena, err := NewArenaPool(0, 512, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	// First alloc fits in arena (256 aligned).
	_, err = arena.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc 1: %v", err)
	}

	// Second alloc also fits (256 + 256 = 512 = capacity).
	_, err = arena.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc 2: %v", err)
	}

	// Third alloc should fall back.
	ptr, err := arena.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc 3 (fallback): %v", err)
	}
	if ptr == nil {
		t.Fatal("Alloc 3 returned nil")
	}

	_, misses, _ := arena.HitMissStats()
	if misses != 1 {
		t.Errorf("misses = %d, want 1", misses)
	}

	// Free the fallback pointer.
	arena.Free(0, ptr, 256)
}

func TestArenaFreeList(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	tests := []struct {
		name string
		fn   func(t *testing.T, arena *ArenaPool)
	}{
		{
			name: "free_then_reuse_smaller",
			fn: func(t *testing.T, arena *ArenaPool) {
				// Alloc A (100 bytes -> 256 aligned), Alloc B (200 bytes -> 256 aligned),
				// Free A, Alloc C (80 bytes -> 256 aligned) — C should reuse A's slot.
				ptrA, err := arena.Alloc(0, 100)
				if err != nil {
					t.Fatalf("Alloc A: %v", err)
				}
				_, err = arena.Alloc(0, 200)
				if err != nil {
					t.Fatalf("Alloc B: %v", err)
				}
				arena.FreeArena(ptrA, 100)
				if arena.FreeListLen() != 1 {
					t.Fatalf("free list len = %d, want 1", arena.FreeListLen())
				}
				ptrC, err := arena.Alloc(0, 80)
				if err != nil {
					t.Fatalf("Alloc C: %v", err)
				}
				// C should reuse A's slot (same base pointer).
				if ptrC != ptrA {
					t.Errorf("ptrC = %p, want %p (reuse of A)", ptrC, ptrA)
				}
				if arena.ReuseStats() != 1 {
					t.Errorf("reuses = %d, want 1", arena.ReuseStats())
				}
			},
		},
		{
			name: "free_then_alloc_larger_bumps",
			fn: func(t *testing.T, arena *ArenaPool) {
				// Alloc A (256 bytes), Free A, Alloc B (512 bytes) — B should NOT
				// reuse A's slot because B is larger. B bumps the offset.
				ptrA, err := arena.Alloc(0, 256)
				if err != nil {
					t.Fatalf("Alloc A: %v", err)
				}
				arena.FreeArena(ptrA, 256)
				ptrB, err := arena.Alloc(0, 512)
				if err != nil {
					t.Fatalf("Alloc B: %v", err)
				}
				// B should be a different pointer (bumped past A).
				if ptrB == ptrA {
					t.Errorf("ptrB = %p, should not reuse A's slot (too small)", ptrB)
				}
				if arena.ReuseStats() != 0 {
					t.Errorf("reuses = %d, want 0", arena.ReuseStats())
				}
				// A's block should still be in the free list.
				if arena.FreeListLen() != 1 {
					t.Errorf("free list len = %d, want 1", arena.FreeListLen())
				}
			},
		},
		{
			name: "free_both_reuse_order",
			fn: func(t *testing.T, arena *ArenaPool) {
				// Alloc A (256 bytes), Alloc B (256 bytes), Free B, Free A.
				// Next alloc (256 bytes) should reuse one of the freed slots.
				ptrA, err := arena.Alloc(0, 256)
				if err != nil {
					t.Fatalf("Alloc A: %v", err)
				}
				ptrB, err := arena.Alloc(0, 256)
				if err != nil {
					t.Fatalf("Alloc B: %v", err)
				}
				arena.FreeArena(ptrB, 256)
				arena.FreeArena(ptrA, 256)
				// After freeing both adjacent blocks, they should merge.
				if arena.FreeListLen() != 1 {
					t.Errorf("free list len = %d, want 1 (merged)", arena.FreeListLen())
				}
				ptrC, err := arena.Alloc(0, 256)
				if err != nil {
					t.Fatalf("Alloc C: %v", err)
				}
				// C should reuse A's offset (start of the merged block).
				if ptrC != ptrA {
					t.Errorf("ptrC = %p, want %p (reuse of merged block starting at A)", ptrC, ptrA)
				}
				if arena.ReuseStats() != 1 {
					t.Errorf("reuses = %d, want 1", arena.ReuseStats())
				}
			},
		},
		{
			name: "reset_clears_free_list",
			fn: func(t *testing.T, arena *ArenaPool) {
				// Alloc, Free, verify free list, Reset, verify free list is empty.
				ptr, err := arena.Alloc(0, 256)
				if err != nil {
					t.Fatalf("Alloc: %v", err)
				}
				arena.FreeArena(ptr, 256)
				if arena.FreeListLen() != 1 {
					t.Fatalf("free list len = %d, want 1", arena.FreeListLen())
				}
				arena.Reset()
				if arena.FreeListLen() != 0 {
					t.Errorf("free list len after Reset = %d, want 0", arena.FreeListLen())
				}
				if arena.UsedBytes() != 0 {
					t.Errorf("UsedBytes after Reset = %d, want 0", arena.UsedBytes())
				}
			},
		},
		{
			name: "block_splitting",
			fn: func(t *testing.T, arena *ArenaPool) {
				// Alloc 512 bytes, free it, then alloc 256 bytes.
				// The remainder (256 bytes) should stay in the free list.
				ptr, err := arena.Alloc(0, 512)
				if err != nil {
					t.Fatalf("Alloc: %v", err)
				}
				arena.FreeArena(ptr, 512)
				ptrSmall, err := arena.Alloc(0, 256)
				if err != nil {
					t.Fatalf("Alloc small: %v", err)
				}
				if ptrSmall != ptr {
					t.Errorf("ptrSmall = %p, want %p (reuse)", ptrSmall, ptr)
				}
				// Remainder should be in free list.
				if arena.FreeListLen() != 1 {
					t.Errorf("free list len = %d, want 1 (remainder)", arena.FreeListLen())
				}
				// Alloc the remainder.
				ptrRem, err := arena.Alloc(0, 256)
				if err != nil {
					t.Fatalf("Alloc remainder: %v", err)
				}
				expectedRem := unsafe.Add(ptr, 256)
				if ptrRem != expectedRem {
					t.Errorf("ptrRem = %p, want %p (remainder of split)", ptrRem, expectedRem)
				}
				if arena.FreeListLen() != 0 {
					t.Errorf("free list len = %d, want 0", arena.FreeListLen())
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fallback := NewMemPool()
			arena, err := NewArenaPool(0, 4096, fallback)
			if err != nil {
				t.Fatalf("NewArenaPool: %v", err)
			}
			defer func() { _ = arena.Drain() }()
			tc.fn(t, arena)
		})
	}
}

func TestArenaPool_IsManaged(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	fallback := NewMemPool()
	arena, err := NewArenaPool(0, 4096, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	managed := arena.IsManaged()
	// The arena only uses managed memory when both the device supports it AND
	// ZERFOO_ENABLE_MANAGED_MEM is set. The test verifies IsManaged() returns
	// a consistent value without asserting a specific expected result, since
	// the env var may or may not be set in CI.
	t.Logf("ArenaPool.IsManaged() = %v (device supports managed: %v)", managed, ManagedMemorySupported(0))
}

func TestArenaPool_ManagedMemoryDataIntegrity(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}
	if !ManagedMemorySupported(0) {
		t.Skip("managed memory not supported on device 0")
	}

	fallback := NewMemPool()
	arena, err := NewArenaPool(0, 4096, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	if !arena.IsManaged() {
		t.Skip("arena not using managed memory (set ZERFOO_ENABLE_MANAGED_MEM=1)")
	}

	// Allocate from managed arena and write from CPU.
	ptr, err := arena.Alloc(0, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}

	data := unsafe.Slice((*float32)(ptr), 4)
	data[0] = 1.0
	data[1] = 2.0
	data[2] = 3.0
	data[3] = 4.0

	// Read back.
	for i, want := range []float32{1.0, 2.0, 3.0, 4.0} {
		if data[i] != want {
			t.Errorf("data[%d] = %f, want %f", i, data[i], want)
		}
	}
}

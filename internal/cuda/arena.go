package cuda

import (
	"fmt"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"unsafe"
)

// freeBlock represents a freed region within the arena that can be reused.
type freeBlock struct {
	offset int // byte offset from arena base
	size   int // aligned size in bytes
}

var defaultArenaInst *ArenaPool

// DefaultArenaPool returns the process-wide ArenaPool singleton, or nil.
func DefaultArenaPool() *ArenaPool {
	return defaultArenaInst
}

// SetDefaultArenaPool registers an ArenaPool as the process-wide default.
func SetDefaultArenaPool(a *ArenaPool) {
	defaultArenaInst = a
}

// ArenaPool is a bump-pointer allocator backed by a single large CUDA
// allocation. Each Alloc advances the offset within the pre-allocated region.
// Free is a no-op for individual pointers. Call Reset() between forward passes
// to reclaim all arena memory at once (zero-cost compared to per-pointer free).
//
// This eliminates cudaMalloc/cudaFree overhead during inference, which is the
// #1 bottleneck for per-token latency on the DGX Spark GPU.
//
// On devices with concurrent managed memory support (e.g., GB10 with
// NVLink-C2C and shared LPDDR5x), the arena is allocated with
// cudaMallocManaged. This makes the arena accessible from both CPU and GPU
// without explicit H2D copies.
//
// Weight tensors and KV cache should NOT use the arena (they persist across
// passes). The arena is only for per-pass intermediates.
type ArenaPool struct {
	mu         sync.Mutex
	base       unsafe.Pointer // start of the CUDA allocation
	capacity   int            // total bytes allocated
	offset     int            // current bump offset
	resetFloor int            // minimum offset after Reset (protects captured graph buffers)
	deviceID   int
	managed    bool // true if base was allocated with cudaMallocManaged
	hits       atomic.Int64
	misses     atomic.Int64 // only incremented if arena is full and falls back
	resets     atomic.Int64
	reuses     atomic.Int64 // incremented when Alloc reuses a free-list block

	// freeList holds freed blocks sorted by offset, available for reuse.
	// Alloc checks for a best-fit block before bumping the offset.
	freeList []freeBlock

	// fallback is the MemPool used when the arena is exhausted.
	fallback *MemPool
	// fallbackPtrs tracks pointers allocated from fallback so Free can route them.
	fallbackPtrs map[unsafe.Pointer]int // ptr -> byteSize

	// overflowStream, when set, makes the exhaustion fallback use stream-ordered
	// cudaMallocAsync instead of a synchronous cudaMalloc. A synchronous
	// cudaMalloc under memory pressure page-fault-thrashes GB10 unified memory
	// and wedges training; the stream-ordered pool does not. See issue #115 and
	// docs/adr/005-stream-ordered-arena-overflow.md.
	overflowStream *Stream
	// asyncFallbackPtrs tracks pointers allocated via the async overflow path so
	// Free routes them to cudaFreeAsync on the overflow stream. ptr -> byteSize.
	asyncFallbackPtrs map[unsafe.Pointer]int
}

// arenaMallocAsyncFn and arenaFreeAsyncFn are indirection points for the
// stream-ordered arena overflow path (issue #115). Tests swap them to assert
// routing decisions without a CUDA device.
var (
	arenaMallocAsyncFn = MallocAsync
	arenaFreeAsyncFn   = FreeAsync
)

// NewArenaPool allocates a contiguous GPU region of the given capacity bytes
// on the specified device. A fallback MemPool handles any overflow.
// On devices with concurrent managed memory support, the arena uses
// cudaMallocManaged to enable zero-copy CPU/GPU access.
func NewArenaPool(deviceID, capacityBytes int, fallback *MemPool) (*ArenaPool, error) {
	if err := SetDevice(deviceID); err != nil {
		return nil, fmt.Errorf("ArenaPool: SetDevice: %w", err)
	}

	managed := ManagedMemorySupported(deviceID) && os.Getenv("ZERFOO_ENABLE_MANAGED_MEM") != ""

	var ptr unsafe.Pointer
	var err error
	if managed {
		ptr, err = MallocManaged(capacityBytes)
		if err != nil {
			// Fall back to regular malloc if managed allocation fails.
			managed = false
			ptr, err = Malloc(capacityBytes)
		}
	} else {
		ptr, err = Malloc(capacityBytes)
	}
	if err != nil {
		return nil, fmt.Errorf("ArenaPool: alloc(%d): %w", capacityBytes, err)
	}

	return &ArenaPool{
		base:         ptr,
		capacity:     capacityBytes,
		deviceID:     deviceID,
		managed:      managed,
		fallback:     fallback,
		fallbackPtrs: make(map[unsafe.Pointer]int),
	}, nil
}

// IsManaged returns true if the arena was allocated with managed memory.
func (a *ArenaPool) IsManaged() bool {
	return a.managed
}

// Alloc returns a device pointer of at least byteSize bytes from the arena.
// Allocations are 256-byte aligned for GPU coalescing. Alloc first checks
// the free-list for a best-fit reusable block. If none is found, it bumps
// the offset. If the arena is full, falls back to the MemPool.
func (a *ArenaPool) Alloc(deviceID, byteSize int) (unsafe.Pointer, error) {
	// Align to 256 bytes for GPU memory access patterns.
	aligned := (byteSize + 255) &^ 255

	a.mu.Lock()

	// Check free-list for a best-fit block (smallest block >= aligned).
	if bestIdx := a.findBestFit(aligned); bestIdx >= 0 {
		blk := a.freeList[bestIdx]
		// Remove the block from the free-list.
		a.freeList = append(a.freeList[:bestIdx], a.freeList[bestIdx+1:]...)
		// If the block is larger than needed, return the remainder to the free-list.
		if remainder := blk.size - aligned; remainder >= 256 {
			a.insertFreeBlock(freeBlock{offset: blk.offset + aligned, size: remainder})
		}
		ptr := unsafe.Add(a.base, blk.offset)
		a.mu.Unlock()
		a.hits.Add(1)
		a.reuses.Add(1)
		return ptr, nil
	}

	if a.offset+aligned <= a.capacity {
		ptr := unsafe.Add(a.base, a.offset)
		a.offset += aligned
		a.mu.Unlock()
		a.hits.Add(1)
		return ptr, nil
	}
	a.mu.Unlock()

	// Arena exhausted -- fall back to MemPool.
	a.misses.Add(1)

	// Capture-aware guard (issue #111, ADR 004): if a CUDA graph capture is
	// active and the fallback is NOT capture-aware, it would issue a synchronous
	// cudaMalloc, which hangs the GB10 driver mid-capture. Refuse instead so the
	// caller can fall back to CPU or fail the capture cleanly. When the fallback
	// IS capture-aware (engine-driven BeginCapture set its capture stream) the
	// async malloc is graph-safe, so this path is preserved.
	if CaptureActive() && !a.fallback.IsCapturing() {
		return nil, ErrCaptureUnsafeAlloc
	}

	// Stream-ordered overflow (issue #115, ADR 005): when an overflow stream is
	// set and the fallback is not in capture-aware mode (engine-driven capture,
	// which already routes through its capture stream), allocate via
	// cudaMallocAsync on the overflow stream instead of a synchronous cudaMalloc.
	// On GB10 unified memory a synchronous cudaMalloc under pressure
	// page-fault-thrashes and freezes training; the stream-ordered pool does not.
	a.mu.Lock()
	overflow := a.overflowStream
	a.mu.Unlock()
	if overflow != nil && !a.fallback.IsCapturing() {
		ptr, err := arenaMallocAsyncFn(aligned, overflow)
		if err != nil {
			return nil, err
		}
		a.mu.Lock()
		if a.asyncFallbackPtrs == nil {
			a.asyncFallbackPtrs = make(map[unsafe.Pointer]int)
		}
		a.asyncFallbackPtrs[ptr] = byteSize
		a.mu.Unlock()
		return ptr, nil
	}

	ptr, err := a.fallback.Alloc(deviceID, byteSize)
	if err != nil {
		return nil, err
	}
	a.mu.Lock()
	a.fallbackPtrs[ptr] = byteSize
	a.mu.Unlock()
	return ptr, nil
}

// findBestFit returns the index of the smallest free-list block that is
// >= needed bytes, or -1 if no block fits. Caller must hold a.mu.
func (a *ArenaPool) findBestFit(needed int) int {
	bestIdx := -1
	bestSize := int(^uint(0) >> 1) // max int
	for i, blk := range a.freeList {
		if blk.size >= needed && blk.size < bestSize {
			bestIdx = i
			bestSize = blk.size
			if blk.size == needed {
				break // exact fit
			}
		}
	}
	return bestIdx
}

// insertFreeBlock adds a block to the free-list, maintaining sort order by
// offset and merging with adjacent blocks. Caller must hold a.mu.
func (a *ArenaPool) insertFreeBlock(blk freeBlock) {
	// Find insertion point (sorted by offset).
	pos := sort.Search(len(a.freeList), func(i int) bool {
		return a.freeList[i].offset >= blk.offset
	})
	// Insert at pos.
	a.freeList = append(a.freeList, freeBlock{})
	copy(a.freeList[pos+1:], a.freeList[pos:])
	a.freeList[pos] = blk

	// Merge with next block if adjacent.
	if pos+1 < len(a.freeList) && a.freeList[pos].offset+a.freeList[pos].size == a.freeList[pos+1].offset {
		a.freeList[pos].size += a.freeList[pos+1].size
		a.freeList = append(a.freeList[:pos+1], a.freeList[pos+2:]...)
	}
	// Merge with previous block if adjacent.
	if pos > 0 && a.freeList[pos-1].offset+a.freeList[pos-1].size == a.freeList[pos].offset {
		a.freeList[pos-1].size += a.freeList[pos].size
		a.freeList = append(a.freeList[:pos], a.freeList[pos+1:]...)
	}
}

// Free returns an arena pointer to the free-list for intra-pass reuse.
// Fallback pointers are returned to the MemPool.
func (a *ArenaPool) Free(deviceID int, ptr unsafe.Pointer, byteSize int) {
	a.mu.Lock()
	// Stream-ordered overflow pointers (issue #115) free via cudaFreeAsync on the
	// overflow stream so the free is ordered after the kernels that used them.
	if _, ok := a.asyncFallbackPtrs[ptr]; ok {
		delete(a.asyncFallbackPtrs, ptr)
		overflow := a.overflowStream
		a.mu.Unlock()
		_ = arenaFreeAsyncFn(ptr, overflow)
		return
	}
	if _, ok := a.fallbackPtrs[ptr]; ok {
		delete(a.fallbackPtrs, ptr)
		a.mu.Unlock()
		a.fallback.Free(deviceID, ptr, byteSize)
		return
	}
	a.mu.Unlock()

	// Arena pointer: add to free-list for intra-pass reuse.
	a.FreeArena(ptr, byteSize)
}

// FreeArena returns an arena allocation to the free-list. The pointer must
// have been returned by a previous Alloc from this arena (not from fallback).
// The byteSize must match the original allocation request (it will be aligned
// to 256 bytes internally). This enables intra-pass buffer reuse: freed
// regions can be reclaimed by subsequent Alloc calls before the next Reset.
func (a *ArenaPool) FreeArena(ptr unsafe.Pointer, byteSize int) {
	if ptr == nil || byteSize <= 0 {
		return
	}
	aligned := (byteSize + 255) &^ 255
	offset := int(uintptr(ptr) - uintptr(a.base))
	if offset < 0 || offset+aligned > a.capacity {
		return // not an arena pointer
	}
	a.mu.Lock()
	a.insertFreeBlock(freeBlock{offset: offset, size: aligned})
	a.mu.Unlock()
}

// AllocManaged delegates to the fallback MemPool (arena is device-only).
func (a *ArenaPool) AllocManaged(deviceID, byteSize int) (unsafe.Pointer, error) {
	return a.fallback.AllocManaged(deviceID, byteSize)
}

// FreeManaged delegates to the fallback MemPool.
func (a *ArenaPool) FreeManaged(deviceID int, ptr unsafe.Pointer, byteSize int) {
	a.fallback.FreeManaged(deviceID, ptr, byteSize)
}

// Reset rewinds the arena offset to the reset floor (default 0), reclaiming
// per-pass allocations while preserving buffers below the floor (e.g. CUDA
// graph captured buffers). The free-list is also cleared since all arena
// memory above the floor is reclaimed.
func (a *ArenaPool) Reset() {
	a.mu.Lock()
	a.offset = a.resetFloor
	a.freeList = a.freeList[:0]
	a.mu.Unlock()
	a.resets.Add(1)
}

// SetOverflowStream sets the stream used for the arena's exhaustion fallback.
// When set, an exhausted Alloc that is not refused by the capture guard
// allocates via cudaMallocAsync on this stream (stream-ordered, non-blocking)
// instead of a synchronous cudaMalloc, which thrashes GB10 unified memory under
// pressure (issue #115, ADR 005). The stream MUST be the one the consuming
// kernels launch on, so the allocation is ordered before its uses.
func (a *ArenaPool) SetOverflowStream(s *Stream) {
	a.mu.Lock()
	a.overflowStream = s
	a.mu.Unlock()
}

// SetResetFloor sets the minimum offset that Reset will rewind to. Allocations
// below this offset are preserved across resets. This is used by CUDA graph
// capture to protect GPU buffers that the captured graph references.
func (a *ArenaPool) SetResetFloor(floor int) {
	a.mu.Lock()
	a.resetFloor = floor
	a.mu.Unlock()
}

// Drain frees the underlying CUDA allocation and drains the fallback pool.
func (a *ArenaPool) Drain() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	var firstErr error
	if a.base != nil {
		if err := SetDevice(a.deviceID); err != nil && firstErr == nil {
			firstErr = err
		}
		if err := Free(a.base); err != nil && firstErr == nil {
			firstErr = err
		}
		a.base = nil
		a.offset = 0
		a.capacity = 0
		a.freeList = nil
	}

	if err := a.fallback.Drain(); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

// Stats returns the arena utilization and fallback pool stats.
func (a *ArenaPool) Stats() (allocations int, totalBytes int) {
	a.mu.Lock()
	arenaUsed := a.offset
	a.mu.Unlock()
	fbAllocs, fbBytes := a.fallback.Stats()
	return 1 + fbAllocs, arenaUsed + fbBytes
}

// HitMissStats returns arena hits, fallback misses, reset count, and
// free-list reuse count.
func (a *ArenaPool) HitMissStats() (hits, misses, resets int64) {
	return a.hits.Load(), a.misses.Load(), a.resets.Load()
}

// ReuseStats returns the number of allocations served from the free-list.
func (a *ArenaPool) ReuseStats() int64 {
	return a.reuses.Load()
}

// FreeListLen returns the current number of blocks in the free-list.
func (a *ArenaPool) FreeListLen() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.freeList)
}

// UsedBytes returns the current arena offset (bytes in use).
func (a *ArenaPool) UsedBytes() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.offset
}

// Capacity returns the total arena capacity in bytes.
func (a *ArenaPool) Capacity() int {
	return a.capacity
}

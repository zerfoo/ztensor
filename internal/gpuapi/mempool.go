package gpuapi

import "unsafe"

// MemPool abstracts a GPU device memory pool with size-bucketed caching.
// Each vendor can reuse the same pool logic since the pool operates on
// opaque device pointers, but the underlying Malloc/Free come from the
// vendor's Runtime.
type MemPool interface {
	// Alloc returns a device pointer of at least byteSize bytes on the given device.
	// May return a cached pointer from a previous Free call.
	Alloc(deviceID, byteSize int) (unsafe.Pointer, error)
	// Free returns a device pointer to the pool for reuse.
	Free(deviceID int, ptr unsafe.Pointer, byteSize int)
	// AllocManaged returns a unified memory pointer accessible from both host
	// and device. Returns an error on backends that do not support managed memory.
	AllocManaged(deviceID, byteSize int) (unsafe.Pointer, error)
	// FreeManaged returns a managed memory pointer to the pool for reuse.
	FreeManaged(deviceID int, ptr unsafe.Pointer, byteSize int)
	// Drain frees all cached pointers back to the device.
	Drain() error
	// Stats returns the number of cached allocations and their total bytes.
	Stats() (allocations int, totalBytes int)
}

// EpochMemPool is implemented by pools whose Reset reclaims allocations
// wholesale (arena/bump allocators). An allocation's lifetime cannot extend
// across a Reset, so a Free that arrives after the Reset that reclaimed it
// -- the canonical case is a Go GC finalizer on a dead GPUStorage firing one
// collection cycle late -- must be dropped: by then the same bytes belong to
// a live allocation of the current epoch, and honoring the stale free would
// poison live data and double-issue the block through the pool's free-list.
//
// Storage that frees lazily (finalizer-driven) captures Epoch() at
// allocation time and releases through FreeAtEpoch. Pools that never
// bulk-reclaim (bucketed MemPool) do not implement this interface and are
// freed unconditionally.
type EpochMemPool interface {
	// Epoch returns the pool's current reset epoch.
	Epoch() uint64
	// FreeAtEpoch frees ptr only if the pool has not been Reset since
	// allocEpoch; otherwise the free is dropped as stale.
	FreeAtEpoch(deviceID int, ptr unsafe.Pointer, byteSize int, allocEpoch uint64)
}

// CaptureAwareAllocator is implemented by memory pools that support
// CUDA graph capture. When capture mode is active, allocations use
// cudaMallocAsync on the capture stream instead of cudaMalloc on the
// default stream.
//
// Pools that do not implement this interface are unaffected by graph
// capture; all tensors must be pre-allocated before capture begins.
type CaptureAwareAllocator interface {
	// SetCaptureStream enables capture-aware allocation.
	// stream is the raw cudaStream_t (unsafe.Pointer).
	SetCaptureStream(stream unsafe.Pointer)
	// ClearCaptureStream disables capture-aware allocation.
	ClearCaptureStream()
	// IsCapturing returns true when capture-aware allocation is active
	// (i.e., SetCaptureStream has been called and not yet cleared).
	IsCapturing() bool
}

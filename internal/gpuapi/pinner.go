package gpuapi

import "unsafe"

// BackwardPinner is implemented by memory pools whose buffers can be pinned
// against reclamation (arena Reset, intra-pass reuse, poison fills) until
// unpinned (ADR 006, save-for-backward contract; issue #128).
//
// Pools that do not implement this interface have no reclamation that can
// outrun a live Go reference (e.g. size-bucketed MemPool buffers are only
// recycled on explicit Free), so pinning is a no-op for them.
type BackwardPinner interface {
	// PinBuffer marks the allocation starting at ptr as un-reclaimable.
	// byteSize is the original allocation request size. Pins are refcounted:
	// each successful PinBuffer must be balanced by exactly one UnpinBuffer.
	// Returns true if a pin was taken (ptr belongs to the pool's pinnable
	// region); false is a no-op (e.g. overflow-fallback pointers).
	PinBuffer(ptr unsafe.Pointer, byteSize int) bool
	// UnpinBuffer drops one pin reference. The buffer becomes reclaimable
	// when the last reference drops.
	UnpinBuffer(ptr unsafe.Pointer)
}

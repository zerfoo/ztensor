package tensor

import "github.com/zerfoo/ztensor/internal/gpuapi"

// PinnableStorage is implemented by storage backends whose underlying memory
// can be pinned against allocator reclamation (ADR 006 save-for-backward
// contract, issue #128). The graph pins tensors registered via
// SaveForBackward so arena Reset / intra-pass reuse cannot recycle them
// before the owning node's Backward has consumed them -- the zerfoo#842 /
// zerfoo#845 / Wolf QK-norm bug class.
//
// CPU storages do not implement this interface: their memory is owned by the
// Go GC, which cannot reclaim it while a reference is live, so saving is
// inherently safe there and pinning is a no-op.
type PinnableStorage interface {
	// PinForBackward pins the storage's allocation in its backing pool.
	// Returns true if a pin was taken; each true return must be balanced by
	// exactly one UnpinForBackward.
	PinForBackward() bool
	// UnpinForBackward drops one pin reference taken by PinForBackward. It
	// must be safe to call after the storage has been freed (the pool defers
	// the actual free until the last pin drops).
	UnpinForBackward()
}

// PinForBackward pins this storage's device allocation in its backing pool,
// if the pool supports pinning (arena-backed pools do; bucketed MemPool and
// non-pooled storage do not, and return false: their memory is never
// recycled behind a live reference).
//
// The pinned pointer is captured so UnpinForBackward stays balanced even if
// the storage is freed between pin and unpin (e.g. the graph executor's
// refcount release path frees a saved intermediate mid-step; the pool defers
// the free until the pin drops, but Free zeroes devicePtr). The storage must
// not be reallocated (TrySet with a different length) while pinned.
func (s *GPUStorage[T]) PinForBackward() bool {
	if s.devicePtr == nil || s.byteSize == 0 || s.pool == nil {
		return false
	}
	pinner, ok := s.pool.(gpuapi.BackwardPinner)
	if !ok {
		return false
	}
	size := s.allocSize
	if size == 0 {
		size = s.byteSize
	}
	if !pinner.PinBuffer(s.devicePtr, size) {
		return false
	}
	s.pinnedPtr = s.devicePtr
	return true
}

// UnpinForBackward drops one pin reference taken by a successful
// PinForBackward. Safe after Free: it targets the pointer captured at pin
// time, not the (possibly zeroed) live device pointer.
func (s *GPUStorage[T]) UnpinForBackward() {
	if s.pinnedPtr == nil || s.pool == nil {
		return
	}
	if pinner, ok := s.pool.(gpuapi.BackwardPinner); ok {
		pinner.UnpinBuffer(s.pinnedPtr)
	}
}

// Statically assert GPUStorage satisfies PinnableStorage.
var _ PinnableStorage = (*GPUStorage[float32])(nil)

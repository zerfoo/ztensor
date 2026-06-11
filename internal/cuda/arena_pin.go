package cuda

import (
	"fmt"
	"os"
	"sort"
	"unsafe"
)

// Buffer pinning (ADR 006 decisions 1-2; zerfoo
// docs/plan-gpu-training-hardening.md T2.2, issue #128).
//
// The save-for-backward contract makes saved forward intermediates
// un-reclaimable until backward consumes them. The arena side of that
// contract is Pin/Unpin: a pinned span survives Reset, is never handed out
// for reuse, and is never poisoned (ADR 006 decision 4) until the last
// Unpin releases it.
//
// Reset-vs-pinned semantics (the "raise the floor" option): Reset rewinds
// the bump offset to the highest pinned byte instead of the reset floor
// when pins are held. This is the simplest provably-safe option: nothing
// at or below the rewound offset can be re-issued by the bump allocator,
// and the free-list is cleared as before, so no alias of a pinned span can
// exist. The watermark consequence is that ALL bytes between the reset
// floor and the highest pinned byte -- including dead, unpinned buffers
// allocated before the pinned one -- stay retained until the pins are
// released; the next Reset after the last Unpin reclaims (and, under
// poison mode, poisons) the whole retained span. Pinned memory is bounded
// by what backward genuinely needs (see the ADR), so the raised watermark
// is accepted in exchange for the simpler invariant.

// pinSpan is the refcount entry for one pinned region, keyed in
// ArenaPool.pins by its byte offset from the arena base.
type pinSpan struct {
	size int // aligned span size in bytes
	refs int // outstanding Pin calls for this offset
}

// arenaPinWarnFn sinks pin-misuse warnings (refcount underflow, Unpin of an
// unknown pointer). Default writes a single stderr line, mirroring
// arenaPoisonWarnFn; tests swap it to capture. Misuse is never a panic in
// production paths: an unbalanced Unpin loses (at worst) some debug
// protection, it must not kill a training run.
var arenaPinWarnFn = func(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "ztensor arena pin: "+format+"\n", args...)
}

// Pin marks the arena span starting at ptr as un-reclaimable: Reset will not
// rewind past it, free-list reuse cannot reach it (a FreeArena on a pinned
// span is deferred), and poison mode will not fill it. byteSize is the
// original allocation request size (aligned to the arena's 256-byte quantum
// internally, mirroring FreeArena). Pins are refcounted per offset: each Pin
// must be balanced by exactly one Unpin, and the span is only released when
// the last reference drops.
//
// Returns true if the pointer lies inside this arena and the pin was taken.
// Pointers outside the arena (fallback/cudaMalloc allocations, CPU memory)
// return false and are a no-op: those allocations are not subject to
// Reset-based reclamation, so they need no pin.
func (a *ArenaPool) Pin(ptr unsafe.Pointer, byteSize int) bool {
	if ptr == nil || byteSize <= 0 {
		return false
	}
	aligned := (byteSize + 255) &^ 255

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.base == nil {
		return false
	}
	offset := int(uintptr(ptr) - uintptr(a.base))
	if offset < 0 || offset >= a.capacity {
		return false // not an arena pointer
	}
	if offset+aligned > a.capacity {
		aligned = a.capacity - offset // clip to the arena (mirrors FreeArena's bound check)
	}

	if a.pins == nil {
		a.pins = make(map[int]*pinSpan)
	}
	if p := a.pins[offset]; p != nil {
		p.refs++
		// Re-pin with a larger size grows the protected span (e.g. a view pin
		// followed by a full-allocation pin). The max wins; over-protection is
		// safe, under-protection is not.
		if aligned > p.size {
			a.pinnedBytes += aligned - p.size
			p.size = aligned
		}
	} else {
		a.pins[offset] = &pinSpan{size: aligned, refs: 1}
		a.pinnedBytes += aligned
	}
	if a.pinnedBytes > a.pinnedHighWater {
		a.pinnedHighWater = a.pinnedBytes
	}
	return true
}

// Unpin drops one reference from the pin at ptr. When the last reference
// drops, the span becomes reclaimable again and any FreeArena that was
// deferred while the span was pinned is applied -- including its poison fill
// under ZTENSOR_ARENA_POISON=1, exactly as if the free had happened then.
//
// Unpin of a pointer that was never pinned (refcount underflow) logs a
// warning and returns; it never panics. Unpin of a non-arena pointer is a
// silent no-op, matching Pin's false return for the same pointer.
func (a *ArenaPool) Unpin(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}
	a.mu.Lock()
	if a.base == nil {
		a.mu.Unlock()
		return
	}
	offset := int(uintptr(ptr) - uintptr(a.base))
	if offset < 0 || offset >= a.capacity {
		a.mu.Unlock()
		return // not an arena pointer; Pin returned false for it too
	}
	p := a.pins[offset]
	if p == nil {
		a.mu.Unlock()
		arenaPinWarnFn("Unpin without matching Pin at arena offset %d", offset)
		return
	}
	p.refs--
	if p.refs > 0 {
		a.mu.Unlock()
		return
	}
	delete(a.pins, offset)
	a.pinnedBytes -= p.size

	// Apply deferred frees now that no pin covers them. Poison-then-insert
	// mirrors FreeArena's ordering: the fill happens before the block enters
	// the free-list, so no allocator path can hand it out mid-fill. The whole
	// release runs under a.mu (like Reset's poison) so a concurrent Reset
	// cannot rewind underneath it. Every deferred block is below the current
	// bump offset: Reset clears deferredFrees, so anything here was deferred
	// after the last Reset, i.e. inside the currently-allocated span.
	for _, blk := range a.takeUnblockedDeferredFreesLocked() {
		if arenaPoisonEnabled {
			a.poisonRegion(unsafe.Add(a.base, blk.offset), blk.size, "Unpin(deferred FreeArena)")
		}
		a.insertFreeBlock(blk)
	}
	a.mu.Unlock()
}

// PinnedBytes returns the bytes currently pinned (sum of pinned span sizes,
// counting each span once regardless of its refcount).
func (a *ArenaPool) PinnedBytes() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.pinnedBytes
}

// PinnedHighWaterBytes returns the maximum PinnedBytes observed over the
// arena's lifetime -- the monitoring number for "how much extra arena does
// the save-for-backward contract cost" (ADR 006 consequence).
func (a *ArenaPool) PinnedHighWaterBytes() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.pinnedHighWater
}

// overlapsPinLocked reports whether [offset, offset+size) intersects any
// pinned span. Caller must hold a.mu. Overlap (not exact-offset equality) is
// used so a pin taken on an interior view pointer still defers a FreeArena
// of the enclosing allocation.
func (a *ArenaPool) overlapsPinLocked(offset, size int) bool {
	for off, p := range a.pins {
		if offset < off+p.size && off < offset+size {
			return true
		}
	}
	return false
}

// takeUnblockedDeferredFreesLocked removes and returns the deferred frees
// that no longer overlap any pinned span. Caller must hold a.mu.
func (a *ArenaPool) takeUnblockedDeferredFreesLocked() []freeBlock {
	if len(a.deferredFrees) == 0 {
		return nil
	}
	var released []freeBlock
	kept := a.deferredFrees[:0]
	for _, blk := range a.deferredFrees {
		if a.overlapsPinLocked(blk.offset, blk.size) {
			kept = append(kept, blk)
		} else {
			released = append(released, blk)
		}
	}
	a.deferredFrees = kept
	return released
}

// pinnedFloorLocked returns the Reset rewind target: the reset floor, raised
// to the end of the highest pinned span ("raise the floor", see the package
// comment above). Caller must hold a.mu.
func (a *ArenaPool) pinnedFloorLocked() int {
	floor := a.resetFloor
	for off, p := range a.pins {
		if end := off + p.size; end > floor {
			floor = end
		}
	}
	return floor
}

// poisonUnpinnedSpanLocked poisons [lo, hi) except the parts covered by
// pinned spans, preserving the ADR 006 decision-4 guarantee ("pinned regions
// are never poisoned") without weakening poison coverage of the dead bytes
// retained below a raised floor. Caller must hold a.mu.
func (a *ArenaPool) poisonUnpinnedSpanLocked(lo, hi int) {
	if hi <= lo || a.base == nil {
		return
	}
	spans := make([]freeBlock, 0, len(a.pins))
	for off, p := range a.pins {
		spans = append(spans, freeBlock{offset: off, size: p.size})
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].offset < spans[j].offset })

	cur := lo
	for _, s := range spans {
		end := s.offset + s.size
		if end <= cur || s.offset >= hi {
			continue
		}
		if s.offset > cur {
			a.poisonRegion(unsafe.Add(a.base, cur), s.offset-cur, "Reset")
		}
		if end > cur {
			cur = end
		}
	}
	if cur < hi {
		a.poisonRegion(unsafe.Add(a.base, cur), hi-cur, "Reset")
	}
}

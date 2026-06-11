package cuda

import "unsafe"

// Testing support for packages that need an ArenaPool without a CUDA device
// (e.g. the graph package's save-for-backward regression tests, which drive
// a real arena through the Forward -> Reset -> Backward hazard schedule).
// Living in internal/ keeps these hooks out of the public module API; they
// must never be called from production paths.

// NewHostBackedArenaForTesting builds an ArenaPool over a caller-owned host
// buffer so pointers returned by Alloc are dereferenceable without a GPU
// (same pattern the poison-mode tests established). The arena keeps the
// buffer alive via its base pointer. Drain must not be called: the buffer is
// not a CUDA allocation.
func NewHostBackedArenaForTesting(buf []byte) *ArenaPool {
	return &ArenaPool{
		base:         unsafe.Pointer(&buf[0]),
		capacity:     len(buf),
		fallback:     NewMemPool(),
		fallbackPtrs: make(map[unsafe.Pointer]int),
	}
}

// SetArenaPoisonEnabledForTesting flips the poison mode, which is normally
// read once from ZTENSOR_ARENA_POISON at process init, and returns a func
// that restores the previous value.
func SetArenaPoisonEnabledForTesting(enabled bool) (restore func()) {
	orig := arenaPoisonEnabled
	arenaPoisonEnabled = enabled
	return func() { arenaPoisonEnabled = orig }
}

// HostPoisonFillForTesting is an ArenaPoisonFillFunc that writes the
// production poison byte pattern directly into host memory -- valid only for
// host-backed test arenas. Install it via SetArenaPoisonFill.
func HostPoisonFillForTesting(ptr unsafe.Pointer, byteLen int) error {
	fillHostPoison(unsafe.Slice((*byte)(ptr), byteLen))
	return nil
}

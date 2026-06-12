package tensor

import "sync"

// Host-access stream ordering contract.
//
// GPU compute engines launch kernels asynchronously on their own streams.
// GPUStorage's host-access paths (TrySlice / Slice / CopyTo / TrySet / Set)
// historically relied on the implicit ordering of synchronous cudaMemcpy
// against the legacy default stream. That implicit ordering is NOT guaranteed
// on cache-coherent unified-memory platforms (Tegra, GH200, GB10 NVLink-C2C):
// there a "synchronous" pageable-memory copy may be serviced by the CPU
// without waiting for kernels pending on a non-default stream, and managed
// storage is accessed directly through the unified pointer with no CUDA call
// at all. The result is a host read observing bytes from BEFORE a still-async
// kernel write (or a host write landing under a pending kernel read) -- the
// Wolf CrossAsset "batch-3 NaN" gradient corruption (zerfoo#850 lineage,
// Bug 11).
//
// The contract fix: a host access to GPU storage must be stream-ordered with
// respect to pending device work that touches it. Compute engines register a
// synchronization hook per device (RegisterHostAccessSync); every GPUStorage
// host-access path invokes the hooks for its device before touching memory.
// The tensor package cannot import compute (compute imports tensor), hence
// this registration indirection -- the same pattern as the arena poison
// kernel fill hook.
//
// Multiple engines may coexist on one device (each with its own stream); all
// registered hooks for the device run. Devices with no registered hook (CPU
// tests, storage created before any engine) sync nothing, preserving the old
// behavior.

// hostAccessSyncRegistry holds the per-device synchronization hooks.
type hostAccessSyncRegistry struct {
	mu    sync.RWMutex
	next  int
	hooks map[int]map[int]func() error // deviceID -> hookID -> hook
}

var hostAccessSync hostAccessSyncRegistry

// RegisterHostAccessSync registers fn as a host-access synchronization hook
// for deviceID. Every GPUStorage host-access path (TrySlice, Slice, CopyTo,
// TrySet, Set) on that device invokes fn before touching memory; fn must
// block until device work pending against that device's storage is complete
// (typically stream.Synchronize() on the engine's stream). The returned
// function unregisters the hook; engines must call it on Close so a dead
// stream is never synchronized.
func RegisterHostAccessSync(deviceID int, fn func() error) (unregister func()) {
	hostAccessSync.mu.Lock()
	defer hostAccessSync.mu.Unlock()

	if hostAccessSync.hooks == nil {
		hostAccessSync.hooks = make(map[int]map[int]func() error)
	}
	if hostAccessSync.hooks[deviceID] == nil {
		hostAccessSync.hooks[deviceID] = make(map[int]func() error)
	}
	id := hostAccessSync.next
	hostAccessSync.next++
	hostAccessSync.hooks[deviceID][id] = fn

	return func() {
		hostAccessSync.mu.Lock()
		defer hostAccessSync.mu.Unlock()
		delete(hostAccessSync.hooks[deviceID], id)
	}
}

// syncForHostAccess runs every registered host-access synchronization hook
// for deviceID, returning the first error. Called by GPUStorage host-access
// paths before reading or writing device memory from the host.
func syncForHostAccess(deviceID int) error {
	hostAccessSync.mu.RLock()
	hooks := make([]func() error, 0, len(hostAccessSync.hooks[deviceID]))
	for _, fn := range hostAccessSync.hooks[deviceID] {
		hooks = append(hooks, fn)
	}
	hostAccessSync.mu.RUnlock()

	for _, fn := range hooks {
		if err := fn(); err != nil {
			return err
		}
	}
	return nil
}

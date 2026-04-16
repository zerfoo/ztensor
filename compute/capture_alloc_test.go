package compute

import (
	"errors"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/internal/gpuapi"
)

// --- fake CaptureAwareAllocator pool for tests ---

type fakeCapturePool struct {
	capturing bool
}

func (p *fakeCapturePool) Alloc(int, int) (unsafe.Pointer, error)          { return nil, nil }
func (p *fakeCapturePool) Free(int, unsafe.Pointer, int)                   {}
func (p *fakeCapturePool) AllocManaged(int, int) (unsafe.Pointer, error)   { return nil, nil }
func (p *fakeCapturePool) FreeManaged(int, unsafe.Pointer, int)            {}
func (p *fakeCapturePool) Drain() error                                    { return nil }
func (p *fakeCapturePool) Stats() (int, int)                               { return 0, 0 }
func (p *fakeCapturePool) SetCaptureStream(_ unsafe.Pointer)               { p.capturing = true }
func (p *fakeCapturePool) ClearCaptureStream()                             { p.capturing = false }
func (p *fakeCapturePool) IsCapturing() bool                               { return p.capturing }

var (
	_ gpuapi.MemPool              = (*fakeCapturePool)(nil)
	_ gpuapi.CaptureAwareAllocator = (*fakeCapturePool)(nil)
)

// --- fake non-capture-aware pool (like CUDAArenaPool) ---

type fakeBasicPool struct{}

func (p *fakeBasicPool) Alloc(int, int) (unsafe.Pointer, error)          { return nil, nil }
func (p *fakeBasicPool) Free(int, unsafe.Pointer, int)                   {}
func (p *fakeBasicPool) AllocManaged(int, int) (unsafe.Pointer, error)   { return nil, nil }
func (p *fakeBasicPool) FreeManaged(int, unsafe.Pointer, int)            {}
func (p *fakeBasicPool) Drain() error                                    { return nil }
func (p *fakeBasicPool) Stats() (int, int)                               { return 0, 0 }

var _ gpuapi.MemPool = (*fakeBasicPool)(nil)

// --- test helpers ---

// swapMallocAsyncFn replaces the package-level mallocAsyncFn and returns
// a restore closure.
func swapMallocAsyncFn(fn func(int, *cuda.Stream) (unsafe.Pointer, error)) func() {
	prev := mallocAsyncFn
	mallocAsyncFn = fn
	return func() { mallocAsyncFn = prev }
}

// swapMallocManagedFn replaces the package-level mallocManagedFn and returns
// a restore closure.
func swapMallocManagedFn(fn func(int) (unsafe.Pointer, error)) func() {
	prev := mallocManagedFn
	mallocManagedFn = fn
	return func() { mallocManagedFn = prev }
}

// swapMemcpyAsyncFn replaces the package-level memcpyAsyncFn and returns
// a restore closure.
func swapMemcpyAsyncFn(fn func(unsafe.Pointer, unsafe.Pointer, int, cuda.MemcpyKind, *cuda.Stream) error) func() {
	prev := memcpyAsyncFn
	memcpyAsyncFn = fn
	return func() { memcpyAsyncFn = prev }
}

// --- allocWeight tests ---

// TestAllocWeight_UsesAsyncWhenCapturing verifies that allocWeight routes
// through cudaMallocAsync when CaptureAwareAllocator is active.
func TestAllocWeight_UsesAsyncWhenCapturing(t *testing.T) {
	var asyncCalled atomic.Bool
	var requestedSize int
	var sentinel byte

	restore := swapMallocAsyncFn(func(size int, _ *cuda.Stream) (unsafe.Pointer, error) {
		asyncCalled.Store(true)
		requestedSize = size
		return unsafe.Pointer(&sentinel), nil
	})
	defer restore()

	// Also stub captureStatusFn so ensureNotCapturing does not interfere.
	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restoreStatus()

	pool := &fakeCapturePool{capturing: true}
	e := &GPUEngine[float32]{
		stream: fakePtrStream{},
		pool:   pool,
	}

	ptr, err := e.allocWeight(4096)
	if err != nil {
		t.Fatalf("allocWeight during capture: unexpected error: %v", err)
	}
	if !asyncCalled.Load() {
		t.Fatal("allocWeight during capture: expected cudaMallocAsync to be called")
	}
	if requestedSize != 4096 {
		t.Fatalf("allocWeight during capture: async alloc size = %d, want 4096", requestedSize)
	}
	if ptr != unsafe.Pointer(&sentinel) {
		t.Fatal("allocWeight during capture: returned pointer does not match async allocation")
	}
}

// TestAllocWeight_UsesManagedWhenNotCapturing verifies that allocWeight
// still uses cudaMallocManaged when capture is NOT active and managedMem
// is true.
func TestAllocWeight_UsesManagedWhenNotCapturing(t *testing.T) {
	var managedCalled atomic.Bool
	var sentinel byte

	restoreManaged := swapMallocManagedFn(func(size int) (unsafe.Pointer, error) {
		managedCalled.Store(true)
		return unsafe.Pointer(&sentinel), nil
	})
	defer restoreManaged()

	var asyncCalled atomic.Bool
	restoreAsync := swapMallocAsyncFn(func(_ int, _ *cuda.Stream) (unsafe.Pointer, error) {
		asyncCalled.Store(true)
		return nil, nil
	})
	defer restoreAsync()

	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusNone, nil
	})
	defer restoreStatus()

	pool := &fakeCapturePool{capturing: false}
	e := &GPUEngine[float32]{
		stream:     fakePtrStream{},
		pool:       pool,
		managedMem: true,
	}

	ptr, err := e.allocWeight(4096)
	if err != nil {
		t.Fatalf("allocWeight (not capturing, managed): unexpected error: %v", err)
	}
	if !managedCalled.Load() {
		t.Fatal("allocWeight (not capturing, managed): expected cudaMallocManaged to be called")
	}
	if asyncCalled.Load() {
		t.Fatal("allocWeight (not capturing, managed): cudaMallocAsync should NOT be called")
	}
	if ptr != unsafe.Pointer(&sentinel) {
		t.Fatal("allocWeight (not capturing, managed): returned pointer does not match managed allocation")
	}
}

// TestAllocWeight_GuardFiresWithoutCaptureAwareAllocator verifies that
// ensureNotCapturing still blocks allocWeight when capture is active
// but the pool does NOT implement CaptureAwareAllocator (e.g.,
// CUDAArenaPool). This is the "raw capture without BeginCapture" path.
func TestAllocWeight_GuardFiresWithoutCaptureAwareAllocator(t *testing.T) {
	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restoreStatus()

	e := &GPUEngine[float32]{
		stream: fakePtrStream{},
		pool:   &fakeBasicPool{},
	}

	ptr, err := e.allocWeight(4096)
	if err == nil {
		t.Fatal("allocWeight with non-capture-aware pool during capture: expected error, got nil")
	}
	if !errors.Is(err, ErrCaptureIncompatibleAllocation) {
		t.Fatalf("allocWeight: expected ErrCaptureIncompatibleAllocation, got %v", err)
	}
	if ptr != nil {
		t.Fatalf("allocWeight: expected nil pointer on guard trip, got %p", ptr)
	}
}

// TestAllocWeight_GuardSkippedWhenCaptureAwareAllocatorActive verifies
// that ensureNotCapturing does NOT fire when CaptureAwareAllocator is
// properly engaged via BeginCapture/WithCapture.
func TestAllocWeight_GuardSkippedWhenCaptureAwareAllocatorActive(t *testing.T) {
	var ensureNotCapturingReached atomic.Bool
	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		ensureNotCapturingReached.Store(true)
		return cuda.CaptureStatusActive, nil
	})
	defer restoreStatus()

	restoreAsync := swapMallocAsyncFn(func(_ int, _ *cuda.Stream) (unsafe.Pointer, error) {
		var sentinel byte
		return unsafe.Pointer(&sentinel), nil
	})
	defer restoreAsync()

	pool := &fakeCapturePool{capturing: true}
	e := &GPUEngine[float32]{
		stream: fakePtrStream{},
		pool:   pool,
	}

	_, err := e.allocWeight(4096)
	if err != nil {
		t.Fatalf("allocWeight with capture-aware allocator active: unexpected error: %v", err)
	}
	if ensureNotCapturingReached.Load() {
		t.Fatal("ensureNotCapturing should NOT be called when CaptureAwareAllocator is active")
	}
}

// --- uploadBytes tests ---

// TestUploadBytes_UsesAsyncWhenCapturing verifies that uploadBytes routes
// through cudaMemcpyAsync when CaptureAwareAllocator is active.
func TestUploadBytes_UsesAsyncWhenCapturing(t *testing.T) {
	var asyncCalled atomic.Bool
	var copiedSize int
	var copiedKind cuda.MemcpyKind

	restoreMemcpy := swapMemcpyAsyncFn(func(_ unsafe.Pointer, _ unsafe.Pointer, count int, kind cuda.MemcpyKind, _ *cuda.Stream) error {
		asyncCalled.Store(true)
		copiedSize = count
		copiedKind = kind
		return nil
	})
	defer restoreMemcpy()

	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restoreStatus()

	pool := &fakeCapturePool{capturing: true}
	e := &GPUEngine[float32]{
		stream: fakePtrStream{},
		pool:   pool,
	}

	src := []byte{0x01, 0x02, 0x03, 0x04}
	var devMem byte
	err := e.uploadBytes(unsafe.Pointer(&devMem), src)
	if err != nil {
		t.Fatalf("uploadBytes during capture: unexpected error: %v", err)
	}
	if !asyncCalled.Load() {
		t.Fatal("uploadBytes during capture: expected cudaMemcpyAsync to be called")
	}
	if copiedSize != 4 {
		t.Fatalf("uploadBytes during capture: copied size = %d, want 4", copiedSize)
	}
	if copiedKind != cuda.MemcpyHostToDevice {
		t.Fatalf("uploadBytes during capture: copy kind = %v, want MemcpyHostToDevice", copiedKind)
	}
}

// TestUploadBytes_UsesSyncWhenNotCapturing verifies that uploadBytes
// falls through to the normal (non-async) path when capture is NOT active.
func TestUploadBytes_UsesSyncWhenNotCapturing(t *testing.T) {
	var asyncCalled atomic.Bool
	restoreMemcpy := swapMemcpyAsyncFn(func(_ unsafe.Pointer, _ unsafe.Pointer, _ int, _ cuda.MemcpyKind, _ *cuda.Stream) error {
		asyncCalled.Store(true)
		return nil
	})
	defer restoreMemcpy()

	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusNone, nil
	})
	defer restoreStatus()

	pool := &fakeCapturePool{capturing: false}
	e := &GPUEngine[float32]{
		stream:     fakePtrStream{},
		pool:       pool,
		managedMem: true,
	}

	// With managedMem=true and not capturing, uploadBytes does a direct CPU copy.
	// We can't test the actual copy without a real managed pointer, but we can
	// verify cudaMemcpyAsync was NOT called.
	src := []byte{0x01, 0x02}
	buf := make([]byte, 2)
	err := e.uploadBytes(unsafe.Pointer(&buf[0]), src)
	if err != nil {
		t.Fatalf("uploadBytes (not capturing, managed): unexpected error: %v", err)
	}
	if asyncCalled.Load() {
		t.Fatal("uploadBytes (not capturing, managed): cudaMemcpyAsync should NOT be called")
	}
	// Verify the sync copy worked.
	if buf[0] != 0x01 || buf[1] != 0x02 {
		t.Fatalf("uploadBytes (not capturing, managed): sync copy produced %v, want [1 2]", buf)
	}
}

// TestUploadBytes_GuardFiresWithoutCaptureAwareAllocator verifies that
// ensureNotCapturing still blocks uploadBytes when capture is active
// but the pool does NOT implement CaptureAwareAllocator.
func TestUploadBytes_GuardFiresWithoutCaptureAwareAllocator(t *testing.T) {
	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restoreStatus()

	e := &GPUEngine[float32]{
		stream: fakePtrStream{},
		pool:   &fakeBasicPool{},
	}

	src := []byte{0x01}
	err := e.uploadBytes(nil, src)
	if err == nil {
		t.Fatal("uploadBytes with non-capture-aware pool during capture: expected error, got nil")
	}
	if !errors.Is(err, ErrCaptureIncompatibleAllocation) {
		t.Fatalf("uploadBytes: expected ErrCaptureIncompatibleAllocation, got %v", err)
	}
}

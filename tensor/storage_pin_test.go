package tensor

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/gpuapi"
)

// fakePinnerPool implements gpuapi.MemPool + gpuapi.BackwardPinner and
// records pin/unpin calls so the GPUStorage delegation can be asserted
// without a GPU.
type fakePinnerPool struct {
	pins   []pinCall
	unpins []unsafe.Pointer
}

type pinCall struct {
	ptr  unsafe.Pointer
	size int
}

func (p *fakePinnerPool) Alloc(_, _ int) (unsafe.Pointer, error)        { return nil, nil }
func (p *fakePinnerPool) Free(_ int, _ unsafe.Pointer, _ int)           {}
func (p *fakePinnerPool) AllocManaged(_, _ int) (unsafe.Pointer, error) { return nil, nil }
func (p *fakePinnerPool) FreeManaged(_ int, _ unsafe.Pointer, _ int)    {}
func (p *fakePinnerPool) Drain() error                                  { return nil }
func (p *fakePinnerPool) Stats() (int, int)                             { return 0, 0 }

func (p *fakePinnerPool) PinBuffer(ptr unsafe.Pointer, byteSize int) bool {
	p.pins = append(p.pins, pinCall{ptr: ptr, size: byteSize})
	return true
}

func (p *fakePinnerPool) UnpinBuffer(ptr unsafe.Pointer) {
	p.unpins = append(p.unpins, ptr)
}

// plainPool implements gpuapi.MemPool WITHOUT BackwardPinner.
type plainPool struct{ fakePinnerPool }

var (
	_ gpuapi.MemPool        = (*fakePinnerPool)(nil)
	_ gpuapi.BackwardPinner = (*fakePinnerPool)(nil)
)

func newPinTestStorage(pool gpuapi.MemPool, buf []byte, allocSize int) *GPUStorage[float32] {
	return &GPUStorage[float32]{
		devicePtr: unsafe.Pointer(&buf[0]),
		length:    len(buf) / 4,
		byteSize:  len(buf),
		pool:      pool,
		allocSize: allocSize,
	}
}

// TestGPUStoragePin_DelegatesToPool: PinForBackward pins the allocation base
// pointer with the ORIGINAL allocation size (allocSize, not a view's
// byteSize), and UnpinForBackward unpins the same pointer.
func TestGPUStoragePin_DelegatesToPool(t *testing.T) {
	pool := &fakePinnerPool{}
	buf := make([]byte, 1024)
	s := newPinTestStorage(pool, buf, 2048) // allocSize > byteSize (e.g. shrunk view)

	if !s.PinForBackward() {
		t.Fatal("PinForBackward returned false for a pinner-backed pool")
	}
	if len(pool.pins) != 1 {
		t.Fatalf("PinBuffer called %d times, want 1", len(pool.pins))
	}
	if pool.pins[0].ptr != unsafe.Pointer(&buf[0]) {
		t.Fatalf("PinBuffer ptr = %p, want %p", pool.pins[0].ptr, unsafe.Pointer(&buf[0]))
	}
	if pool.pins[0].size != 2048 {
		t.Fatalf("PinBuffer size = %d, want 2048 (allocSize)", pool.pins[0].size)
	}

	s.UnpinForBackward()
	if len(pool.unpins) != 1 || pool.unpins[0] != unsafe.Pointer(&buf[0]) {
		t.Fatalf("UnpinBuffer calls = %v, want one call with %p", pool.unpins, unsafe.Pointer(&buf[0]))
	}
}

// TestGPUStoragePin_ZeroAllocSizeFallsBackToByteSize: legacy storages with
// allocSize 0 pin byteSize bytes.
func TestGPUStoragePin_ZeroAllocSizeFallsBackToByteSize(t *testing.T) {
	pool := &fakePinnerPool{}
	buf := make([]byte, 512)
	s := newPinTestStorage(pool, buf, 0)

	if !s.PinForBackward() {
		t.Fatal("PinForBackward returned false")
	}
	if pool.pins[0].size != 512 {
		t.Fatalf("PinBuffer size = %d, want 512 (byteSize fallback)", pool.pins[0].size)
	}
}

// TestGPUStoragePin_NonPinnerPoolNoop: pools without BackwardPinner (bucketed
// MemPool) make pinning a no-op returning false.
func TestGPUStoragePin_NonPinnerPoolNoop(t *testing.T) {
	pool := &plainPool{}
	buf := make([]byte, 256)
	// Hide the embedded pinner methods behind a MemPool-only wrapper.
	var mp gpuapi.MemPool = struct{ gpuapi.MemPool }{pool}
	s := newPinTestStorage(mp, buf, 256)

	if s.PinForBackward() {
		t.Fatal("PinForBackward returned true for a non-pinner pool")
	}
	s.UnpinForBackward() // must be a silent no-op
}

// TestGPUStoragePin_UnpinAfterFreeUsesCapturedPtr: Free zeroes devicePtr (the
// graph executor's refcount release path frees saved intermediates mid-step;
// the pool defers the actual free while pinned). UnpinForBackward must still
// release the pin taken at pin time.
func TestGPUStoragePin_UnpinAfterFreeUsesCapturedPtr(t *testing.T) {
	pool := &fakePinnerPool{}
	buf := make([]byte, 256)
	s := newPinTestStorage(pool, buf, 256)
	base := unsafe.Pointer(&buf[0])

	if !s.PinForBackward() {
		t.Fatal("PinForBackward returned false")
	}
	if err := s.Free(); err != nil {
		t.Fatalf("Free: %v", err)
	}
	if s.devicePtr != nil {
		t.Fatal("Free did not zero devicePtr (test premise broken)")
	}

	s.UnpinForBackward()
	if len(pool.unpins) != 1 || pool.unpins[0] != base {
		t.Fatalf("UnpinBuffer after Free = %v, want one call with captured ptr %p", pool.unpins, base)
	}
}

// TestGPUStoragePin_UnpinWithoutPinNoop: UnpinForBackward without a prior
// successful pin must not call the pool.
func TestGPUStoragePin_UnpinWithoutPinNoop(t *testing.T) {
	pool := &fakePinnerPool{}
	buf := make([]byte, 256)
	s := newPinTestStorage(pool, buf, 256)

	s.UnpinForBackward()
	if len(pool.unpins) != 0 {
		t.Fatalf("UnpinBuffer called %d times without a pin, want 0", len(pool.unpins))
	}
}

// TestGPUStoragePin_NilOrEmptyStorage: nil device pointers and zero-byte
// storages never pin.
func TestGPUStoragePin_NilOrEmptyStorage(t *testing.T) {
	pool := &fakePinnerPool{}
	s := &GPUStorage[float32]{pool: pool}
	if s.PinForBackward() {
		t.Fatal("PinForBackward returned true for nil devicePtr")
	}
	s.UnpinForBackward()
	if len(pool.pins) != 0 || len(pool.unpins) != 0 {
		t.Fatalf("pool touched for nil storage: pins=%d unpins=%d", len(pool.pins), len(pool.unpins))
	}
}

// TestCPUStorage_NotPinnable: CPU storage must NOT implement PinnableStorage
// (GC-owned memory cannot be reclaimed behind a live reference; the graph
// treats it as a no-op).
func TestCPUStorage_NotPinnable(t *testing.T) {
	var s Storage[float32] = NewCPUStorage([]float32{1, 2, 3})
	if _, ok := s.(PinnableStorage); ok {
		t.Fatal("CPU storage unexpectedly implements PinnableStorage")
	}
}

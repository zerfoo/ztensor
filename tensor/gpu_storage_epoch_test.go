package tensor

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/gpuapi"
)

// fakeEpochPool implements gpuapi.MemPool + gpuapi.EpochMemPool and records
// how frees arrive, so the test can assert that GPUStorage releases
// pool-backed memory through the epoch-guarded path with the epoch captured
// at ALLOCATION time (Bug 11 residual: GC-finalizer frees firing after the
// Reset that reclaimed the allocation must be droppable by the pool).
type fakeEpochPool struct {
	epoch        uint64
	plainFrees   int
	epochFrees   int
	gotFreeEpoch uint64
}

func (p *fakeEpochPool) Alloc(_, _ int) (unsafe.Pointer, error)        { return nil, nil }
func (p *fakeEpochPool) Free(_ int, _ unsafe.Pointer, _ int)           { p.plainFrees++ }
func (p *fakeEpochPool) AllocManaged(_, _ int) (unsafe.Pointer, error) { return nil, nil }
func (p *fakeEpochPool) FreeManaged(_ int, _ unsafe.Pointer, _ int)    {}
func (p *fakeEpochPool) Drain() error                                  { return nil }
func (p *fakeEpochPool) Stats() (int, int)                             { return 0, 0 }

func (p *fakeEpochPool) Epoch() uint64 { return p.epoch }
func (p *fakeEpochPool) FreeAtEpoch(_ int, _ unsafe.Pointer, _ int, allocEpoch uint64) {
	p.epochFrees++
	p.gotFreeEpoch = allocEpoch
}

var (
	_ gpuapi.MemPool      = (*fakeEpochPool)(nil)
	_ gpuapi.EpochMemPool = (*fakeEpochPool)(nil)
)

// withFakeDefaultRuntime installs fakeAsyncRuntime (host_access_sync_test.go)
// as the package default runtime so pool-backed storage can be constructed
// without a CUDA device, restoring the previous runtime on cleanup.
func withFakeDefaultRuntime(t *testing.T) {
	t.Helper()
	defaultRuntimeOnce.Do(func() {}) // ensure lazy init cannot overwrite the fake
	prev := defaultRuntime
	defaultRuntime = fakeAsyncRuntime{}
	t.Cleanup(func() { defaultRuntime = prev })
}

func TestGPUStorageFromPool_FreesThroughAllocationEpoch(t *testing.T) {
	withFakeDefaultRuntime(t)

	pool := &fakeEpochPool{epoch: 7}
	buf := make([]float32, 4)

	s, err := NewGPUStorageFromPool[float32](unsafe.Pointer(&buf[0]), len(buf), pool, 0)
	if err != nil {
		t.Fatalf("NewGPUStorageFromPool: %v", err)
	}

	// The pool resets between allocation and the (finalizer-driven) Free.
	pool.epoch = 9

	if err := s.Free(); err != nil {
		t.Fatalf("Free: %v", err)
	}

	if pool.plainFrees != 0 {
		t.Fatalf("storage bypassed the epoch guard: %d plain Free calls", pool.plainFrees)
	}
	if pool.epochFrees != 1 {
		t.Fatalf("epoch-guarded frees = %d, want 1", pool.epochFrees)
	}
	if pool.gotFreeEpoch != 7 {
		t.Fatalf("free carried epoch %d, want the allocation-time epoch 7", pool.gotFreeEpoch)
	}
}

func TestGPUStorageView_PropagatesAllocationEpoch(t *testing.T) {
	withFakeDefaultRuntime(t)

	pool := &fakeEpochPool{epoch: 3}
	buf := make([]float32, 4)

	s, err := NewGPUStorageFromPool[float32](unsafe.Pointer(&buf[0]), len(buf), pool, 0)
	if err != nil {
		t.Fatalf("NewGPUStorageFromPool: %v", err)
	}
	v := s.View(2)

	// Drop the parent first (refcount 2 -> 1: no pool release), then the
	// view (1 -> 0: the actual release, which must carry epoch 3).
	pool.epoch = 5
	if err := s.Free(); err != nil {
		t.Fatalf("parent Free: %v", err)
	}
	if pool.epochFrees != 0 {
		t.Fatalf("parent Free released memory while the view is live")
	}
	if err := v.Free(); err != nil {
		t.Fatalf("view Free: %v", err)
	}
	if pool.epochFrees != 1 || pool.gotFreeEpoch != 3 {
		t.Fatalf("view free: epochFrees=%d gotEpoch=%d, want 1 and 3",
			pool.epochFrees, pool.gotFreeEpoch)
	}
}

package gpuapi

import (
	"fmt"
	"sync"
	"unsafe"
)

// SYCLMemPool implements the MemPool interface for SYCL devices.
// It uses a simple size-bucketed cache of SYCL device buffers.
type SYCLMemPool struct {
	mu      sync.Mutex
	rt      *SYCLRuntime
	cache   map[int][]unsafe.Pointer // byteSize -> []pi_mem
	allocs  int
	totalSz int
}

// NewSYCLMemPool creates a new SYCL memory pool.
func NewSYCLMemPool(rt *SYCLRuntime) *SYCLMemPool {
	return &SYCLMemPool{
		rt:    rt,
		cache: make(map[int][]unsafe.Pointer),
	}
}

func (p *SYCLMemPool) Alloc(_ int, byteSize int) (unsafe.Pointer, error) {
	p.mu.Lock()
	if ptrs := p.cache[byteSize]; len(ptrs) > 0 {
		ptr := ptrs[len(ptrs)-1]
		p.cache[byteSize] = ptrs[:len(ptrs)-1]
		p.allocs--
		p.totalSz -= byteSize
		p.mu.Unlock()
		return ptr, nil
	}
	p.mu.Unlock()

	return p.rt.Malloc(byteSize)
}

func (p *SYCLMemPool) Free(_ int, ptr unsafe.Pointer, byteSize int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.cache[byteSize] = append(p.cache[byteSize], ptr)
	p.allocs++
	p.totalSz += byteSize
}

func (p *SYCLMemPool) AllocManaged(_, _ int) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("AllocManaged: not supported on SYCL backend")
}

func (p *SYCLMemPool) FreeManaged(_ int, _ unsafe.Pointer, _ int) {}

func (p *SYCLMemPool) Drain() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	for sz, ptrs := range p.cache {
		for _, ptr := range ptrs {
			_ = p.rt.Free(ptr)
		}
		delete(p.cache, sz)
	}
	p.allocs = 0
	p.totalSz = 0
	return nil
}

func (p *SYCLMemPool) Stats() (int, int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.allocs, p.totalSz
}

// Compile-time interface assertion.
var _ MemPool = (*SYCLMemPool)(nil)

package gpuapi

import (
	"fmt"
	"sync"
	"unsafe"
)

// FPGAMemPool implements the MemPool interface for FPGA accelerators.
// It uses a simple size-bucketed cache of FPGA device buffers.
type FPGAMemPool struct {
	mu      sync.Mutex
	rt      *FPGARuntime
	cache   map[int][]unsafe.Pointer // byteSize -> []cl_mem
	allocs  int
	totalSz int
}

// NewFPGAMemPool creates a new FPGA memory pool.
func NewFPGAMemPool(rt *FPGARuntime) *FPGAMemPool {
	return &FPGAMemPool{
		rt:    rt,
		cache: make(map[int][]unsafe.Pointer),
	}
}

func (p *FPGAMemPool) Alloc(_ int, byteSize int) (unsafe.Pointer, error) {
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

func (p *FPGAMemPool) Free(_ int, ptr unsafe.Pointer, byteSize int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.cache[byteSize] = append(p.cache[byteSize], ptr)
	p.allocs++
	p.totalSz += byteSize
}

func (p *FPGAMemPool) AllocManaged(_, _ int) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("AllocManaged: not supported on FPGA backend")
}

func (p *FPGAMemPool) FreeManaged(_ int, _ unsafe.Pointer, _ int) {}

func (p *FPGAMemPool) Drain() error {
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

func (p *FPGAMemPool) Stats() (int, int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.allocs, p.totalSz
}

// Compile-time interface assertion.
var _ MemPool = (*FPGAMemPool)(nil)

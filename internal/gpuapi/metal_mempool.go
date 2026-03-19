package gpuapi

import (
	"fmt"
	"sync"
	"unsafe"
)

// MetalMemPool implements the MemPool interface for Metal.
// It uses a simple size-bucketed cache of Metal buffers.
type MetalMemPool struct {
	mu      sync.Mutex
	rt      *MetalRuntime
	cache   map[int][]unsafe.Pointer // byteSize -> []id<MTLBuffer>
	allocs  int
	totalSz int
}

// NewMetalMemPool creates a new Metal memory pool.
func NewMetalMemPool(rt *MetalRuntime) *MetalMemPool {
	return &MetalMemPool{
		rt:    rt,
		cache: make(map[int][]unsafe.Pointer),
	}
}

func (p *MetalMemPool) Alloc(_ int, byteSize int) (unsafe.Pointer, error) {
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

func (p *MetalMemPool) Free(_ int, ptr unsafe.Pointer, byteSize int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.cache[byteSize] = append(p.cache[byteSize], ptr)
	p.allocs++
	p.totalSz += byteSize
}

func (p *MetalMemPool) AllocManaged(_, _ int) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("AllocManaged: not supported on Metal backend")
}

func (p *MetalMemPool) FreeManaged(_ int, _ unsafe.Pointer, _ int) {}

func (p *MetalMemPool) Drain() error {
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

func (p *MetalMemPool) Stats() (int, int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.allocs, p.totalSz
}

// Compile-time interface assertion.
var _ MemPool = (*MetalMemPool)(nil)

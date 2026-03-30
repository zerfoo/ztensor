package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// CUDAMemPool implements the MemPool interface by wrapping cuda.MemPool.
type CUDAMemPool struct {
	inner *cuda.MemPool
}

// NewCUDAMemPool creates a new CUDA memory pool adapter.
func NewCUDAMemPool() *CUDAMemPool {
	return &CUDAMemPool{inner: cuda.NewMemPool()}
}

// NewCUDAMemPoolFrom wraps an existing cuda.MemPool.
func NewCUDAMemPoolFrom(pool *cuda.MemPool) *CUDAMemPool {
	return &CUDAMemPool{inner: pool}
}

func (p *CUDAMemPool) Alloc(deviceID, byteSize int) (unsafe.Pointer, error) {
	return p.inner.Alloc(deviceID, byteSize)
}

func (p *CUDAMemPool) Free(deviceID int, ptr unsafe.Pointer, byteSize int) {
	p.inner.Free(deviceID, ptr, byteSize)
}

func (p *CUDAMemPool) AllocManaged(deviceID, byteSize int) (unsafe.Pointer, error) {
	return p.inner.AllocManaged(deviceID, byteSize)
}

func (p *CUDAMemPool) FreeManaged(deviceID int, ptr unsafe.Pointer, byteSize int) {
	p.inner.FreeManaged(deviceID, ptr, byteSize)
}

func (p *CUDAMemPool) Drain() error {
	return p.inner.Drain()
}

func (p *CUDAMemPool) Stats() (int, int) {
	return p.inner.Stats()
}

// SetCaptureStream enables capture-aware allocation mode on the
// underlying cuda.MemPool. During capture, fresh allocations use
// cudaMallocAsync on the given stream.
func (p *CUDAMemPool) SetCaptureStream(stream unsafe.Pointer) {
	p.inner.SetCaptureStream(cuda.StreamFromPtr(stream))
}

// ClearCaptureStream disables capture-aware allocation mode.
func (p *CUDAMemPool) ClearCaptureStream() {
	p.inner.ClearCaptureStream()
}

// Inner returns the underlying cuda.MemPool for backward compatibility.
func (p *CUDAMemPool) Inner() *cuda.MemPool {
	return p.inner
}

// Compile-time interface assertions.
var _ MemPool = (*CUDAMemPool)(nil)
var _ CaptureAwareAllocator = (*CUDAMemPool)(nil)

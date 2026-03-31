package gpuapi

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/internal/sycl"
)

// SYCLRuntime implements the Runtime interface using the SYCL runtime
// (Intel oneAPI DPC++).
type SYCLRuntime struct {
	ctx *sycl.Context
}

// NewSYCLRuntime returns a new SYCL runtime adapter.
// Returns nil if SYCL is not available on this system.
func NewSYCLRuntime() *SYCLRuntime {
	if !sycl.Available() {
		return nil
	}
	return &SYCLRuntime{}
}

func (r *SYCLRuntime) DeviceType() device.Type { return device.SYCL }

func (r *SYCLRuntime) SetDevice(deviceID int) error {
	if r.ctx != nil {
		_ = r.ctx.Destroy()
	}
	ctx, err := sycl.NewContext(deviceID)
	if err != nil {
		return err
	}
	r.ctx = ctx
	return nil
}

func (r *SYCLRuntime) GetDeviceCount() (int, error) {
	if !sycl.Available() {
		return 0, fmt.Errorf("sycl: not available")
	}
	return sycl.GetDeviceCount()
}

func (r *SYCLRuntime) Malloc(byteSize int) (unsafe.Pointer, error) {
	return r.ctx.Malloc(byteSize)
}

func (r *SYCLRuntime) Free(ptr unsafe.Pointer) error {
	return r.ctx.Free(ptr)
}

func (r *SYCLRuntime) Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	return r.ctx.Memcpy(dst, src, count, syclMemcpyKind(kind))
}

func (r *SYCLRuntime) MemcpyAsync(dst, src unsafe.Pointer, count int, kind MemcpyKind, _ Stream) error {
	// SYCL queues are in-order by default; fall back to synchronous copy.
	return r.Memcpy(dst, src, count, kind)
}

func (r *SYCLRuntime) MemsetAsync(_ unsafe.Pointer, _ int, _ int, _ Stream) error {
	return fmt.Errorf("SYCLRuntime.MemsetAsync: not yet implemented")
}

func (r *SYCLRuntime) MemcpyPeer(dst unsafe.Pointer, _ int, src unsafe.Pointer, _ int, count int) error {
	// SYCL does not support direct peer-to-peer transfer between devices.
	// Fall back to D2H + H2D via a host buffer.
	buf := make([]byte, count)
	hostPtr := unsafe.Pointer(unsafe.SliceData(buf))
	if err := r.ctx.Memcpy(hostPtr, src, count, sycl.MemcpyDeviceToHost); err != nil {
		return err
	}
	return r.ctx.Memcpy(dst, hostPtr, count, sycl.MemcpyHostToDevice)
}

func (r *SYCLRuntime) CreateStream() (Stream, error) {
	s, err := r.ctx.CreateStream()
	if err != nil {
		return nil, err
	}
	return &syclStreamWrapper{stream: s}, nil
}

// syclStreamWrapper adapts sycl.Stream to the GRAL Stream interface.
type syclStreamWrapper struct {
	stream *sycl.Stream
}

func (w *syclStreamWrapper) Synchronize() error { return w.stream.Synchronize() }
func (w *syclStreamWrapper) Destroy() error      { return w.stream.Destroy() }
func (w *syclStreamWrapper) Ptr() unsafe.Pointer  { return w.stream.Ptr() }

// syclMemcpyKind converts a GRAL MemcpyKind to a SYCL MemcpyKind.
func syclMemcpyKind(kind MemcpyKind) sycl.MemcpyKind {
	switch kind {
	case MemcpyHostToDevice:
		return sycl.MemcpyHostToDevice
	case MemcpyDeviceToHost:
		return sycl.MemcpyDeviceToHost
	case MemcpyDeviceToDevice:
		return sycl.MemcpyDeviceToDevice
	default:
		return sycl.MemcpyHostToDevice
	}
}

// Compile-time interface assertion.
var _ Runtime = (*SYCLRuntime)(nil)

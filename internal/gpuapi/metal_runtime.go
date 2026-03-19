package gpuapi

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/internal/metal"
)

// MetalRuntime implements the Runtime interface using Apple Metal.
type MetalRuntime struct {
	ctx *metal.Context
}

// NewMetalRuntime returns a new Metal runtime adapter.
// Returns nil if Metal is not available on this system.
func NewMetalRuntime() *MetalRuntime {
	if !metal.Available() {
		return nil
	}
	return &MetalRuntime{}
}

func (r *MetalRuntime) DeviceType() device.Type { return device.Metal }

func (r *MetalRuntime) SetDevice(deviceID int) error {
	if r.ctx != nil {
		_ = r.ctx.Destroy()
	}
	ctx, err := metal.NewContext(deviceID)
	if err != nil {
		return err
	}
	r.ctx = ctx
	return nil
}

func (r *MetalRuntime) GetDeviceCount() (int, error) {
	if !metal.Available() {
		return 0, fmt.Errorf("metal: not available")
	}
	return metal.GetDeviceCount()
}

func (r *MetalRuntime) Malloc(byteSize int) (unsafe.Pointer, error) {
	return r.ctx.Malloc(byteSize)
}

func (r *MetalRuntime) Free(ptr unsafe.Pointer) error {
	return r.ctx.Free(ptr)
}

func (r *MetalRuntime) Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	return r.ctx.Memcpy(dst, src, count, metalMemcpyKind(kind))
}

func (r *MetalRuntime) MemcpyAsync(dst, src unsafe.Pointer, count int, kind MemcpyKind, _ Stream) error {
	// Metal shared storage mode makes buffers CPU-accessible.
	// For simplicity, fall back to synchronous copy.
	return r.Memcpy(dst, src, count, kind)
}

func (r *MetalRuntime) MemcpyPeer(dst unsafe.Pointer, _ int, src unsafe.Pointer, _ int, count int) error {
	// Metal does not support direct peer-to-peer between discrete GPUs.
	// Fall back to D2H + H2D via a host buffer.
	buf := make([]byte, count)
	hostPtr := unsafe.Pointer(unsafe.SliceData(buf))
	if err := r.ctx.Memcpy(hostPtr, src, count, metal.MemcpyDeviceToHost); err != nil {
		return err
	}
	return r.ctx.Memcpy(dst, hostPtr, count, metal.MemcpyHostToDevice)
}

func (r *MetalRuntime) CreateStream() (Stream, error) {
	s, err := r.ctx.CreateStream()
	if err != nil {
		return nil, err
	}
	return &metalStreamWrapper{stream: s}, nil
}

// metalStreamWrapper adapts metal.Stream to the GRAL Stream interface.
type metalStreamWrapper struct {
	stream *metal.Stream
}

func (w *metalStreamWrapper) Synchronize() error { return w.stream.Synchronize() }
func (w *metalStreamWrapper) Destroy() error      { return w.stream.Destroy() }
func (w *metalStreamWrapper) Ptr() unsafe.Pointer  { return w.stream.Ptr() }

// metalMemcpyKind converts a GRAL MemcpyKind to a Metal MemcpyKind.
func metalMemcpyKind(kind MemcpyKind) metal.MemcpyKind {
	switch kind {
	case MemcpyHostToDevice:
		return metal.MemcpyHostToDevice
	case MemcpyDeviceToHost:
		return metal.MemcpyDeviceToHost
	case MemcpyDeviceToDevice:
		return metal.MemcpyDeviceToDevice
	default:
		return metal.MemcpyHostToDevice
	}
}

// Compile-time interface assertion.
var _ Runtime = (*MetalRuntime)(nil)

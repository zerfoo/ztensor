//go:build darwin

package metal

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// Metal buffer storage modes.
const (
	mtlStorageModeShared  = 0 // MTLStorageModeShared
	mtlResourceStorageShift = 4
)

// Context holds a Metal device, command queue, and related state.
type Context struct {
	device   uintptr // id<MTLDevice>
	queue    uintptr // id<MTLCommandQueue>
	deviceID int
}

// Stream wraps a Metal command queue for stream-like semantics.
type Stream struct {
	queue uintptr // id<MTLCommandQueue>
}

// MemcpyKind specifies the direction of a memory copy.
type MemcpyKind int

const (
	MemcpyHostToDevice MemcpyKind = iota
	MemcpyDeviceToHost
	MemcpyDeviceToDevice
)

func lib() *MetalLib {
	return Lib()
}

// GetDeviceCount returns the number of Metal GPU devices.
func GetDeviceCount() (int, error) {
	l := lib()
	if l == nil {
		return 0, fmt.Errorf("metal not available")
	}

	// MTLCopyAllDevices() returns NSArray<id<MTLDevice>>*
	arr := cuda.Ccall(l.selCopyAllDevices)
	if arr == 0 {
		return 0, nil
	}
	count := l.MsgSend(arr, l.selCount)
	// Release the array.
	l.MsgSend(arr, l.selRelease)
	return int(count), nil
}

// NewContext creates a Metal context on the specified device.
func NewContext(deviceID int) (*Context, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("metal not available")
	}

	// Get all Metal devices.
	arr := cuda.Ccall(l.selCopyAllDevices)
	if arr == 0 {
		return nil, fmt.Errorf("metal: MTLCopyAllDevices returned nil")
	}
	count := int(l.MsgSend(arr, l.selCount))
	if deviceID < 0 {
		deviceID = 0
	}
	if deviceID >= count {
		l.MsgSend(arr, l.selRelease)
		return nil, fmt.Errorf("metal device %d not found (have %d)", deviceID, count)
	}

	dev := l.MsgSend(arr, l.selObjectAtIndex, uintptr(deviceID))
	l.MsgSend(dev, l.selRetain)
	l.MsgSend(arr, l.selRelease)

	// Create command queue.
	queue := l.MsgSend(dev, l.selNewCommandQueue)
	if queue == 0 {
		l.MsgSend(dev, l.selRelease)
		return nil, fmt.Errorf("metal: newCommandQueue failed")
	}

	return &Context{
		device:   dev,
		queue:    queue,
		deviceID: deviceID,
	}, nil
}

// Destroy releases the Metal context resources.
func (c *Context) Destroy() error {
	l := lib()
	if l == nil {
		return nil
	}
	if c.queue != 0 {
		l.MsgSend(c.queue, l.selRelease)
		c.queue = 0
	}
	if c.device != 0 {
		l.MsgSend(c.device, l.selRelease)
		c.device = 0
	}
	return nil
}

// Malloc allocates a Metal buffer of the given size with shared storage mode.
// Returns the id<MTLBuffer> handle cast to unsafe.Pointer.
func (c *Context) Malloc(size int) (unsafe.Pointer, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("metal not available")
	}

	// [device newBufferWithLength:size options:MTLResourceStorageModeShared]
	options := uintptr(mtlStorageModeShared << mtlResourceStorageShift)
	buf := l.MsgSend(c.device, l.selNewBufferWithLen, uintptr(size), options)
	if buf == 0 {
		return nil, fmt.Errorf("metal: newBufferWithLength(%d) failed", size)
	}
	return unsafe.Pointer(buf), nil //nolint:govet
}

// Free releases a Metal buffer.
func (c *Context) Free(ptr unsafe.Pointer) error {
	if ptr == nil {
		return nil
	}
	l := lib()
	if l == nil {
		return fmt.Errorf("metal not available")
	}
	l.MsgSend(uintptr(ptr), l.selRelease)
	return nil
}

// bufferContents returns the CPU-accessible pointer for a shared Metal buffer.
func (c *Context) bufferContents(buf uintptr) unsafe.Pointer {
	l := lib()
	return unsafe.Pointer(l.MsgSend(buf, l.selContents)) //nolint:govet
}

// bufferLength returns the size of a Metal buffer.
func (c *Context) bufferLength(buf uintptr) int {
	l := lib()
	return int(l.MsgSend(buf, l.selLength))
}

// Memcpy copies data between host and Metal device memory.
// Metal uses shared storage mode, so buffers are CPU-accessible.
// For H2D and D2H, we memcpy through the buffer's contents pointer.
func (c *Context) Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	switch kind {
	case MemcpyHostToDevice:
		// dst is a Metal buffer, src is host memory.
		dstPtr := c.bufferContents(uintptr(dst))
		copy(unsafe.Slice((*byte)(dstPtr), count), unsafe.Slice((*byte)(src), count))
	case MemcpyDeviceToHost:
		// src is a Metal buffer, dst is host memory.
		srcPtr := c.bufferContents(uintptr(src))
		copy(unsafe.Slice((*byte)(dst), count), unsafe.Slice((*byte)(srcPtr), count))
	case MemcpyDeviceToDevice:
		// Both are Metal buffers.
		srcPtr := c.bufferContents(uintptr(src))
		dstPtr := c.bufferContents(uintptr(dst))
		copy(unsafe.Slice((*byte)(dstPtr), count), unsafe.Slice((*byte)(srcPtr), count))
	default:
		return fmt.Errorf("metal: unsupported MemcpyKind: %d", kind)
	}
	return nil
}

// CreateStream creates a new command queue.
func (c *Context) CreateStream() (*Stream, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("metal not available")
	}
	queue := l.MsgSend(c.device, l.selNewCommandQueue)
	if queue == 0 {
		return nil, fmt.Errorf("metal: newCommandQueue failed")
	}
	return &Stream{queue: queue}, nil
}

// Synchronize is a no-op for the command queue itself.
// Metal synchronization happens at the command buffer level.
func (s *Stream) Synchronize() error {
	return nil
}

// Destroy releases the command queue.
func (s *Stream) Destroy() error {
	if s.queue == 0 {
		return nil
	}
	l := lib()
	if l == nil {
		return nil
	}
	l.MsgSend(s.queue, l.selRelease)
	s.queue = 0
	return nil
}

// Ptr returns the underlying command queue handle as unsafe.Pointer.
func (s *Stream) Ptr() unsafe.Pointer {
	return unsafe.Pointer(s.queue) //nolint:govet
}

// Device returns the underlying MTLDevice handle.
func (c *Context) Device() uintptr {
	return c.device
}

// Queue returns the default command queue handle.
func (c *Context) Queue() uintptr {
	return c.queue
}

// DeviceID returns the device ordinal.
func (c *Context) DeviceID() int {
	return c.deviceID
}

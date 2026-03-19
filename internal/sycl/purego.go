//go:build linux

package sycl

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// SYCLLib holds dlopen handles and resolved function pointers for the SYCL
// runtime. Currently supports Intel oneAPI Level Zero runtime via libsycl.so.
// Function pointers are resolved at Open() time via dlsym.
type SYCLLib struct {
	handle uintptr

	// Core runtime functions (Level Zero / SYCL PI).
	fnPlatformGet      uintptr // piPlatformsGet
	fnDeviceGet        uintptr // piDevicesGet
	fnContextCreate    uintptr // piContextCreate
	fnQueueCreate      uintptr // piQueueCreate
	fnMemBufferCreate  uintptr // piMemBufferCreate
	fnMemRelease       uintptr // piMemRelease
	fnEnqueueMemRead   uintptr // piEnqueueMemBufferRead
	fnEnqueueMemWrite  uintptr // piEnqueueMemBufferWrite
	fnEnqueueMemCopy   uintptr // piEnqueueMemBufferCopy
	fnQueueFinish      uintptr // piQueueFinish
	fnQueueRelease     uintptr // piQueueRelease
	fnContextRelease   uintptr // piContextRelease
	fnDeviceGetInfo    uintptr // piDeviceGetInfo
}

var (
	globalLib  *SYCLLib
	globalOnce sync.Once
	errGlobal  error
)

// SYCL runtime library paths.
const (
	syclLibPath    = "libsycl.so"
	syclLibPath7   = "libsycl.so.7"
	syclLibPath6   = "libsycl.so.6"
	levelZeroPath  = "libze_loader.so.1"
)

// Open loads the SYCL runtime library via dlopen and resolves required
// function pointers. It tries libsycl.so (Intel oneAPI DPC++ runtime) first.
func Open() (*SYCLLib, error) {
	lib := &SYCLLib{}

	var err error

	// Try loading the SYCL library.
	lib.handle, err = cuda.DlopenPath(syclLibPath)
	if err != nil {
		lib.handle, err = cuda.DlopenPath(syclLibPath7)
		if err != nil {
			lib.handle, err = cuda.DlopenPath(syclLibPath6)
			if err != nil {
				return nil, fmt.Errorf("sycl: no SYCL runtime found: %w", err)
			}
		}
	}

	// Resolve SYCL PI (Plugin Interface) function pointers.
	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"piPlatformsGet", &lib.fnPlatformGet},
		{"piDevicesGet", &lib.fnDeviceGet},
		{"piContextCreate", &lib.fnContextCreate},
		{"piQueueCreate", &lib.fnQueueCreate},
		{"piMemBufferCreate", &lib.fnMemBufferCreate},
		{"piMemRelease", &lib.fnMemRelease},
		{"piEnqueueMemBufferRead", &lib.fnEnqueueMemRead},
		{"piEnqueueMemBufferWrite", &lib.fnEnqueueMemWrite},
		{"piEnqueueMemBufferCopy", &lib.fnEnqueueMemCopy},
		{"piQueueFinish", &lib.fnQueueFinish},
		{"piQueueRelease", &lib.fnQueueRelease},
		{"piContextRelease", &lib.fnContextRelease},
		{"piDeviceGetInfo", &lib.fnDeviceGetInfo},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(lib.handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("sycl: %w", err)
		}
		*s.ptr = addr
	}

	return lib, nil
}

// Available returns true if a SYCL runtime is available on this machine.
func Available() bool {
	globalOnce.Do(func() {
		globalLib, errGlobal = Open()
	})
	return errGlobal == nil
}

// Lib returns the global SYCLLib instance, or nil if not available.
func Lib() *SYCLLib {
	if !Available() {
		return nil
	}
	return globalLib
}

// MemcpyKind specifies the direction of a memory copy.
type MemcpyKind int

const (
	MemcpyHostToDevice MemcpyKind = iota
	MemcpyDeviceToHost
	MemcpyDeviceToDevice
)

// PI memory flags.
const (
	piMemFlagsReadWrite = 1 << 0 // PI_MEM_FLAGS_ACCESS_RW
)

// Context holds a SYCL platform, device, context, and queue.
type Context struct {
	platform uintptr // pi_platform
	device   uintptr // pi_device
	context  uintptr // pi_context
	queue    uintptr // pi_queue
	deviceID int
}

// Stream wraps a SYCL queue for stream-like semantics.
type Stream struct {
	queue uintptr // pi_queue
}

// GetDeviceCount returns the number of SYCL-capable devices.
func GetDeviceCount() (int, error) {
	l := Lib()
	if l == nil {
		return 0, fmt.Errorf("sycl not available")
	}

	// Get platforms.
	var numPlatforms uint32
	cuda.Ccall(l.fnPlatformGet, 0, 0, uintptr(unsafe.Pointer(&numPlatforms)))
	if numPlatforms == 0 {
		return 0, nil
	}

	platforms := make([]uintptr, numPlatforms)
	cuda.Ccall(l.fnPlatformGet, uintptr(numPlatforms), uintptr(unsafe.Pointer(&platforms[0])), uintptr(unsafe.Pointer(&numPlatforms)))

	// Count all devices across all platforms.
	var total uint32
	for _, p := range platforms {
		var count uint32
		// PI_DEVICE_TYPE_ALL = 0xFFFFFFFF
		cuda.Ccall(l.fnDeviceGet, p, 0xFFFFFFFF, 0, 0, uintptr(unsafe.Pointer(&count)))
		total += count
	}
	return int(total), nil
}

// NewContext creates a SYCL context on the specified device.
func NewContext(deviceID int) (*Context, error) {
	l := Lib()
	if l == nil {
		return nil, fmt.Errorf("sycl not available")
	}

	// Get platforms.
	var numPlatforms uint32
	cuda.Ccall(l.fnPlatformGet, 0, 0, uintptr(unsafe.Pointer(&numPlatforms)))
	if numPlatforms == 0 {
		return nil, fmt.Errorf("sycl: no platforms found")
	}

	platforms := make([]uintptr, numPlatforms)
	cuda.Ccall(l.fnPlatformGet, uintptr(numPlatforms), uintptr(unsafe.Pointer(&platforms[0])), uintptr(unsafe.Pointer(&numPlatforms)))

	// Find the device across all platforms.
	idx := 0
	for _, p := range platforms {
		var count uint32
		cuda.Ccall(l.fnDeviceGet, p, 0xFFFFFFFF, 0, 0, uintptr(unsafe.Pointer(&count)))
		if count == 0 {
			continue
		}
		if idx+int(count) > deviceID {
			devices := make([]uintptr, count)
			cuda.Ccall(l.fnDeviceGet, p, 0xFFFFFFFF, uintptr(count), uintptr(unsafe.Pointer(&devices[0])), uintptr(unsafe.Pointer(&count)))

			dev := devices[deviceID-idx]

			// Create context.
			var ctx uintptr
			ret := cuda.Ccall(l.fnContextCreate, 0, 1, uintptr(unsafe.Pointer(&dev)), 0, uintptr(unsafe.Pointer(&ctx)))
			if ret != 0 {
				return nil, fmt.Errorf("sycl: piContextCreate failed: %d", ret)
			}

			// Create queue.
			var queue uintptr
			ret = cuda.Ccall(l.fnQueueCreate, ctx, dev, 0, uintptr(unsafe.Pointer(&queue)))
			if ret != 0 {
				return nil, fmt.Errorf("sycl: piQueueCreate failed: %d", ret)
			}

			return &Context{
				platform: p,
				device:   dev,
				context:  ctx,
				queue:    queue,
				deviceID: deviceID,
			}, nil
		}
		idx += int(count)
	}

	return nil, fmt.Errorf("sycl: device %d not found (have %d)", deviceID, idx)
}

// Destroy releases the SYCL context resources.
func (c *Context) Destroy() error {
	l := Lib()
	if l == nil {
		return nil
	}
	if c.queue != 0 {
		cuda.Ccall(l.fnQueueRelease, c.queue)
		c.queue = 0
	}
	if c.context != 0 {
		cuda.Ccall(l.fnContextRelease, c.context)
		c.context = 0
	}
	return nil
}

// Malloc allocates device memory via SYCL.
func (c *Context) Malloc(size int) (unsafe.Pointer, error) {
	l := Lib()
	if l == nil {
		return nil, fmt.Errorf("sycl not available")
	}

	var mem uintptr
	ret := cuda.Ccall(l.fnMemBufferCreate, c.context, uintptr(piMemFlagsReadWrite), uintptr(size), 0, uintptr(unsafe.Pointer(&mem)))
	if ret != 0 {
		return nil, fmt.Errorf("sycl: piMemBufferCreate(%d) failed: %d", size, ret)
	}
	return unsafe.Pointer(mem), nil //nolint:govet
}

// Free releases SYCL device memory.
func (c *Context) Free(ptr unsafe.Pointer) error {
	if ptr == nil {
		return nil
	}
	l := Lib()
	if l == nil {
		return fmt.Errorf("sycl not available")
	}
	cuda.Ccall(l.fnMemRelease, uintptr(ptr))
	return nil
}

// Memcpy copies data between host and SYCL device memory.
func (c *Context) Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	l := Lib()
	if l == nil {
		return fmt.Errorf("sycl not available")
	}

	switch kind {
	case MemcpyHostToDevice:
		// dst is pi_mem, src is host pointer.
		ret := cuda.Ccall(l.fnEnqueueMemWrite, c.queue, uintptr(dst), 1, 0, uintptr(count), uintptr(src), 0, 0, 0)
		if ret != 0 {
			return fmt.Errorf("sycl: piEnqueueMemBufferWrite failed: %d", ret)
		}
	case MemcpyDeviceToHost:
		// src is pi_mem, dst is host pointer.
		ret := cuda.Ccall(l.fnEnqueueMemRead, c.queue, uintptr(src), 1, 0, uintptr(count), uintptr(dst), 0, 0, 0)
		if ret != 0 {
			return fmt.Errorf("sycl: piEnqueueMemBufferRead failed: %d", ret)
		}
	case MemcpyDeviceToDevice:
		// Both are pi_mem.
		ret := cuda.Ccall(l.fnEnqueueMemCopy, c.queue, uintptr(src), uintptr(dst), 0, 0, uintptr(count), 0, 0, 0)
		if ret != 0 {
			return fmt.Errorf("sycl: piEnqueueMemBufferCopy failed: %d", ret)
		}
		cuda.Ccall(l.fnQueueFinish, c.queue)
	default:
		return fmt.Errorf("sycl: unsupported MemcpyKind: %d", kind)
	}
	return nil
}

// CreateStream creates a new SYCL queue.
func (c *Context) CreateStream() (*Stream, error) {
	l := Lib()
	if l == nil {
		return nil, fmt.Errorf("sycl not available")
	}
	var queue uintptr
	ret := cuda.Ccall(l.fnQueueCreate, c.context, c.device, 0, uintptr(unsafe.Pointer(&queue)))
	if ret != 0 {
		return nil, fmt.Errorf("sycl: piQueueCreate failed: %d", ret)
	}
	return &Stream{queue: queue}, nil
}

// Synchronize waits for all commands in the queue to complete.
func (s *Stream) Synchronize() error {
	l := Lib()
	if l == nil {
		return nil
	}
	ret := cuda.Ccall(l.fnQueueFinish, s.queue)
	if ret != 0 {
		return fmt.Errorf("sycl: piQueueFinish failed: %d", ret)
	}
	return nil
}

// Destroy releases the SYCL queue.
func (s *Stream) Destroy() error {
	if s.queue == 0 {
		return nil
	}
	l := Lib()
	if l == nil {
		return nil
	}
	cuda.Ccall(l.fnQueueRelease, s.queue)
	s.queue = 0
	return nil
}

// Ptr returns the underlying queue handle as unsafe.Pointer.
func (s *Stream) Ptr() unsafe.Pointer {
	return unsafe.Pointer(s.queue) //nolint:govet
}

// DeviceID returns the device ordinal.
func (c *Context) DeviceID() int {
	return c.deviceID
}

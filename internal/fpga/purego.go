//go:build linux

package fpga

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// FPGALib holds dlopen handles and resolved function pointers for the FPGA
// runtime. Currently supports Intel FPGA OpenCL Runtime and Xilinx XRT.
// Function pointers are resolved at Open() time via dlsym.
type FPGALib struct {
	handle uintptr

	// Core runtime functions.
	fnGetPlatformIDs uintptr // clGetPlatformIDs
	fnGetDeviceIDs   uintptr // clGetDeviceIDs
	fnCreateContext  uintptr // clCreateContext
	fnCreateQueue    uintptr // clCreateCommandQueue
	fnCreateBuffer   uintptr // clCreateBuffer
	fnReleaseMemObj  uintptr // clReleaseMemObject
	fnEnqueueRead    uintptr // clEnqueueReadBuffer
	fnEnqueueWrite   uintptr // clEnqueueWriteBuffer
	fnEnqueueCopy    uintptr // clEnqueueCopyBuffer
	fnFinish         uintptr // clFinish
	fnReleaseQueue   uintptr // clReleaseCommandQueue
	fnReleaseContext uintptr // clReleaseContext
}

var (
	globalLib  *FPGALib
	globalOnce sync.Once
	errGlobal  error
)

// FPGA runtime library paths (Intel and Xilinx).
const (
	intelFPGALibPath = "libintel_fpga_opencl.so"
	xilinxXRTLibPath = "libxrt_coreutil.so"
	openCLLibPath    = "libOpenCL.so.1"
)

// Open loads the FPGA runtime library via dlopen and resolves required
// function pointers. It tries OpenCL (which both Intel and Xilinx runtimes
// expose) first.
func Open() (*FPGALib, error) {
	lib := &FPGALib{}

	var err error

	// Try loading the OpenCL library (used by Intel FPGA and Xilinx XRT).
	lib.handle, err = cuda.DlopenPath(openCLLibPath)
	if err != nil {
		// Fall back to Intel-specific path.
		lib.handle, err = cuda.DlopenPath(intelFPGALibPath)
		if err != nil {
			// Fall back to Xilinx XRT.
			lib.handle, err = cuda.DlopenPath(xilinxXRTLibPath)
			if err != nil {
				return nil, fmt.Errorf("fpga: no FPGA runtime found: %w", err)
			}
		}
	}

	// Resolve OpenCL function pointers.
	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"clGetPlatformIDs", &lib.fnGetPlatformIDs},
		{"clGetDeviceIDs", &lib.fnGetDeviceIDs},
		{"clCreateContext", &lib.fnCreateContext},
		{"clCreateCommandQueue", &lib.fnCreateQueue},
		{"clCreateBuffer", &lib.fnCreateBuffer},
		{"clReleaseMemObject", &lib.fnReleaseMemObj},
		{"clEnqueueReadBuffer", &lib.fnEnqueueRead},
		{"clEnqueueWriteBuffer", &lib.fnEnqueueWrite},
		{"clEnqueueCopyBuffer", &lib.fnEnqueueCopy},
		{"clFinish", &lib.fnFinish},
		{"clReleaseCommandQueue", &lib.fnReleaseQueue},
		{"clReleaseContext", &lib.fnReleaseContext},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(lib.handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("fpga: %w", err)
		}
		*s.ptr = addr
	}

	return lib, nil
}

// Available returns true if an FPGA runtime is available on this machine.
func Available() bool {
	globalOnce.Do(func() {
		globalLib, errGlobal = Open()
	})
	return errGlobal == nil
}

// Lib returns the global FPGALib instance, or nil if not available.
func Lib() *FPGALib {
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

// Context holds an FPGA device context, command queue, and related state.
type Context struct {
	platform uintptr // cl_platform_id
	device   uintptr // cl_device_id
	context  uintptr // cl_context
	queue    uintptr // cl_command_queue
	deviceID int
}

// Stream wraps an FPGA command queue for stream-like semantics.
type Stream struct {
	queue uintptr // cl_command_queue
}

// OpenCL constants.
const (
	clDeviceTypeAccelerator = 1 << 3 // CL_DEVICE_TYPE_ACCELERATOR
	clMemReadWrite          = 1 << 0 // CL_MEM_READ_WRITE
	clTrue                  = 1
	clFalse                 = 0
)

// GetDeviceCount returns the number of FPGA accelerator devices.
func GetDeviceCount() (int, error) {
	l := Lib()
	if l == nil {
		return 0, fmt.Errorf("fpga not available")
	}

	// Get platforms.
	var numPlatforms uint32
	cuda.Ccall(l.fnGetPlatformIDs, 0, 0, uintptr(unsafe.Pointer(&numPlatforms)))
	if numPlatforms == 0 {
		return 0, nil
	}

	platforms := make([]uintptr, numPlatforms)
	cuda.Ccall(l.fnGetPlatformIDs, uintptr(numPlatforms), uintptr(unsafe.Pointer(&platforms[0])), 0)

	// Count accelerator devices across all platforms.
	var total uint32
	for _, p := range platforms {
		var count uint32
		cuda.Ccall(l.fnGetDeviceIDs, p, uintptr(clDeviceTypeAccelerator), 0, 0, uintptr(unsafe.Pointer(&count)))
		total += count
	}
	return int(total), nil
}

// NewContext creates an FPGA context on the specified device.
func NewContext(deviceID int) (*Context, error) {
	l := Lib()
	if l == nil {
		return nil, fmt.Errorf("fpga not available")
	}

	// Get platforms.
	var numPlatforms uint32
	cuda.Ccall(l.fnGetPlatformIDs, 0, 0, uintptr(unsafe.Pointer(&numPlatforms)))
	if numPlatforms == 0 {
		return nil, fmt.Errorf("fpga: no OpenCL platforms found")
	}

	platforms := make([]uintptr, numPlatforms)
	cuda.Ccall(l.fnGetPlatformIDs, uintptr(numPlatforms), uintptr(unsafe.Pointer(&platforms[0])), 0)

	// Find the device across all platforms.
	idx := 0
	for _, p := range platforms {
		var count uint32
		cuda.Ccall(l.fnGetDeviceIDs, p, uintptr(clDeviceTypeAccelerator), 0, 0, uintptr(unsafe.Pointer(&count)))
		if count == 0 {
			continue
		}
		if idx+int(count) > deviceID {
			devices := make([]uintptr, count)
			cuda.Ccall(l.fnGetDeviceIDs, p, uintptr(clDeviceTypeAccelerator), uintptr(count), uintptr(unsafe.Pointer(&devices[0])), 0)

			dev := devices[deviceID-idx]

			// Create context.
			var errCode int32
			ctx := cuda.Ccall(l.fnCreateContext, 0, 1, uintptr(unsafe.Pointer(&dev)), 0, 0, uintptr(unsafe.Pointer(&errCode)))
			if errCode != 0 {
				return nil, fmt.Errorf("fpga: clCreateContext failed: %d", errCode)
			}

			// Create command queue.
			queue := cuda.Ccall(l.fnCreateQueue, ctx, dev, 0, uintptr(unsafe.Pointer(&errCode)))
			if errCode != 0 {
				return nil, fmt.Errorf("fpga: clCreateCommandQueue failed: %d", errCode)
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

	return nil, fmt.Errorf("fpga: device %d not found (have %d)", deviceID, idx)
}

// Destroy releases the FPGA context resources.
func (c *Context) Destroy() error {
	l := Lib()
	if l == nil {
		return nil
	}
	if c.queue != 0 {
		cuda.Ccall(l.fnReleaseQueue, c.queue)
		c.queue = 0
	}
	if c.context != 0 {
		cuda.Ccall(l.fnReleaseContext, c.context)
		c.context = 0
	}
	return nil
}

// Malloc allocates device memory on the FPGA.
func (c *Context) Malloc(size int) (unsafe.Pointer, error) {
	l := Lib()
	if l == nil {
		return nil, fmt.Errorf("fpga not available")
	}

	var errCode int32
	mem := cuda.Ccall(l.fnCreateBuffer, c.context, uintptr(clMemReadWrite), uintptr(size), 0, uintptr(unsafe.Pointer(&errCode)))
	if errCode != 0 {
		return nil, fmt.Errorf("fpga: clCreateBuffer(%d) failed: %d", size, errCode)
	}
	return unsafe.Pointer(mem), nil //nolint:govet
}

// Free releases FPGA device memory.
func (c *Context) Free(ptr unsafe.Pointer) error {
	if ptr == nil {
		return nil
	}
	l := Lib()
	if l == nil {
		return fmt.Errorf("fpga not available")
	}
	cuda.Ccall(l.fnReleaseMemObj, uintptr(ptr))
	return nil
}

// Memcpy copies data between host and FPGA device memory.
func (c *Context) Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	l := Lib()
	if l == nil {
		return fmt.Errorf("fpga not available")
	}

	switch kind {
	case MemcpyHostToDevice:
		// dst is cl_mem, src is host pointer.
		ret := cuda.Ccall(l.fnEnqueueWrite, c.queue, uintptr(dst), uintptr(clTrue), 0, uintptr(count), uintptr(src), 0, 0, 0)
		if ret != 0 {
			return fmt.Errorf("fpga: clEnqueueWriteBuffer failed: %d", ret)
		}
	case MemcpyDeviceToHost:
		// src is cl_mem, dst is host pointer.
		ret := cuda.Ccall(l.fnEnqueueRead, c.queue, uintptr(src), uintptr(clTrue), 0, uintptr(count), uintptr(dst), 0, 0, 0)
		if ret != 0 {
			return fmt.Errorf("fpga: clEnqueueReadBuffer failed: %d", ret)
		}
	case MemcpyDeviceToDevice:
		// Both are cl_mem.
		ret := cuda.Ccall(l.fnEnqueueCopy, c.queue, uintptr(src), uintptr(dst), 0, 0, uintptr(count), 0, 0, 0)
		if ret != 0 {
			return fmt.Errorf("fpga: clEnqueueCopyBuffer failed: %d", ret)
		}
		// Wait for copy to finish.
		cuda.Ccall(l.fnFinish, c.queue)
	default:
		return fmt.Errorf("fpga: unsupported MemcpyKind: %d", kind)
	}
	return nil
}

// CreateStream creates a new command queue.
func (c *Context) CreateStream() (*Stream, error) {
	l := Lib()
	if l == nil {
		return nil, fmt.Errorf("fpga not available")
	}
	var errCode int32
	queue := cuda.Ccall(l.fnCreateQueue, c.context, c.device, 0, uintptr(unsafe.Pointer(&errCode)))
	if errCode != 0 {
		return nil, fmt.Errorf("fpga: clCreateCommandQueue failed: %d", errCode)
	}
	return &Stream{queue: queue}, nil
}

// Synchronize waits for all commands in the queue to complete.
func (s *Stream) Synchronize() error {
	l := Lib()
	if l == nil {
		return nil
	}
	ret := cuda.Ccall(l.fnFinish, s.queue)
	if ret != 0 {
		return fmt.Errorf("fpga: clFinish failed: %d", ret)
	}
	return nil
}

// Destroy releases the command queue.
func (s *Stream) Destroy() error {
	if s.queue == 0 {
		return nil
	}
	l := Lib()
	if l == nil {
		return nil
	}
	cuda.Ccall(l.fnReleaseQueue, s.queue)
	s.queue = 0
	return nil
}

// Ptr returns the underlying command queue handle as unsafe.Pointer.
func (s *Stream) Ptr() unsafe.Pointer {
	return unsafe.Pointer(s.queue) //nolint:govet
}

// DeviceID returns the device ordinal.
func (c *Context) DeviceID() int {
	return c.deviceID
}

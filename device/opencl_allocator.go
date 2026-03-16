//go:build opencl

package device

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/opencl"
)

// openclAllocator is the memory allocator for a specific OpenCL device.
type openclAllocator struct {
	deviceID int
}

// NewOpenCLAllocator creates a new OpenCL device memory allocator bound to the
// given device.
func NewOpenCLAllocator(deviceID int) Allocator {
	return &openclAllocator{deviceID: deviceID}
}

// Allocate allocates a cl_mem buffer of the given size on the OpenCL device.
// A temporary context is created for the allocation.
func (a *openclAllocator) Allocate(size int) (any, error) {
	if size < 0 {
		return nil, fmt.Errorf("allocation size cannot be negative: %d", size)
	}

	if size == 0 {
		return unsafe.Pointer(nil), nil
	}

	ctx, err := opencl.NewContext(a.deviceID)
	if err != nil {
		return nil, fmt.Errorf("OpenCL NewContext(%d) failed: %w", a.deviceID, err)
	}

	ptr, err := ctx.Malloc(size)
	if err != nil {
		_ = ctx.Destroy()
		return nil, fmt.Errorf("OpenCL allocation of %d bytes on device %d failed: %w", size, a.deviceID, err)
	}

	return ptr, nil
}

// Free releases an OpenCL cl_mem buffer previously allocated with Allocate.
func (a *openclAllocator) Free(ptr any) error {
	devPtr, ok := ptr.(unsafe.Pointer)
	if !ok {
		return fmt.Errorf("expected unsafe.Pointer, got %T", ptr)
	}

	if devPtr == nil {
		return nil
	}

	ctx, err := opencl.NewContext(a.deviceID)
	if err != nil {
		return fmt.Errorf("OpenCL NewContext(%d) failed: %w", a.deviceID, err)
	}
	defer func() { _ = ctx.Destroy() }()

	return ctx.Free(devPtr)
}

// Statically assert that the type implements the interface.
var _ Allocator = (*openclAllocator)(nil)

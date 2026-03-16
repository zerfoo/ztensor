//go:build rocm

package device

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/hip"
)

// rocmAllocator is the memory allocator for a specific ROCm device.
type rocmAllocator struct {
	deviceID int
}

// NewROCmAllocator creates a new ROCm device memory allocator bound to the
// given device.
func NewROCmAllocator(deviceID int) Allocator {
	return &rocmAllocator{deviceID: deviceID}
}

// Allocate allocates size bytes of HIP device memory and returns the device pointer.
// SetDevice is called before hipMalloc to ensure allocation on the correct GPU.
func (a *rocmAllocator) Allocate(size int) (any, error) {
	if size < 0 {
		return nil, fmt.Errorf("allocation size cannot be negative: %d", size)
	}

	if size == 0 {
		return unsafe.Pointer(nil), nil
	}

	if err := hip.SetDevice(a.deviceID); err != nil {
		return nil, fmt.Errorf("ROCm SetDevice(%d) failed: %w", a.deviceID, err)
	}

	ptr, err := hip.Malloc(size)
	if err != nil {
		return nil, fmt.Errorf("ROCm allocation of %d bytes on device %d failed: %w", size, a.deviceID, err)
	}

	return ptr, nil
}

// Free releases HIP device memory previously allocated with Allocate.
// SetDevice is called before hipFree to ensure correct device context.
func (a *rocmAllocator) Free(ptr any) error {
	devPtr, ok := ptr.(unsafe.Pointer)
	if !ok {
		return fmt.Errorf("expected unsafe.Pointer, got %T", ptr)
	}

	if devPtr == nil {
		return nil
	}

	if err := hip.SetDevice(a.deviceID); err != nil {
		return fmt.Errorf("ROCm SetDevice(%d) failed: %w", a.deviceID, err)
	}

	return hip.Free(devPtr)
}

// Statically assert that the type implements the interface.
var _ Allocator = (*rocmAllocator)(nil)

//go:build rocm

package device

import (
	"fmt"

	"github.com/zerfoo/ztensor/internal/hip"
)

// rocmDevice represents an AMD GPU accessible via the HIP runtime.
type rocmDevice struct {
	id        string
	deviceID  int
	allocator Allocator
}

// newROCmDevice creates a ROCm device instance for the given device ordinal.
func newROCmDevice(deviceID int) *rocmDevice {
	return &rocmDevice{
		id:        fmt.Sprintf("rocm:%d", deviceID),
		deviceID:  deviceID,
		allocator: NewROCmAllocator(deviceID),
	}
}

// ID returns the device's identifier (e.g., "rocm:0").
func (d *rocmDevice) ID() string { return d.id }

// GetAllocator returns the ROCm device memory allocator.
func (d *rocmDevice) GetAllocator() Allocator { return d.allocator }

// Type returns the ROCm device type.
func (d *rocmDevice) Type() Type { return ROCm }

// Statically assert that the type implements the interface.
var _ Device = (*rocmDevice)(nil)

func init() {
	count, err := hip.GetDeviceCount()
	if err != nil {
		return // No HIP runtime or driver; silently skip registration
	}

	for i := range count {
		registerDevice(newROCmDevice(i))
	}
}

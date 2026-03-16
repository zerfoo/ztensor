//go:build opencl

package device

import (
	"fmt"

	"github.com/zerfoo/ztensor/internal/opencl"
)

// openclDevice represents a GPU accessible via OpenCL.
type openclDevice struct {
	id        string
	deviceID  int
	allocator Allocator
}

// newOpenCLDevice creates an OpenCL device instance for the given device ordinal.
func newOpenCLDevice(deviceID int) *openclDevice {
	return &openclDevice{
		id:        fmt.Sprintf("opencl:%d", deviceID),
		deviceID:  deviceID,
		allocator: NewOpenCLAllocator(deviceID),
	}
}

// ID returns the device's identifier (e.g., "opencl:0").
func (d *openclDevice) ID() string { return d.id }

// GetAllocator returns the OpenCL device memory allocator.
func (d *openclDevice) GetAllocator() Allocator { return d.allocator }

// Type returns the OpenCL device type.
func (d *openclDevice) Type() Type { return OpenCL }

// Statically assert that the type implements the interface.
var _ Device = (*openclDevice)(nil)

func init() {
	count, err := opencl.GetDeviceCount()
	if err != nil {
		return // No OpenCL runtime or driver; silently skip registration
	}

	for i := range count {
		registerDevice(newOpenCLDevice(i))
	}
}

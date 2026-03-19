//go:build !linux

package fpga

import (
	"fmt"
	"unsafe"
)

// Available returns false on non-linux platforms.
func Available() bool { return false }

// Lib returns nil on non-linux platforms.
func Lib() *FPGALib { return nil }

// FPGALib is a stub on non-linux platforms.
type FPGALib struct{}

// Context is a stub on non-linux platforms.
type Context struct{}

// Stream is a stub on non-linux platforms.
type Stream struct{}

// MemcpyKind specifies the direction of a memory copy.
type MemcpyKind int

const (
	MemcpyHostToDevice MemcpyKind = iota
	MemcpyDeviceToHost
	MemcpyDeviceToDevice
)

// GetDeviceCount returns 0 on non-linux platforms.
func GetDeviceCount() (int, error) { return 0, fmt.Errorf("fpga not available") }

// NewContext returns an error on non-linux platforms.
func NewContext(int) (*Context, error) { return nil, fmt.Errorf("fpga not available") }

func (c *Context) Destroy() error                                    { return nil }
func (c *Context) Malloc(int) (unsafe.Pointer, error)                { return nil, fmt.Errorf("fpga not available") }
func (c *Context) Free(unsafe.Pointer) error                         { return nil }
func (c *Context) Memcpy(_, _ unsafe.Pointer, _ int, _ MemcpyKind) error { return fmt.Errorf("fpga not available") }
func (c *Context) CreateStream() (*Stream, error)                    { return nil, fmt.Errorf("fpga not available") }
func (c *Context) DeviceID() int                                     { return 0 }

func (s *Stream) Synchronize() error  { return nil }
func (s *Stream) Destroy() error      { return nil }
func (s *Stream) Ptr() unsafe.Pointer { return nil }

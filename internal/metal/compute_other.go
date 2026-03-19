//go:build !darwin

package metal

import "fmt"

// ComputePipeline is a stub on non-darwin platforms.
type ComputePipeline struct{}

// ComputeContext is a stub on non-darwin platforms.
type ComputeContext struct{}

// MTLSize represents a Metal size structure.
type MTLSize struct {
	Width  uint64
	Height uint64
	Depth  uint64
}

// BufferBinding pairs a Metal buffer handle with an offset.
type BufferBinding struct {
	Buffer uintptr
	Offset int
}

// NewComputeContext returns an error on non-darwin platforms.
func NewComputeContext(_ *Context) (*ComputeContext, error) {
	return nil, fmt.Errorf("metal compute not available")
}

// GetPipeline returns an error on non-darwin platforms.
func (cc *ComputeContext) GetPipeline(_ string) (*ComputePipeline, error) {
	return nil, fmt.Errorf("metal compute not available")
}

// Dispatch returns an error on non-darwin platforms.
func (cc *ComputeContext) Dispatch(_ *ComputePipeline, _, _ MTLSize, _ map[int]BufferBinding, _ map[int][]byte, _ ...map[int]int) error {
	return fmt.Errorf("metal compute not available")
}

// Destroy is a no-op on non-darwin platforms.
func (cc *ComputeContext) Destroy() {}

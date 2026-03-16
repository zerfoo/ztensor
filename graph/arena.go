package graph

import "github.com/zerfoo/ztensor/tensor"

// BufferArena pre-allocates tensor buffers for use by an ExecutionPlan.
// All buffers are created once and reused across Run() calls.
// Frozen slots (parameters, constants) are not zeroed on Reset.
type BufferArena[T tensor.Numeric] struct {
	buffers []*tensor.TensorNumeric[T]
	frozen  []bool
}

// NewBufferArena pre-allocates one tensor per shape.
func NewBufferArena[T tensor.Numeric](shapes [][]int) *BufferArena[T] {
	a := &BufferArena[T]{
		buffers: make([]*tensor.TensorNumeric[T], len(shapes)),
		frozen:  make([]bool, len(shapes)),
	}
	for i, shape := range shapes {
		t, _ := tensor.New[T](shape, nil)
		a.buffers[i] = t
	}
	return a
}

// Get returns the pre-allocated buffer at index idx.
func (a *BufferArena[T]) Get(idx int) *tensor.TensorNumeric[T] {
	return a.buffers[idx]
}

// Set replaces the buffer at idx with the given tensor and optionally
// marks it as frozen (skip during Reset).
func (a *BufferArena[T]) Set(idx int, t *tensor.TensorNumeric[T], freeze bool) {
	a.buffers[idx] = t
	a.frozen[idx] = freeze
}

// Len returns the number of buffer slots.
func (a *BufferArena[T]) Len() int {
	return len(a.buffers)
}

// Reset zeros all non-frozen buffer data for the next execution step.
func (a *BufferArena[T]) Reset() {
	for i, buf := range a.buffers {
		if a.frozen[i] {
			continue
		}
		data := buf.Data()
		clear(data)
	}
}

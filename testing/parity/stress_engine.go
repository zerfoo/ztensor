package parity

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/tensor"
)

// hostArenaStorage backs a float32 tensor with a span of a host-backed
// cuda.ArenaPool (the poison-test / WolfHazard pattern from
// graph/save_for_backward_test.go), implementing tensor.PinnableStorage so
// the save-for-backward contract pins it. This is the CPU stand-in for
// GPUStorage-over-the-CUDA-arena: same lifetime rules, dereferenceable
// without a GPU.
type hostArenaStorage struct {
	arena *cuda.ArenaPool
	ptr   unsafe.Pointer
	n     int
}

func (s *hostArenaStorage) Len() int                { return s.n }
func (s *hostArenaStorage) Slice() []float32        { return unsafe.Slice((*float32)(s.ptr), s.n) }
func (s *hostArenaStorage) Set(d []float32)         { copy(s.Slice(), d) }
func (s *hostArenaStorage) DeviceType() device.Type { return device.CPU }
func (s *hostArenaStorage) PinForBackward() bool    { return s.arena.Pin(s.ptr, s.n*4) }
func (s *hostArenaStorage) UnpinForBackward()       { s.arena.Unpin(s.ptr) }

var (
	_ tensor.Storage[float32] = (*hostArenaStorage)(nil)
	_ tensor.PinnableStorage  = (*hostArenaStorage)(nil)
)

// StressEngine is the CI candidate engine: it delegates every operation to
// an inner CPU engine, then relocates the result into a host-backed arena.
// Every intermediate a node computes through the engine -- cached forward
// outputs, layernorm statistics, fixture caches -- therefore lives in
// arena memory with GPU lifetime semantics: an arena Reset between forward
// and backward recycles (and, under poison mode, NaN-fills) every span that
// is not pinned via the save-for-backward contract. This is what lets CI
// catch the zerfoo#842 cached-intermediate corruption class without a GPU.
//
// The embedded Engine serves the methods not overridden below; the
// overridden set covers everything the gradcheck registry ops and the
// parity fixtures call. StressEngine deliberately does NOT implement the
// optional fused interfaces (e.g. compute.TransposeBMatMuler), matching the
// plain CPU reference engine, so both sides take identical numeric paths.
type StressEngine struct {
	compute.Engine[float32]
	arena *cuda.ArenaPool
}

// NewStressEngine wraps inner with a fresh host-backed arena of arenaBytes
// capacity. Size it generously for the op set (the stress comes from the
// schedule's Reset, not from capacity): an exhausted host-backed arena
// falls back to the CUDA MemPool, which has no device in CI and errors.
func NewStressEngine(inner compute.Engine[float32], arenaBytes int) *StressEngine {
	return &StressEngine{
		Engine: inner,
		arena:  cuda.NewHostBackedArenaForTesting(make([]byte, arenaBytes)),
	}
}

// Arena exposes the backing arena (tests assert pin accounting on it).
func (e *StressEngine) Arena() *cuda.ArenaPool { return e.arena }

// ResetArena rewinds the arena exactly like the GPU engine's ResetPool:
// pinned spans survive, everything else becomes reusable (and is poisoned
// under ZTENSOR_ARENA_POISON semantics). Wire it as Side.Reset.
func (e *StressEngine) ResetArena() { e.arena.Reset() }

// relocate copies t into a fresh arena span and returns an arena-backed
// tensor with the same shape and values. dst-carrying calls and empty
// tensors pass through untouched.
func (e *StressEngine) relocate(t *t32, err error) (*t32, error) {
	if err != nil || t == nil || len(t.Data()) == 0 {
		return t, err
	}
	n := len(t.Data())
	ptr, aerr := e.arena.Alloc(0, n*4)
	if aerr != nil {
		return nil, fmt.Errorf("parity stress engine: arena alloc of %d bytes: %w", n*4, aerr)
	}
	st := &hostArenaStorage{arena: e.arena, ptr: ptr, n: n}
	copy(st.Slice(), t.Data())
	out, terr := tensor.NewWithStorage(t.Shape(), tensor.Storage[float32](st))
	if terr != nil {
		return nil, fmt.Errorf("parity stress engine: wrapping arena storage: %w", terr)
	}
	return out, nil
}

// The overrides below are mechanical: delegate, then relocate. Calls that
// supply an explicit dst keep the inner engine's dst semantics and are not
// relocated (the registry ops never pass dst).

func (e *StressEngine) Add(ctx context.Context, a, b *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Add(ctx, a, b, dst...)
	}
	return e.relocate(e.Engine.Add(ctx, a, b))
}

func (e *StressEngine) Sub(ctx context.Context, a, b *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Sub(ctx, a, b, dst...)
	}
	return e.relocate(e.Engine.Sub(ctx, a, b))
}

func (e *StressEngine) Mul(ctx context.Context, a, b *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Mul(ctx, a, b, dst...)
	}
	return e.relocate(e.Engine.Mul(ctx, a, b))
}

func (e *StressEngine) Div(ctx context.Context, a, b *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Div(ctx, a, b, dst...)
	}
	return e.relocate(e.Engine.Div(ctx, a, b))
}

func (e *StressEngine) Pow(ctx context.Context, base, exp *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Pow(ctx, base, exp, dst...)
	}
	return e.relocate(e.Engine.Pow(ctx, base, exp))
}

func (e *StressEngine) MatMul(ctx context.Context, a, b *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.MatMul(ctx, a, b, dst...)
	}
	return e.relocate(e.Engine.MatMul(ctx, a, b))
}

func (e *StressEngine) Transpose(ctx context.Context, a *t32, axes []int, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Transpose(ctx, a, axes, dst...)
	}
	return e.relocate(e.Engine.Transpose(ctx, a, axes))
}

func (e *StressEngine) Reshape(ctx context.Context, a *t32, shape []int, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Reshape(ctx, a, shape, dst...)
	}
	return e.relocate(e.Engine.Reshape(ctx, a, shape))
}

func (e *StressEngine) Exp(ctx context.Context, a *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Exp(ctx, a, dst...)
	}
	return e.relocate(e.Engine.Exp(ctx, a))
}

func (e *StressEngine) Log(ctx context.Context, a *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Log(ctx, a, dst...)
	}
	return e.relocate(e.Engine.Log(ctx, a))
}

func (e *StressEngine) Sqrt(ctx context.Context, a *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Sqrt(ctx, a, dst...)
	}
	return e.relocate(e.Engine.Sqrt(ctx, a))
}

func (e *StressEngine) Rsqrt(ctx context.Context, a *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Rsqrt(ctx, a, dst...)
	}
	return e.relocate(e.Engine.Rsqrt(ctx, a))
}

func (e *StressEngine) Sin(ctx context.Context, a *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Sin(ctx, a, dst...)
	}
	return e.relocate(e.Engine.Sin(ctx, a))
}

func (e *StressEngine) Cos(ctx context.Context, a *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Cos(ctx, a, dst...)
	}
	return e.relocate(e.Engine.Cos(ctx, a))
}

func (e *StressEngine) Tanh(ctx context.Context, a *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Tanh(ctx, a, dst...)
	}
	return e.relocate(e.Engine.Tanh(ctx, a))
}

func (e *StressEngine) TanhPrime(ctx context.Context, a, upstream *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.TanhPrime(ctx, a, upstream, dst...)
	}
	return e.relocate(e.Engine.TanhPrime(ctx, a, upstream))
}

func (e *StressEngine) UnaryOp(ctx context.Context, a *t32, op func(float32) float32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.UnaryOp(ctx, a, op, dst...)
	}
	return e.relocate(e.Engine.UnaryOp(ctx, a, op))
}

func (e *StressEngine) AddScalar(ctx context.Context, a *t32, scalar float32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.AddScalar(ctx, a, scalar, dst...)
	}
	return e.relocate(e.Engine.AddScalar(ctx, a, scalar))
}

func (e *StressEngine) MulScalar(ctx context.Context, a *t32, scalar float32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.MulScalar(ctx, a, scalar, dst...)
	}
	return e.relocate(e.Engine.MulScalar(ctx, a, scalar))
}

func (e *StressEngine) DivScalar(ctx context.Context, a *t32, scalar float32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.DivScalar(ctx, a, scalar, dst...)
	}
	return e.relocate(e.Engine.DivScalar(ctx, a, scalar))
}

func (e *StressEngine) Softmax(ctx context.Context, a *t32, axis int, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Softmax(ctx, a, axis, dst...)
	}
	return e.relocate(e.Engine.Softmax(ctx, a, axis))
}

func (e *StressEngine) ReduceSum(ctx context.Context, a *t32, axis int, keepDims bool, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.ReduceSum(ctx, a, axis, keepDims, dst...)
	}
	return e.relocate(e.Engine.ReduceSum(ctx, a, axis, keepDims))
}

func (e *StressEngine) ReduceMean(ctx context.Context, a *t32, axis int, keepDims bool, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.ReduceMean(ctx, a, axis, keepDims, dst...)
	}
	return e.relocate(e.Engine.ReduceMean(ctx, a, axis, keepDims))
}

func (e *StressEngine) ReduceMax(ctx context.Context, a *t32, axis int, keepDims bool, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.ReduceMax(ctx, a, axis, keepDims, dst...)
	}
	return e.relocate(e.Engine.ReduceMax(ctx, a, axis, keepDims))
}

func (e *StressEngine) Repeat(ctx context.Context, a *t32, axis, repetitions int, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.Repeat(ctx, a, axis, repetitions, dst...)
	}
	return e.relocate(e.Engine.Repeat(ctx, a, axis, repetitions))
}

// Dropout / DropoutBackward delegate to the inner engine's Dropouter
// capability and relocate the result into the arena, so the parity harness
// exercises the dropout op under the same reset-between-fwd-bwd schedules as
// every other op. The masked output is a pure function of (seed, offset, p) --
// it survives a relocate intact, and (per the recompute-in-backward design) no
// mask is saved across resets.
func (e *StressEngine) Dropout(ctx context.Context, a *t32, p float64, seed uint64, training bool, dst ...*t32) (*t32, error) {
	d := e.Engine.(compute.Dropouter[float32]) //nolint:errcheck // inner CPU engine implements Dropouter
	if len(dst) > 0 {
		return d.Dropout(ctx, a, p, seed, training, dst...)
	}
	return e.relocate(d.Dropout(ctx, a, p, seed, training))
}

func (e *StressEngine) DropoutBackward(ctx context.Context, g *t32, p float64, seed uint64, training bool, dst ...*t32) (*t32, error) {
	d := e.Engine.(compute.Dropouter[float32]) //nolint:errcheck // inner CPU engine implements Dropouter
	if len(dst) > 0 {
		return d.DropoutBackward(ctx, g, p, seed, training, dst...)
	}
	return e.relocate(d.DropoutBackward(ctx, g, p, seed, training))
}

func (e *StressEngine) HadamardTransform(ctx context.Context, a *t32, dst ...*t32) (*t32, error) {
	if len(dst) > 0 {
		return e.Engine.HadamardTransform(ctx, a, dst...)
	}
	return e.relocate(e.Engine.HadamardTransform(ctx, a))
}

package graph

import (
	"context"
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// fakePinStorage implements tensor.Storage[float32] and
// tensor.PinnableStorage, counting pin/unpin calls so the graph's
// save-for-backward lifecycle can be asserted without an arena.
type fakePinStorage struct {
	data   []float32
	pins   int
	unpins int
}

func (s *fakePinStorage) Len() int                { return len(s.data) }
func (s *fakePinStorage) Slice() []float32        { return s.data }
func (s *fakePinStorage) Set(d []float32)         { copy(s.data, d) }
func (s *fakePinStorage) DeviceType() device.Type { return device.CPU }
func (s *fakePinStorage) PinForBackward() bool    { s.pins++; return true }
func (s *fakePinStorage) UnpinForBackward()       { s.unpins++ }

var (
	_ tensor.Storage[float32] = (*fakePinStorage)(nil)
	_ tensor.PinnableStorage  = (*fakePinStorage)(nil)
)

func newFakePinTensor(t *testing.T, n int) (*tensor.TensorNumeric[float32], *fakePinStorage) {
	t.Helper()
	s := &fakePinStorage{data: make([]float32, n)}
	tt, err := tensor.NewWithStorage([]int{n}, tensor.Storage[float32](s))
	if err != nil {
		t.Fatalf("NewWithStorage: %v", err)
	}
	return tt, s
}

// savingNode is a SaverAware pass-through node that registers toSave during
// Forward. backwardErr, when set, is returned by Backward.
type savingNode struct {
	NoParameters[float32]
	saver       Saver[float32]
	toSave      []*tensor.TensorNumeric[float32]
	backwardErr error
}

func (n *savingNode) OpType() string                     { return "SavingNode" }
func (n *savingNode) Attributes() map[string]interface{} { return nil }
func (n *savingNode) OutputShape() []int                 { return nil }
func (n *savingNode) SetSaver(s Saver[float32])          { n.saver = s }

func (n *savingNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if n.saver != nil {
		n.saver.SaveForBackward(n.toSave...)
	}
	return inputs[0], nil
}

func (n *savingNode) Backward(_ context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	if n.backwardErr != nil {
		return nil, n.backwardErr
	}
	return []*tensor.TensorNumeric[float32]{outputGradient}, nil
}

func buildSavingGraph(t *testing.T, node *savingNode) *Graph[float32] {
	t.Helper()
	b := NewBuilder[float32](nil)
	in := b.Input([]int{2})
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	return g
}

func mustTensor(t *testing.T, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	tt, err := tensor.New([]int{len(data)}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	return tt
}

// TestSaveForBackward_PinUnpinLifecycle: save during Forward pins; the
// node's Backward returning unpins; multi-tensor saves are all tracked.
func TestSaveForBackward_PinUnpinLifecycle(t *testing.T) {
	t1, s1 := newFakePinTensor(t, 4)
	t2, s2 := newFakePinTensor(t, 8)
	node := &savingNode{toSave: []*tensor.TensorNumeric[float32]{t1, t2}}
	g := buildSavingGraph(t, node)

	if node.saver == nil {
		t.Fatal("Builder.Build did not wire a Saver into the SaverAware node")
	}

	x := mustTensor(t, []float32{1, 2})
	if _, err := g.Forward(context.Background(), x); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if s1.pins != 1 || s2.pins != 1 || s1.unpins != 0 || s2.unpins != 0 {
		t.Fatalf("after Forward: pins=(%d,%d) unpins=(%d,%d), want (1,1)/(0,0)", s1.pins, s2.pins, s1.unpins, s2.unpins)
	}
	if got := g.SavedForBackward(node); len(got) != 2 || got[0] != t1 || got[1] != t2 {
		t.Fatalf("SavedForBackward = %v, want [t1 t2]", got)
	}

	if err := g.Backward(context.Background(), types.FullBackprop, mustTensor(t, []float32{1, 1})); err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if s1.unpins != 1 || s2.unpins != 1 {
		t.Fatalf("after Backward: unpins=(%d,%d), want (1,1)", s1.unpins, s2.unpins)
	}
	if got := g.SavedForBackward(node); got != nil {
		t.Fatalf("SavedForBackward after Backward = %v, want nil", got)
	}
}

// TestSaveForBackward_ForwardOnlyReleasedOnNextForward: an inference loop
// (Forward without Backward) must not accumulate pins; the next Forward
// releases the previous pass's saved sets.
func TestSaveForBackward_ForwardOnlyReleasedOnNextForward(t *testing.T) {
	t1, s1 := newFakePinTensor(t, 4)
	node := &savingNode{toSave: []*tensor.TensorNumeric[float32]{t1}}
	g := buildSavingGraph(t, node)

	x := mustTensor(t, []float32{1, 2})
	for i := 1; i <= 3; i++ {
		if _, err := g.Forward(context.Background(), x); err != nil {
			t.Fatalf("Forward %d: %v", i, err)
		}
		if s1.pins != i || s1.unpins != i-1 {
			t.Fatalf("after Forward %d: pins=%d unpins=%d, want pins=%d unpins=%d", i, s1.pins, s1.unpins, i, i-1)
		}
	}
}

// TestSaveForBackward_BackwardErrorStillReleases: a failing Backward must
// not leak the node's pins into the next step.
func TestSaveForBackward_BackwardErrorStillReleases(t *testing.T) {
	t1, s1 := newFakePinTensor(t, 4)
	node := &savingNode{
		toSave:      []*tensor.TensorNumeric[float32]{t1},
		backwardErr: context.DeadlineExceeded,
	}
	g := buildSavingGraph(t, node)

	if _, err := g.Forward(context.Background(), mustTensor(t, []float32{1, 2})); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if err := g.Backward(context.Background(), types.FullBackprop, mustTensor(t, []float32{1, 1})); err == nil {
		t.Fatal("Backward: expected error")
	}
	if s1.unpins != 1 {
		t.Fatalf("unpins after failed Backward = %d, want 1", s1.unpins)
	}
}

// TestSaveForBackward_NonPinnableStorageNoop: CPU-backed tensors are
// recorded but not pinned -- CPU engines are unaffected by the contract.
func TestSaveForBackward_NonPinnableStorageNoop(t *testing.T) {
	cpuT := mustTensor(t, []float32{3, 4})
	node := &savingNode{toSave: []*tensor.TensorNumeric[float32]{cpuT, nil}}
	g := buildSavingGraph(t, node)

	if _, err := g.Forward(context.Background(), mustTensor(t, []float32{1, 2})); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	// Nil tensors are ignored; the CPU tensor is recorded without a pin.
	if got := g.SavedForBackward(node); len(got) != 1 || got[0] != cpuT {
		t.Fatalf("SavedForBackward = %v, want [cpuT]", got)
	}
	if err := g.Backward(context.Background(), types.FullBackprop, mustTensor(t, []float32{1, 1})); err != nil {
		t.Fatalf("Backward: %v", err)
	}
}

// ---------------------------------------------------------------------------
// THE KEY REGRESSION TEST (S2.1.1): the Wolf hazard schedule against a REAL
// arena. A node writes a forward intermediate into arena-backed storage
// (mirroring the zerfoo#842 LayerNorm cached-variance pattern), the arena is
// Reset between forward and backward (Wolf's per-sample ResetPool), and
// Backward reads the intermediate.
//
//   - With SaveForBackward, the pin survives the Reset and Backward reads the
//     CORRECT value.
//   - With a raw struct-field cache (the deprecated pattern) under
//     ZTENSOR_ARENA_POISON=1, Backward reads NaN -- proving the contract is
//     what fixes the bug class, and the poison mode is what exposes it.
// ---------------------------------------------------------------------------

// arenaF32Storage backs a tensor with a span of a host-backed cuda.ArenaPool
// (the poison-test pattern), implementing tensor.PinnableStorage so the
// graph's save-for-backward contract pins it.
type arenaF32Storage struct {
	arena *cuda.ArenaPool
	ptr   unsafe.Pointer
	n     int
}

func (s *arenaF32Storage) Len() int                { return s.n }
func (s *arenaF32Storage) Slice() []float32        { return unsafe.Slice((*float32)(s.ptr), s.n) }
func (s *arenaF32Storage) Set(d []float32)         { copy(s.Slice(), d) }
func (s *arenaF32Storage) DeviceType() device.Type { return device.CPU }
func (s *arenaF32Storage) PinForBackward() bool    { return s.arena.Pin(s.ptr, s.n*4) }
func (s *arenaF32Storage) UnpinForBackward()       { s.arena.Unpin(s.ptr) }

var (
	_ tensor.Storage[float32] = (*arenaF32Storage)(nil)
	_ tensor.PinnableStorage  = (*arenaF32Storage)(nil)
)

// varianceCachingNode mirrors the LayerNorm-variance bug shape: Forward
// computes an intermediate into arena memory and caches the tensor in a
// struct field; Backward reads it. useContract toggles SaveForBackward.
type varianceCachingNode struct {
	NoParameters[float32]
	arena       *cuda.ArenaPool
	useContract bool
	saver       Saver[float32]
	cachedVar   *tensor.TensorNumeric[float32] // the deprecated raw cache
	readBack    []float32                      // what Backward observed
}

func (n *varianceCachingNode) OpType() string                     { return "VarianceCaching" }
func (n *varianceCachingNode) Attributes() map[string]interface{} { return nil }
func (n *varianceCachingNode) OutputShape() []int                 { return nil }
func (n *varianceCachingNode) SetSaver(s Saver[float32])          { n.saver = s }

func (n *varianceCachingNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	const elems = 64 // 256 bytes: one arena alignment quantum
	ptr, err := n.arena.Alloc(0, elems*4)
	if err != nil {
		return nil, err
	}
	st := &arenaF32Storage{arena: n.arena, ptr: ptr, n: elems}
	v, err := tensor.NewWithStorage([]int{elems}, tensor.Storage[float32](st))
	if err != nil {
		return nil, err
	}
	vals := st.Slice()
	for i := range vals {
		vals[i] = 2.5 // the "variance" intermediate
	}
	n.cachedVar = v
	if n.useContract && n.saver != nil {
		n.saver.SaveForBackward(v)
	}
	return inputs[0], nil
}

func (n *varianceCachingNode) Backward(_ context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	n.readBack = append([]float32(nil), n.cachedVar.Data()...)
	return []*tensor.TensorNumeric[float32]{outputGradient}, nil
}

func runWolfHazardSchedule(t *testing.T, useContract bool) (*varianceCachingNode, *cuda.ArenaPool) {
	t.Helper()
	restore := cuda.SetArenaPoisonEnabledForTesting(true)
	t.Cleanup(restore)
	cuda.SetArenaPoisonFill(cuda.HostPoisonFillForTesting)
	t.Cleanup(func() { cuda.SetArenaPoisonFill(nil) })

	arena := cuda.NewHostBackedArenaForTesting(make([]byte, 8192))
	node := &varianceCachingNode{arena: arena, useContract: useContract}

	b := NewBuilder[float32](nil)
	in := b.Input([]int{2})
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Forward: the node materializes its intermediate in the arena.
	if _, err := g.Forward(context.Background(), mustTensor(t, []float32{1, 2})); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	// Step boundary: Wolf's per-sample ResetPool resets the arena BETWEEN
	// forward and backward (the hazard schedule of zerfoo#842/#845).
	arena.Reset()
	// Backward: the node reads its intermediate.
	if err := g.Backward(context.Background(), types.FullBackprop, mustTensor(t, []float32{1, 1})); err != nil {
		t.Fatalf("Backward: %v", err)
	}
	return node, arena
}

func TestSaveForBackward_WolfHazard_ContractReadsCorrectValue(t *testing.T) {
	node, arena := runWolfHazardSchedule(t, true)

	for i, v := range node.readBack {
		if v != 2.5 {
			t.Fatalf("readBack[%d] = %v, want 2.5 (SaveForBackward must protect the intermediate across Reset)", i, v)
		}
	}
	// The graph unpinned the saved set after Backward...
	if got := arena.PinnedBytes(); got != 0 {
		t.Fatalf("PinnedBytes after Backward = %d, want 0", got)
	}
	// ...so the next Reset reclaims (and poisons) the retained span.
	arena.Reset()
	if got := arena.UsedBytes(); got != 0 {
		t.Fatalf("UsedBytes after post-step Reset = %d, want 0", got)
	}
	if v := node.cachedVar.Data()[0]; !math.IsNaN(float64(v)) {
		t.Fatalf("intermediate after release+Reset = %v, want NaN (poison reclaims released span)", v)
	}
}

func TestSaveForBackward_WolfHazard_RawCacheReadsPoison(t *testing.T) {
	node, _ := runWolfHazardSchedule(t, false)

	for i, v := range node.readBack {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("readBack[%d] = %v, want NaN (raw struct-field cache must read poison after Reset)", i, v)
		}
	}
}

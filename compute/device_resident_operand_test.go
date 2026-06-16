package compute

import (
	"context"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/internal/gpuapi"
	"github.com/zerfoo/ztensor/log"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// countingRuntime is a fakeRuntime that counts host->device Memcpy calls so a
// test can assert exactly how many times an operand is uploaded. All other
// behavior mirrors fakeRuntime (host-to-host copies).
//
// This is the instrument for the device-resident-operand invariant (ADR 075
// L1, T14L1.1): once a weight/bias parameter has been promoted to GPUStorage,
// every subsequent read of it -- forward, backward, or through a transpose /
// reshape view the backward constructs -- must be zero-copy. A re-upload of an
// already-device-resident operand is the 211 GB/epoch firehose this lever kills.
type countingRuntime struct {
	h2d int // count of MemcpyHostToDevice calls
}

func (*countingRuntime) DeviceType() device.Type        { return device.CPU }
func (*countingRuntime) SetDevice(int) error            { return nil }
func (*countingRuntime) GetDeviceCount() (int, error)   { return 1, nil }
func (*countingRuntime) Malloc(int) (unsafe.Pointer, error) {
	// Backing storage is provided by the fakeMemPool; weight uploads in this
	// test go through the pool, and the per-op getDevicePtr H2D path also uses
	// the pool. Malloc is only hit by NewGPUStorageFromPtr-style direct allocs,
	// which this test does not exercise, so a nil pointer is sufficient.
	return nil, nil
}
func (*countingRuntime) Free(unsafe.Pointer) error { return nil }
func (r *countingRuntime) Memcpy(dst, src unsafe.Pointer, count int, kind gpuapi.MemcpyKind) error {
	if kind == gpuapi.MemcpyHostToDevice {
		r.h2d++
	}
	if dst != nil && src != nil {
		copy(unsafe.Slice((*byte)(dst), count), unsafe.Slice((*byte)(src), count))
	}
	return nil
}
func (r *countingRuntime) MemcpyAsync(dst, src unsafe.Pointer, count int, kind gpuapi.MemcpyKind, _ gpuapi.Stream) error {
	return r.Memcpy(dst, src, count, kind)
}
func (*countingRuntime) MemsetAsync(unsafe.Pointer, int, int, gpuapi.Stream) error { return nil }
func (*countingRuntime) MemcpyPeer(unsafe.Pointer, int, unsafe.Pointer, int, int) error {
	return nil
}
func (*countingRuntime) CreateStream() (gpuapi.Stream, error) { return fakeStream{}, nil }

var _ gpuapi.Runtime = (*countingRuntime)(nil)

func newCountingGPUEngine(t *testing.T) (*GPUEngine[float32], *countingRuntime, *fakeMemPool) {
	t.Helper()
	rt := &countingRuntime{}
	pool := newFakeMemPool()
	eng := &GPUEngine[float32]{
		cpu:           NewCPUEngine[float32](numeric.Float32Ops{}),
		runtime:       rt,
		pool:          pool,
		stream:        fakeStream{},
		logger:        log.Nop(),
		deviceID:      0,
		dtype:         DTypeF32,
		maxAllocBytes: DefaultMaxAllocBytes,
	}
	return eng, rt, pool
}

// TestDeviceResidentOperand_NoReuploadAfterPromotion is the T14L1.1 universal
// quality gate: a parameter that has been promoted to GPUStorage (the form
// UploadWeights leaves it in) must be read zero-copy on every subsequent
// getDevicePtr -- the read the forward MatMul does AND the read the backward's
// transpose/dW path does. Reading it twice must trigger ZERO host->device
// copies, not one-per-read.
//
// A CPU-backed tensor, by contrast, uploads once per read; this test pins both
// halves of the contract so a regression that re-homes a weight operand to the
// host (dropping device residency through a view/transpose) is caught here
// instead of only by a 211 GB/epoch nsys profile on the GB10.
func TestDeviceResidentOperand_NoReuploadAfterPromotion(t *testing.T) {
	eng, rt, _ := newCountingGPUEngine(t)
	ctx := context.Background()
	_ = ctx

	// A [256,256] weight-class operand -- the exact shape the T11.0c backtrace
	// found re-uploaded 3,053x/step.
	const rows, cols = 256, 256
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i%7) * 0.5
	}
	w, err := tensor.New[float32]([]int{rows, cols}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	// Baseline: a CPU-backed operand uploads once per read (the firehose).
	if _, c1, err := getDevicePtr(eng, w); err != nil {
		t.Fatalf("getDevicePtr cpu read 1: %v", err)
	} else {
		c1()
	}
	if rt.h2d != 1 {
		t.Fatalf("CPU-backed first read: want 1 H2D, got %d", rt.h2d)
	}
	if _, c2, err := getDevicePtr(eng, w); err != nil {
		t.Fatalf("getDevicePtr cpu read 2: %v", err)
	} else {
		c2()
	}
	if rt.h2d != 2 {
		t.Fatalf("CPU-backed second read re-uploads (the firehose): want 2 H2D, got %d", rt.h2d)
	}

	// Promote the operand to GPUStorage, exactly as UploadWeights does (a
	// non-owning view over a device pointer). From here, every read must be
	// zero-copy regardless of how many times forward+backward touch it.
	view := tensor.NewGPUStorageViewFromPtr[float32](unsafe.Pointer(&data[0]), rows*cols, eng.deviceID)
	w.SetStorage(view)

	before := rt.h2d
	for i := 0; i < 5; i++ { // forward read + backward reads, several ops
		ptr, cleanup, err := getDevicePtr(eng, w)
		if err != nil {
			t.Fatalf("getDevicePtr gpu read %d: %v", i, err)
		}
		if ptr == nil {
			t.Fatalf("getDevicePtr gpu read %d returned nil ptr", i)
		}
		cleanup()
	}
	if got := rt.h2d - before; got != 0 {
		t.Fatalf("device-resident operand re-uploaded %d times across 5 reads; want 0 "+
			"(this is the T14L1.1 firehose -- a GPUStorage operand must be zero-copy)", got)
	}
}

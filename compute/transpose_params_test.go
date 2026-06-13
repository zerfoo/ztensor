package compute

import (
	"strings"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// allocRuntime wraps fakeRuntime with a Malloc that returns real (host)
// memory and records frees, so transposeNDDeviceParams' alloc + upload +
// Close-free lifecycle can be asserted without CUDA hardware.
type allocRuntime struct {
	fakeRuntime
	allocs int
	frees  int
	bufs   []unsafe.Pointer
}

func (r *allocRuntime) Malloc(byteSize int) (unsafe.Pointer, error) {
	buf := make([]byte, byteSize)
	r.allocs++
	ptr := unsafe.Pointer(&buf[0])
	r.bufs = append(r.bufs, ptr)
	return ptr, nil
}

func (r *allocRuntime) Free(unsafe.Pointer) error {
	r.frees++
	return nil
}

// TestTransposeNDDeviceParams_CacheAndLayout: one device block per distinct
// signature, laid out as {inStrides, outStrides, perm}, cache hits return
// the identical pointer, and Close frees every block. The device residency
// itself is the contract that keeps CUDA-graph replays away from recycled
// Go heap (Wolf T9.1 illegal-memory-access).
func TestTransposeNDDeviceParams_CacheAndLayout(t *testing.T) {
	rt := &allocRuntime{}
	eng := &GPUEngine[float32]{runtime: rt}

	in := []int32{12, 4, 1}
	out := []int32{48, 4, 1}
	perm := []int32{1, 0, 2}

	p1, err := eng.transposeNDDeviceParams(in, out, perm)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	// Layout: 3*rank int32 contiguous.
	got := unsafe.Slice((*int32)(p1), 9)
	want := []int32{12, 4, 1, 48, 4, 1, 1, 0, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("block[%d] = %d, want %d", i, got[i], want[i])
		}
	}

	p2, err := eng.transposeNDDeviceParams(in, out, perm)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}
	if p1 != p2 {
		t.Error("same signature must return the cached pointer")
	}
	if rt.allocs != 1 {
		t.Errorf("allocs = %d, want 1 (cache hit must not re-allocate)", rt.allocs)
	}

	p3, err := eng.transposeNDDeviceParams([]int32{4, 1}, []int32{2, 1}, []int32{1, 0})
	if err != nil {
		t.Fatalf("third call: %v", err)
	}
	if p3 == p1 {
		t.Error("distinct signatures must not share a block")
	}
	if rt.allocs != 2 {
		t.Errorf("allocs = %d, want 2", rt.allocs)
	}

	// Close frees every cached block exactly once.
	eng.transposeParamsMu.Lock()
	n := len(eng.transposeParams)
	eng.transposeParamsMu.Unlock()
	if n != 2 {
		t.Fatalf("cache size = %d, want 2", n)
	}
	_ = eng.Close()
	if rt.frees < 2 {
		t.Errorf("frees = %d, want >= 2 (Close must free the parameter blocks)", rt.frees)
	}
}

// TestTransposeNDDeviceParams_RefusesNewSignatureDuringCapture: allocating
// a parameter block requires cudaMalloc, which is illegal during stream
// capture -- a cold-cache transpose inside a capture region must fail
// loudly instead of corrupting the graph.
func TestTransposeNDDeviceParams_RefusesNewSignatureDuringCapture(t *testing.T) {
	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restoreStatus()

	rt := &allocRuntime{}
	eng := &GPUEngine[float32]{runtime: rt, stream: fakePtrStream{}}

	// Warm one signature outside capture... not possible with the stub
	// already active; assert the cold-cache failure instead.
	_, err := eng.transposeNDDeviceParams([]int32{4, 1}, []int32{2, 1}, []int32{1, 0})
	if err == nil {
		t.Fatal("expected error for new signature during capture")
	}
	if !strings.Contains(err.Error(), "capture") {
		t.Errorf("error %q should mention capture", err)
	}
	if rt.allocs != 0 {
		t.Errorf("allocs = %d, want 0 (no allocation during capture)", rt.allocs)
	}
}

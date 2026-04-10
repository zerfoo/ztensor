package compute

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestGPUEngine_Reshape_HonorsDst is the regression test for zerfoo/ztensor#81.
// Pre-fix, GPUEngine.Reshape's zero-copy GPUStorage fast-path returned a fresh
// tensor aliasing the source storage but ignored the caller-provided dst,
// leaving dst's pre-allocated (zero) buffer untouched. Callers that discarded
// the return value (e.g. zerfoo PatchTST GPU backward) silently fed all-zero
// gradients into encoderBackward and froze training loss. The fix mutates dst
// to alias the reshaped view; this test asserts that contract.
func TestGPUEngine_Reshape_HonorsDst(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	ops := numeric.Float32Ops{}
	eng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()

	// Source: a [4,4] tensor on the GPU with non-zero data.
	src := make([]float32, 16)
	for i := range src {
		src[i] = float32(i + 1) // 1..16
	}
	srcGS, err := tensor.NewGPUStorageFromSlice[float32](src)
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice src: %v", err)
	}
	srcGPU, err := tensor.NewWithStorage[float32]([]int{4, 4}, srcGS)
	if err != nil {
		t.Fatalf("NewWithStorage src: %v", err)
	}

	// Destination: pre-allocate a [2,8] GPU tensor full of poison (0xDEADBEEF
	// pattern as a recognisable non-zero value). The pre-fix bug left this
	// buffer untouched; the post-fix contract requires dst to reflect src.
	poison := make([]float32, 16)
	for i := range poison {
		poison[i] = -999.0
	}
	dstGS, err := tensor.NewGPUStorageFromSlice[float32](poison)
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice dst: %v", err)
	}
	dst, err := tensor.NewWithStorage[float32]([]int{2, 8}, dstGS)
	if err != nil {
		t.Fatalf("NewWithStorage dst: %v", err)
	}

	// Reshape src into dst's shape, passing dst as the output buffer. Discard
	// the return value to mirror the zerfoo call pattern that triggered #81.
	ret, err := eng.Reshape(ctx, srcGPU, []int{2, 8}, dst)
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}

	// Contract 1: ret must be the same tensor object as dst (dst-honoring).
	if ret != dst {
		t.Errorf("Reshape returned a fresh tensor instead of mutating dst; "+
			"caller-provided dst was ignored. ret=%p dst=%p", ret, dst)
	}

	// Contract 2: dst's shape must be the requested shape.
	if got := dst.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 8 {
		t.Errorf("dst.Shape() = %v, want [2 8]", got)
	}

	// Contract 3: dst's data must reflect src's data, not the poison pattern.
	dstStorage, ok := dst.GetStorage().(*tensor.GPUStorage[float32])
	if !ok {
		t.Fatalf("dst storage is not *GPUStorage[float32]: %T", dst.GetStorage())
	}
	got := dstStorage.Slice()
	if len(got) != 16 {
		t.Fatalf("dst.GetStorage().Slice() len = %d, want 16", len(got))
	}
	for i, v := range got {
		want := float32(i + 1)
		if v != want {
			t.Errorf("dst.Data()[%d] = %v, want %v "+
				"(stale pre-allocated buffer — Reshape ignored dst)", i, v, want)
		}
	}
}

// TestGPUEngine_Reshape_NoDst preserves the no-dst behavior: Reshape returns a
// fresh tensor aliasing the source view. This is the fast-path most callers use.
func TestGPUEngine_Reshape_NoDst(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	ops := numeric.Float32Ops{}
	eng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()

	src := make([]float32, 12)
	for i := range src {
		src[i] = float32(i)
	}
	srcGS, err := tensor.NewGPUStorageFromSlice[float32](src)
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice: %v", err)
	}
	srcGPU, err := tensor.NewWithStorage[float32]([]int{3, 4}, srcGS)
	if err != nil {
		t.Fatalf("NewWithStorage: %v", err)
	}

	out, err := eng.Reshape(ctx, srcGPU, []int{2, 6})
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	if got := out.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 6 {
		t.Errorf("out.Shape() = %v, want [2 6]", got)
	}
	outGS, ok := out.GetStorage().(*tensor.GPUStorage[float32])
	if !ok {
		t.Fatalf("out storage is not *GPUStorage[float32]: %T", out.GetStorage())
	}
	for i, v := range outGS.Slice() {
		if v != float32(i) {
			t.Errorf("out.Data()[%d] = %v, want %v", i, v, float32(i))
		}
	}
}

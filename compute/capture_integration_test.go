package compute

import (
	"errors"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/log"
	"github.com/zerfoo/ztensor/numeric"
)

// TestIntegration_WithCapture_AllocWeightRoutesAsync verifies the combined
// E2 flow: WithCapture sets CaptureAwareAllocator mode, and allocWeight
// routes through mallocAsyncFn (not mallocManagedFn) during capture.
func TestIntegration_WithCapture_AllocWeightRoutesAsync(t *testing.T) {
	var asyncCalled atomic.Bool
	var managedCalled atomic.Bool
	var sentinel byte

	restoreCapture := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restoreCapture()

	restoreAsync := swapMallocAsyncFn(func(size int, _ *cuda.Stream) (unsafe.Pointer, error) {
		asyncCalled.Store(true)
		return unsafe.Pointer(&sentinel), nil
	})
	defer restoreAsync()

	restoreManaged := swapMallocManagedFn(func(size int) (unsafe.Pointer, error) {
		managedCalled.Store(true)
		return unsafe.Pointer(&sentinel), nil
	})
	defer restoreManaged()

	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restoreStatus()

	pool := &fakeCapturePool{}
	e := &GPUEngine[float32]{
		stream:     fakePtrStream{},
		pool:       pool,
		managedMem: true,
		logger:     log.Nop(),
	}

	handle, err := e.WithCapture(func() error {
		ptr, allocErr := e.allocWeight(4096)
		if allocErr != nil {
			return allocErr
		}
		if ptr == nil {
			t.Error("allocWeight during capture returned nil pointer")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("WithCapture: unexpected error: %v", err)
	}
	if handle.ptr == nil {
		t.Fatal("WithCapture: expected non-nil graph handle")
	}
	if !asyncCalled.Load() {
		t.Fatal("allocWeight inside WithCapture should route through mallocAsyncFn")
	}
	if managedCalled.Load() {
		t.Fatal("allocWeight inside WithCapture should NOT use mallocManagedFn")
	}
}

// TestIntegration_WithCapture_UploadBytesRoutesAsync verifies that
// uploadBytes uses memcpyAsyncFn during a WithCapture region.
func TestIntegration_WithCapture_UploadBytesRoutesAsync(t *testing.T) {
	var asyncCopyCalled atomic.Bool
	var copiedKind cuda.MemcpyKind

	restoreCapture := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restoreCapture()

	restoreMemcpy := swapMemcpyAsyncFn(func(_ unsafe.Pointer, _ unsafe.Pointer, _ int, kind cuda.MemcpyKind, _ *cuda.Stream) error {
		asyncCopyCalled.Store(true)
		copiedKind = kind
		return nil
	})
	defer restoreMemcpy()

	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restoreStatus()

	pool := &fakeCapturePool{}
	e := &GPUEngine[float32]{
		stream:     fakePtrStream{},
		pool:       pool,
		managedMem: true,
		logger:     log.Nop(),
	}

	handle, err := e.WithCapture(func() error {
		src := []byte{0xDE, 0xAD, 0xBE, 0xEF}
		var devMem byte
		return e.uploadBytes(unsafe.Pointer(&devMem), src)
	})
	if err != nil {
		t.Fatalf("WithCapture: unexpected error: %v", err)
	}
	if handle.ptr == nil {
		t.Fatal("WithCapture: expected non-nil graph handle")
	}
	if !asyncCopyCalled.Load() {
		t.Fatal("uploadBytes inside WithCapture should route through memcpyAsyncFn")
	}
	if copiedKind != cuda.MemcpyHostToDevice {
		t.Fatalf("uploadBytes during capture: copy kind = %v, want MemcpyHostToDevice", copiedKind)
	}
}

// TestIntegration_WithCapture_GuardFiresWithoutCaptureAwareAllocator verifies
// the combined flow: WithCapture on an engine whose pool does NOT implement
// CaptureAwareAllocator returns ErrCaptureIncompatibleAllocation when
// allocWeight is called during capture.
func TestIntegration_WithCapture_GuardFiresWithoutCaptureAwareAllocator(t *testing.T) {
	restoreCapture := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restoreCapture()

	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restoreStatus()

	e := &GPUEngine[float32]{
		stream: fakePtrStream{},
		pool:   &fakeBasicPool{},
		logger: log.Nop(),
	}

	_, err := e.WithCapture(func() error {
		_, allocErr := e.allocWeight(4096)
		return allocErr
	})
	if err == nil {
		t.Fatal("WithCapture with non-capture-aware pool: expected error, got nil")
	}
	if !errors.Is(err, ErrCaptureIncompatibleAllocation) {
		t.Fatalf("expected ErrCaptureIncompatibleAllocation, got %v", err)
	}
}

// TestIntegration_EndCapture_ClearsCaptureAwareMode verifies that after
// WithCapture completes, subsequent allocWeight calls use the normal
// (non-async) path. The CaptureAwareAllocator's capturing state must be
// false after EndCapture.
func TestIntegration_EndCapture_ClearsCaptureAwareMode(t *testing.T) {
	var sentinel byte

	restoreCapture := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restoreCapture()

	// During capture, status is Active; after, it's None.
	captureActive := true
	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		if captureActive {
			return cuda.CaptureStatusActive, nil
		}
		return cuda.CaptureStatusNone, nil
	})
	defer restoreStatus()

	var asyncCalled atomic.Bool
	restoreAsync := swapMallocAsyncFn(func(_ int, _ *cuda.Stream) (unsafe.Pointer, error) {
		asyncCalled.Store(true)
		return unsafe.Pointer(&sentinel), nil
	})
	defer restoreAsync()

	var managedCalled atomic.Bool
	restoreManaged := swapMallocManagedFn(func(_ int) (unsafe.Pointer, error) {
		managedCalled.Store(true)
		return unsafe.Pointer(&sentinel), nil
	})
	defer restoreManaged()

	pool := &fakeCapturePool{}
	e := &GPUEngine[float32]{
		stream:     fakePtrStream{},
		pool:       pool,
		managedMem: true,
		logger:     log.Nop(),
	}

	// Run WithCapture — fn is a no-op.
	_, err := e.WithCapture(func() error { return nil })
	if err != nil {
		t.Fatalf("WithCapture: unexpected error: %v", err)
	}

	// After WithCapture, the pool should no longer be in capturing mode.
	if pool.IsCapturing() {
		t.Fatal("pool should NOT be capturing after WithCapture completes")
	}

	// Switch capture status to None for subsequent alloc.
	captureActive = false
	asyncCalled.Store(false)
	managedCalled.Store(false)

	// allocWeight should now use the normal managed path, not async.
	ptr, err := e.allocWeight(4096)
	if err != nil {
		t.Fatalf("allocWeight after WithCapture: unexpected error: %v", err)
	}
	if ptr == nil {
		t.Fatal("allocWeight after WithCapture: expected non-nil pointer")
	}
	if asyncCalled.Load() {
		t.Fatal("allocWeight after WithCapture should NOT use mallocAsyncFn")
	}
	if !managedCalled.Load() {
		t.Fatal("allocWeight after WithCapture should use mallocManagedFn (managedMem=true)")
	}
}

// TestIntegration_CaptureAllocCount_ResetOnEndCapture verifies the end-to-end
// flow: allocWeight during capture without CaptureAwareAllocator increments
// captureAllocCount, and EndCapture (via WithCapture) resets it to zero.
func TestIntegration_CaptureAllocCount_ResetOnEndCapture(t *testing.T) {
	restoreCapture := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restoreCapture()

	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restoreStatus()

	e := &GPUEngine[float32]{
		stream: fakePtrStream{},
		pool:   &fakeBasicPool{},
		logger: log.Nop(),
	}

	// WithCapture's fn triggers two allocWeight attempts — both fail with
	// ErrCaptureIncompatibleAllocation, but the counter still increments.
	_, _ = e.WithCapture(func() error {
		_, _ = e.allocWeight(4096)
		_, _ = e.allocWeight(8192)
		// Return nil so EndCapture runs the full path.
		return nil
	})

	// After WithCapture (which calls EndCapture), the counter should be 0.
	if got := e.CaptureAllocCount(); got != 0 {
		t.Fatalf("CaptureAllocCount after WithCapture: got %d, want 0", got)
	}
}

// TestIntegration_WorkspacePrealloc_Idempotent verifies that calling
// preAllocateWorkspaces twice does not double-allocate. This is a
// complementary integration check alongside the unit test.
func TestIntegration_WorkspacePrealloc_Idempotent(t *testing.T) {
	pool := newFakeMemPool()
	eng := &GPUEngine[float32]{
		cpu:           NewCPUEngine[float32](numeric.Float32Ops{}),
		runtime:       fakeRuntime{},
		pool:          pool,
		stream:        fakeStream{},
		logger:        log.Nop(),
		deviceID:      0,
		dtype:         DTypeF32,
		maxAllocBytes: DefaultMaxAllocBytes,
	}

	eng.preAllocateWorkspaces()
	allocsFirst := pool.allocCount

	eng.preAllocateWorkspaces()
	allocsSecond := pool.allocCount

	if allocsSecond != allocsFirst {
		t.Fatalf("second preAllocateWorkspaces added %d allocs, want 0",
			allocsSecond-allocsFirst)
	}
	if eng.fp8Scratch == nil {
		t.Fatal("fp8Scratch should be non-nil after preAllocateWorkspaces")
	}
}

// TestIntegration_BeginEndCapture_PoolLifecycle verifies that BeginCapture
// activates the CaptureAwareAllocator and EndCapture deactivates it, even
// when EndCapture encounters an error.
func TestIntegration_BeginEndCapture_PoolLifecycle(t *testing.T) {
	tests := []struct {
		name    string
		endErr  error
		wantErr bool
	}{
		{name: "end succeeds", endErr: nil, wantErr: false},
		{name: "end fails", endErr: errors.New("graph capture failed"), wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			restoreCapture := stubCapturePipeline(
				happyBegin,
				func(_ *cuda.Stream) (*cuda.Graph, error) {
					if tt.endErr != nil {
						return nil, tt.endErr
					}
					return &cuda.Graph{}, nil
				},
				happyInstantiate,
				happyDestroy,
			)
			defer restoreCapture()

			pool := &fakeCapturePool{}
			e := &GPUEngine[float32]{
				stream: fakePtrStream{},
				pool:   pool,
				logger: log.Nop(),
			}

			if err := e.BeginCapture(); err != nil {
				t.Fatalf("BeginCapture: unexpected error: %v", err)
			}
			if !pool.IsCapturing() {
				t.Fatal("pool should be capturing after BeginCapture")
			}

			_, err := e.EndCapture()
			if (err != nil) != tt.wantErr {
				t.Fatalf("EndCapture: error = %v, wantErr = %v", err, tt.wantErr)
			}
			if pool.IsCapturing() {
				t.Fatal("pool should NOT be capturing after EndCapture")
			}
		})
	}
}

// TestIntegration_BeginCapture_RollbackOnFailure verifies that if
// BeginCapture fails, the CaptureAwareAllocator is rolled back to
// non-capturing state.
func TestIntegration_BeginCapture_RollbackOnFailure(t *testing.T) {
	beginErr := errors.New("stream begin capture failed")
	restoreCapture := stubCapturePipeline(
		func(_ *cuda.Stream) error { return beginErr },
		happyEnd,
		happyInstantiate,
		happyDestroy,
	)
	defer restoreCapture()

	pool := &fakeCapturePool{}
	e := &GPUEngine[float32]{
		stream: fakePtrStream{},
		pool:   pool,
		logger: log.Nop(),
	}

	err := e.BeginCapture()
	if err == nil {
		t.Fatal("BeginCapture: expected error, got nil")
	}
	if !errors.Is(err, beginErr) {
		t.Fatalf("BeginCapture: expected wrapped begin error, got %v", err)
	}
	if pool.IsCapturing() {
		t.Fatal("pool should NOT be capturing after failed BeginCapture")
	}
}

// TestIntegration_WithCapture_AllocAndUploadCombined verifies that both
// allocWeight and uploadBytes work correctly inside a single WithCapture
// call — the typical real-world pattern for weight upload during capture.
func TestIntegration_WithCapture_AllocAndUploadCombined(t *testing.T) {
	var allocCalls, copyCalls atomic.Int32
	var sentinel byte

	restoreCapture := stubCapturePipeline(happyBegin, happyEnd, happyInstantiate, happyDestroy)
	defer restoreCapture()

	restoreAsync := swapMallocAsyncFn(func(_ int, _ *cuda.Stream) (unsafe.Pointer, error) {
		allocCalls.Add(1)
		return unsafe.Pointer(&sentinel), nil
	})
	defer restoreAsync()

	restoreMemcpy := swapMemcpyAsyncFn(func(_ unsafe.Pointer, _ unsafe.Pointer, _ int, _ cuda.MemcpyKind, _ *cuda.Stream) error {
		copyCalls.Add(1)
		return nil
	})
	defer restoreMemcpy()

	restoreStatus := swapCaptureStatusFn(func(_ *cuda.Stream) (cuda.CaptureStatus, error) {
		return cuda.CaptureStatusActive, nil
	})
	defer restoreStatus()

	pool := &fakeCapturePool{}
	e := &GPUEngine[float32]{
		stream: fakePtrStream{},
		pool:   pool,
		logger: log.Nop(),
	}

	handle, err := e.WithCapture(func() error {
		// Allocate then upload — mimics UploadWeights flow.
		ptr, allocErr := e.allocWeight(1024)
		if allocErr != nil {
			return allocErr
		}
		return e.uploadBytes(ptr, []byte{0x01, 0x02, 0x03, 0x04})
	})
	if err != nil {
		t.Fatalf("WithCapture: unexpected error: %v", err)
	}
	if handle.ptr == nil {
		t.Fatal("WithCapture: expected non-nil graph handle")
	}
	if got := allocCalls.Load(); got != 1 {
		t.Fatalf("expected 1 async alloc call, got %d", got)
	}
	if got := copyCalls.Load(); got != 1 {
		t.Fatalf("expected 1 async copy call, got %d", got)
	}
}

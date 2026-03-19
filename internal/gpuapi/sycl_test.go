package gpuapi_test

import (
	"runtime"
	"testing"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/internal/gpuapi"
	"github.com/zerfoo/ztensor/internal/sycl"
)

func skipUnlessSYCL(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("SYCL runtime is only available on linux")
	}
	if !sycl.Available() {
		t.Skip("SYCL runtime not available")
	}
}

func TestSYCL_DeviceDiscovery(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("SYCL runtime is only available on linux")
	}

	// On non-SYCL systems, NewSYCLRuntime returns nil gracefully.
	rt := gpuapi.NewSYCLRuntime()
	if rt == nil {
		t.Log("SYCL not available on this system, verifying graceful fallback")
		count, err := sycl.GetDeviceCount()
		if err == nil {
			t.Fatalf("expected error from GetDeviceCount when SYCL unavailable, got count=%d", count)
		}
		return
	}

	if got := rt.DeviceType(); got != device.SYCL {
		t.Fatalf("DeviceType() = %v, want %v", got, device.SYCL)
	}

	count, err := rt.GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount() error: %v", err)
	}
	if count < 1 {
		t.Fatalf("GetDeviceCount() = %d, want >= 1 on a machine with SYCL", count)
	}
	t.Logf("SYCL devices found: %d", count)
}

func TestSYCL_MemoryAlloc(t *testing.T) {
	skipUnlessSYCL(t)

	rt := gpuapi.NewSYCLRuntime()
	if rt == nil {
		t.Fatal("NewSYCLRuntime returned nil")
	}

	if err := rt.SetDevice(0); err != nil {
		t.Fatalf("SetDevice(0) error: %v", err)
	}

	// Test Malloc + Free don't panic.
	ptr, err := rt.Malloc(1024)
	if err != nil {
		t.Fatalf("Malloc(1024) error: %v", err)
	}
	if ptr == nil {
		t.Fatal("Malloc(1024) returned nil pointer")
	}
	if err := rt.Free(ptr); err != nil {
		t.Fatalf("Free() error: %v", err)
	}

	// Test CreateStream.
	stream, err := rt.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream() error: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize() error: %v", err)
	}
	if err := stream.Destroy(); err != nil {
		t.Fatalf("Destroy() error: %v", err)
	}
}

func TestSYCL_BasicOps(t *testing.T) {
	skipUnlessSYCL(t)

	// Verify BLAS and Kernels types can be instantiated.
	blas := gpuapi.NewSYCLBlas()
	if blas == nil {
		t.Fatal("NewSYCLBlas returned nil")
	}

	kernels := gpuapi.NewSYCLKernels()
	if kernels == nil {
		t.Fatal("NewSYCLKernels returned nil")
	}

	// Verify BLAS operations return errors (not panics) when called.
	if err := blas.Sgemm(2, 2, 2, 1.0, nil, nil, 0.0, nil); err == nil {
		t.Fatal("expected error from Sgemm on stub SYCL BLAS")
	}

	// Verify kernel operations return errors (not panics) when called.
	if err := kernels.Add(nil, nil, nil, 0, nil); err == nil {
		t.Fatal("expected error from Add on stub SYCL kernels")
	}

	if err := blas.Destroy(); err != nil {
		t.Fatalf("BLAS Destroy() error: %v", err)
	}
}

func TestSYCL_RuntimeInterface(t *testing.T) {
	// Compile-time assertions are in the source files.
	// This test verifies the SYCL types satisfy their interfaces.
	var _ gpuapi.Runtime = (*gpuapi.SYCLRuntime)(nil)
	var _ gpuapi.BLAS = (*gpuapi.SYCLBlas)(nil)
	var _ gpuapi.KernelRunner = (*gpuapi.SYCLKernels)(nil)
	var _ gpuapi.MemPool = (*gpuapi.SYCLMemPool)(nil)
	var _ gpuapi.DNN = (*gpuapi.SYCLDnn)(nil)
	t.Log("all SYCL types satisfy their GRAL interfaces")
}

func TestSYCL_MemPool(t *testing.T) {
	skipUnlessSYCL(t)

	rt := gpuapi.NewSYCLRuntime()
	if rt == nil {
		t.Fatal("NewSYCLRuntime returned nil")
	}

	if err := rt.SetDevice(0); err != nil {
		t.Fatalf("SetDevice(0) error: %v", err)
	}

	pool := gpuapi.NewSYCLMemPool(rt)

	// Alloc + Free + re-Alloc should reuse.
	ptr1, err := pool.Alloc(0, 4096)
	if err != nil {
		t.Fatalf("Alloc error: %v", err)
	}
	pool.Free(0, ptr1, 4096)

	allocs, totalBytes := pool.Stats()
	if allocs != 1 || totalBytes != 4096 {
		t.Fatalf("Stats after Free: allocs=%d totalBytes=%d, want 1/4096", allocs, totalBytes)
	}

	ptr2, err := pool.Alloc(0, 4096)
	if err != nil {
		t.Fatalf("re-Alloc error: %v", err)
	}
	if ptr2 != ptr1 {
		t.Log("re-Alloc returned different pointer (acceptable but expected same)")
	}

	pool.Free(0, ptr2, 4096)
	if err := pool.Drain(); err != nil {
		t.Fatalf("Drain error: %v", err)
	}

	allocs, totalBytes = pool.Stats()
	if allocs != 0 || totalBytes != 0 {
		t.Fatalf("Stats after Drain: allocs=%d totalBytes=%d, want 0/0", allocs, totalBytes)
	}
}

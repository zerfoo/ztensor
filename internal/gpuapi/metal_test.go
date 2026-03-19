package gpuapi_test

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/internal/gpuapi"
	"github.com/zerfoo/ztensor/internal/metal"
)

func TestMetal_DeviceDetection(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal is only available on darwin")
	}
	if !metal.Available() {
		t.Skip("Metal framework not available")
	}

	rt := gpuapi.NewMetalRuntime()
	if rt == nil {
		t.Fatal("NewMetalRuntime returned nil on darwin with Metal available")
	}

	if got := rt.DeviceType(); got != device.Metal {
		t.Fatalf("DeviceType() = %v, want %v", got, device.Metal)
	}

	count, err := rt.GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount() error: %v", err)
	}
	if count < 1 {
		t.Fatalf("GetDeviceCount() = %d, want >= 1 on a Mac with Metal", count)
	}
	t.Logf("Metal devices found: %d", count)
}

func TestMetal_BasicOps(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal is only available on darwin")
	}
	if !metal.Available() {
		t.Skip("Metal framework not available")
	}

	rt := gpuapi.NewMetalRuntime()
	if rt == nil {
		t.Fatal("NewMetalRuntime returned nil")
	}

	if err := rt.SetDevice(0); err != nil {
		t.Fatalf("SetDevice(0) error: %v", err)
	}

	// Test Malloc + Free.
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

func TestMetal_MemcpyRoundTrip(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal is only available on darwin")
	}
	if !metal.Available() {
		t.Skip("Metal framework not available")
	}

	rt := gpuapi.NewMetalRuntime()
	if rt == nil {
		t.Fatal("NewMetalRuntime returned nil")
	}

	if err := rt.SetDevice(0); err != nil {
		t.Fatalf("SetDevice(0) error: %v", err)
	}

	data := []byte{0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE}
	n := len(data)

	// Allocate device buffer.
	devBuf, err := rt.Malloc(n)
	if err != nil {
		t.Fatalf("Malloc(%d) error: %v", n, err)
	}
	defer rt.Free(devBuf)

	// Host -> Device.
	hostSrc := unsafe.Pointer(unsafe.SliceData(data))
	if err := rt.Memcpy(devBuf, hostSrc, n, gpuapi.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D error: %v", err)
	}

	// Device -> Host.
	result := make([]byte, n)
	hostDst := unsafe.Pointer(unsafe.SliceData(result))
	if err := rt.Memcpy(hostDst, devBuf, n, gpuapi.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H error: %v", err)
	}

	for i := range data {
		if data[i] != result[i] {
			t.Fatalf("round-trip mismatch at index %d: got %#x, want %#x", i, result[i], data[i])
		}
	}
}

func TestMetal_MemPool(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal is only available on darwin")
	}
	if !metal.Available() {
		t.Skip("Metal framework not available")
	}

	rt := gpuapi.NewMetalRuntime()
	if rt == nil {
		t.Fatal("NewMetalRuntime returned nil")
	}

	if err := rt.SetDevice(0); err != nil {
		t.Fatalf("SetDevice(0) error: %v", err)
	}

	pool := gpuapi.NewMetalMemPool(rt)

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

func TestMetal_InterfaceAssertions(t *testing.T) {
	// Compile-time assertions are in the source files.
	// This test verifies the Metal types satisfy their interfaces.
	var _ gpuapi.Runtime = (*gpuapi.MetalRuntime)(nil)
	var _ gpuapi.BLAS = (*gpuapi.MetalBlas)(nil)
	var _ gpuapi.KernelRunner = (*gpuapi.MetalKernels)(nil)
	var _ gpuapi.MemPool = (*gpuapi.MetalMemPool)(nil)
	var _ gpuapi.DNN = (*gpuapi.MetalDNN)(nil)
	t.Log("all Metal types satisfy their GRAL interfaces")
}

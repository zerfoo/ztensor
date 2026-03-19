package gpuapi_test

import (
	"runtime"
	"testing"
	"time"
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/internal/fpga"
	"github.com/zerfoo/ztensor/internal/gpuapi"
)

func skipUnlessFPGA(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("FPGA runtime is only available on linux")
	}
	if !fpga.Available() {
		t.Skip("FPGA runtime not available")
	}
}

func TestFPGA_DeviceDetection(t *testing.T) {
	skipUnlessFPGA(t)

	rt := gpuapi.NewFPGARuntime()
	if rt == nil {
		t.Fatal("NewFPGARuntime returned nil on linux with FPGA available")
	}

	if got := rt.DeviceType(); got != device.FPGA {
		t.Fatalf("DeviceType() = %v, want %v", got, device.FPGA)
	}

	count, err := rt.GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount() error: %v", err)
	}
	if count < 1 {
		t.Fatalf("GetDeviceCount() = %d, want >= 1 on a machine with FPGA", count)
	}
	t.Logf("FPGA devices found: %d", count)
}

func TestFPGA_BasicOps(t *testing.T) {
	skipUnlessFPGA(t)

	rt := gpuapi.NewFPGARuntime()
	if rt == nil {
		t.Fatal("NewFPGARuntime returned nil")
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

func TestFPGA_MemcpyRoundTrip(t *testing.T) {
	skipUnlessFPGA(t)

	rt := gpuapi.NewFPGARuntime()
	if rt == nil {
		t.Fatal("NewFPGARuntime returned nil")
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

func TestFPGA_MemPool(t *testing.T) {
	skipUnlessFPGA(t)

	rt := gpuapi.NewFPGARuntime()
	if rt == nil {
		t.Fatal("NewFPGARuntime returned nil")
	}

	if err := rt.SetDevice(0); err != nil {
		t.Fatalf("SetDevice(0) error: %v", err)
	}

	pool := gpuapi.NewFPGAMemPool(rt)

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

func TestFPGA_Latency(t *testing.T) {
	skipUnlessFPGA(t)

	rt := gpuapi.NewFPGARuntime()
	if rt == nil {
		t.Fatal("NewFPGARuntime returned nil")
	}

	if err := rt.SetDevice(0); err != nil {
		t.Fatalf("SetDevice(0) error: %v", err)
	}

	// Measure allocation latency.
	const iterations = 100
	const bufSize = 4096

	start := time.Now()
	for i := 0; i < iterations; i++ {
		ptr, err := rt.Malloc(bufSize)
		if err != nil {
			t.Fatalf("Malloc iteration %d error: %v", i, err)
		}
		if err := rt.Free(ptr); err != nil {
			t.Fatalf("Free iteration %d error: %v", i, err)
		}
	}
	allocDur := time.Since(start)
	t.Logf("FPGA alloc+free latency: %v avg over %d iterations", allocDur/time.Duration(iterations), iterations)

	// Measure memcpy latency (H2D + D2H round-trip).
	data := make([]byte, bufSize)
	for i := range data {
		data[i] = byte(i)
	}
	result := make([]byte, bufSize)

	devBuf, err := rt.Malloc(bufSize)
	if err != nil {
		t.Fatalf("Malloc error: %v", err)
	}
	defer rt.Free(devBuf)

	start = time.Now()
	for i := 0; i < iterations; i++ {
		hostSrc := unsafe.Pointer(unsafe.SliceData(data))
		if err := rt.Memcpy(devBuf, hostSrc, bufSize, gpuapi.MemcpyHostToDevice); err != nil {
			t.Fatalf("Memcpy H2D iteration %d error: %v", i, err)
		}
		hostDst := unsafe.Pointer(unsafe.SliceData(result))
		if err := rt.Memcpy(hostDst, devBuf, bufSize, gpuapi.MemcpyDeviceToHost); err != nil {
			t.Fatalf("Memcpy D2H iteration %d error: %v", i, err)
		}
	}
	memcpyDur := time.Since(start)
	t.Logf("FPGA memcpy round-trip latency: %v avg over %d iterations (%d bytes)", memcpyDur/time.Duration(iterations), iterations, bufSize)

	// Measure stream creation latency.
	start = time.Now()
	for i := 0; i < iterations; i++ {
		stream, err := rt.CreateStream()
		if err != nil {
			t.Fatalf("CreateStream iteration %d error: %v", i, err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("Synchronize iteration %d error: %v", i, err)
		}
		if err := stream.Destroy(); err != nil {
			t.Fatalf("Destroy iteration %d error: %v", i, err)
		}
	}
	streamDur := time.Since(start)
	t.Logf("FPGA stream create+sync+destroy latency: %v avg over %d iterations", streamDur/time.Duration(iterations), iterations)
}

func TestFPGA_InterfaceAssertions(t *testing.T) {
	// Compile-time assertions are in the source files.
	// This test verifies the FPGA types satisfy their interfaces.
	var _ gpuapi.Runtime = (*gpuapi.FPGARuntime)(nil)
	var _ gpuapi.BLAS = (*gpuapi.FPGABlas)(nil)
	var _ gpuapi.KernelRunner = (*gpuapi.FPGAKernels)(nil)
	var _ gpuapi.MemPool = (*gpuapi.FPGAMemPool)(nil)
	var _ gpuapi.DNN = (*gpuapi.FPGADnn)(nil)
	t.Log("all FPGA types satisfy their GRAL interfaces")
}

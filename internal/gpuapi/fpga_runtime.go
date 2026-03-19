package gpuapi

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/internal/fpga"
)

// FPGARuntime implements the Runtime interface using an FPGA accelerator
// via OpenCL (Intel FPGA / Xilinx XRT).
type FPGARuntime struct {
	ctx *fpga.Context
}

// NewFPGARuntime returns a new FPGA runtime adapter.
// Returns nil if no FPGA runtime is available on this system.
func NewFPGARuntime() *FPGARuntime {
	if !fpga.Available() {
		return nil
	}
	return &FPGARuntime{}
}

func (r *FPGARuntime) DeviceType() device.Type { return device.FPGA }

func (r *FPGARuntime) SetDevice(deviceID int) error {
	if r.ctx != nil {
		_ = r.ctx.Destroy()
	}
	ctx, err := fpga.NewContext(deviceID)
	if err != nil {
		return err
	}
	r.ctx = ctx
	return nil
}

func (r *FPGARuntime) GetDeviceCount() (int, error) {
	if !fpga.Available() {
		return 0, fmt.Errorf("fpga: not available")
	}
	return fpga.GetDeviceCount()
}

func (r *FPGARuntime) Malloc(byteSize int) (unsafe.Pointer, error) {
	return r.ctx.Malloc(byteSize)
}

func (r *FPGARuntime) Free(ptr unsafe.Pointer) error {
	return r.ctx.Free(ptr)
}

func (r *FPGARuntime) Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	return r.ctx.Memcpy(dst, src, count, fpgaMemcpyKind(kind))
}

func (r *FPGARuntime) MemcpyAsync(dst, src unsafe.Pointer, count int, kind MemcpyKind, _ Stream) error {
	// FPGA OpenCL queues are in-order; fall back to synchronous copy.
	return r.Memcpy(dst, src, count, kind)
}

func (r *FPGARuntime) MemcpyPeer(dst unsafe.Pointer, _ int, src unsafe.Pointer, _ int, count int) error {
	// FPGA does not support direct peer-to-peer transfer.
	// Fall back to D2H + H2D via a host buffer.
	buf := make([]byte, count)
	hostPtr := unsafe.Pointer(unsafe.SliceData(buf))
	if err := r.ctx.Memcpy(hostPtr, src, count, fpga.MemcpyDeviceToHost); err != nil {
		return err
	}
	return r.ctx.Memcpy(dst, hostPtr, count, fpga.MemcpyHostToDevice)
}

func (r *FPGARuntime) CreateStream() (Stream, error) {
	s, err := r.ctx.CreateStream()
	if err != nil {
		return nil, err
	}
	return &fpgaStreamWrapper{stream: s}, nil
}

// fpgaStreamWrapper adapts fpga.Stream to the GRAL Stream interface.
type fpgaStreamWrapper struct {
	stream *fpga.Stream
}

func (w *fpgaStreamWrapper) Synchronize() error { return w.stream.Synchronize() }
func (w *fpgaStreamWrapper) Destroy() error      { return w.stream.Destroy() }
func (w *fpgaStreamWrapper) Ptr() unsafe.Pointer  { return w.stream.Ptr() }

// fpgaMemcpyKind converts a GRAL MemcpyKind to an FPGA MemcpyKind.
func fpgaMemcpyKind(kind MemcpyKind) fpga.MemcpyKind {
	switch kind {
	case MemcpyHostToDevice:
		return fpga.MemcpyHostToDevice
	case MemcpyDeviceToHost:
		return fpga.MemcpyDeviceToHost
	case MemcpyDeviceToDevice:
		return fpga.MemcpyDeviceToDevice
	default:
		return fpga.MemcpyHostToDevice
	}
}

// Compile-time interface assertion.
var _ Runtime = (*FPGARuntime)(nil)

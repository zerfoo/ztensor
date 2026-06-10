package gpuapi

import (
	"math"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/internal/cuda/kernels"
)

// registerArenaPoisonKernelFill upgrades the arena's poison fill (ADR 006
// decision 4, ZTENSOR_ARENA_POISON=1) from the default host-staged Memcpy to
// the on-device fill kernel. Called from NewCUDAArenaPool, i.e. only on real
// CUDA engines where the kernel library is expected to be loadable; if the
// kernel library is missing, kernels.Fill returns an error at fill time and
// the arena logs a warning per skipped fill.
//
// internal/cuda cannot call kernels.Fill directly (the kernels package
// imports internal/cuda), hence this registration indirection.
func registerArenaPoisonKernelFill() {
	if !cuda.ArenaPoisonEnabled() {
		return // keep the default; avoids touching global state when off
	}
	cuda.SetArenaPoisonFill(arenaPoisonKernelFill)
}

// arenaPoisonKernelFill fills the region with cuda.ArenaPoisonWord via the
// elementwise fill kernel. The value is passed bit-exact: Float32frombits ->
// floatBits inside kernels.Fill is a pure bits round-trip and the kernel
// stores (never computes on) the value, so the NaN payload survives.
//
// The launch goes to the legacy default stream (nil), which synchronizes
// with the engine's blocking streams: any later kernel writing the reused
// region is ordered after the poison fill. byteLen is a multiple of the
// arena's 256-byte quantum, so byteLen/4 covers the region exactly.
func arenaPoisonKernelFill(ptr unsafe.Pointer, byteLen int) error {
	n := byteLen / 4
	if n == 0 {
		return nil
	}
	return kernels.Fill(ptr, math.Float32frombits(cuda.ArenaPoisonWord), n, nil)
}

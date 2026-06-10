package cuda

import (
	"fmt"
	"os"
	"unsafe"
)

// Poison-on-reset debug mode (ADR 006 decision 4; zerfoo
// docs/plan-gpu-training-hardening.md T1.4, issue #128).
//
// When ZTENSOR_ARENA_POISON=1, every arena region that becomes reusable --
// the [resetFloor, offset) span on Reset, and freed blocks entering the
// free-list via FreeArena -- is filled with a NaN bit pattern BEFORE it can
// be handed out again. A node that cached a forward intermediate in a struct
// field and reads it after the arena reclaimed the buffer (the zerfoo#842
// LayerNorm variance / zerfoo#845 gradient buffer / Wolf QK-norm bug class)
// then sees a deterministic NaN at the corruption site, instead of a delayed,
// non-deterministic training NaN days later.
//
// Off by default. The flag is read once at process init, mirroring the other
// env toggles (ZERFOO_DEBUG_GPU, ZERFOO_ENABLE_MANAGED_MEM); when unset the
// only cost is one branch on Reset/FreeArena and none on Alloc.
var arenaPoisonEnabled = os.Getenv("ZTENSOR_ARENA_POISON") == "1"

// ArenaPoisonEnabled reports whether ZTENSOR_ARENA_POISON=1 was set at
// process start. Engines use it to log that the (slow) debug mode is active.
func ArenaPoisonEnabled() bool { return arenaPoisonEnabled }

// ArenaPoisonWord is the 32-bit poison pattern repeated across reclaimed
// regions. The fill itself is dtype-agnostic bytes (little-endian
// 00 00 F8 7F, repeated); what a stale read sees depends on how the caller
// types the memory:
//
//   - f32 reads decode to a quiet NaN (exponent all-ones, quiet bit set,
//     mantissa 0x780000 -- distinguishable from the canonical 0x7FC00000
//     qNaN in a memory dump).
//   - f64 reads at 8-byte alignment see 0x7FF8_0000_7FF8_0000, ALSO a quiet
//     NaN. This is why 0x7FF80000 is used instead of the canonical f32 qNaN
//     0x7FC00000 named in the plan: 0x7FC00000 repeated decodes to a large
//     finite f64 (~2.2e307), not a NaN, while 0x7FF80000 repeated is the
//     high word of the canonical f64 qNaN and poisons both float widths.
//   - i32/u32 reads see 0x7FF80000 (2146959360), a recognizable sentinel.
//
// CUDA devices and all supported hosts (amd64, arm64) are little-endian, so
// the byte pattern decodes identically on both sides of a managed mapping.
const ArenaPoisonWord uint32 = 0x7FF80000

// ArenaPoisonFillFunc fills byteLen bytes at the device pointer ptr with the
// ArenaPoisonWord pattern. byteLen is always a positive multiple of the
// arena's 256-byte alignment quantum (so byteLen/4 whole words).
type ArenaPoisonFillFunc func(ptr unsafe.Pointer, byteLen int) error

// arenaPoisonFillFn is the device fill used by poisonRegion. The default
// stages the pattern in host memory and issues synchronous H2D Memcpys in
// 4 MiB chunks -- correct wherever the CUDA runtime is loaded, but slow
// (acceptable for a debug mode; documented in docs/design.md).
// internal/gpuapi upgrades it to the on-device fill kernel at engine arena
// construction via SetArenaPoisonFill. Tests swap in host-memory fills.
var arenaPoisonFillFn ArenaPoisonFillFunc = arenaPoisonFillHostStaged

// SetArenaPoisonFill overrides the poison fill implementation. Passing nil
// restores the default host-staged Memcpy fill.
func SetArenaPoisonFill(fn ArenaPoisonFillFunc) {
	if fn == nil {
		fn = arenaPoisonFillHostStaged
	}
	arenaPoisonFillFn = fn
}

// arenaPoisonWarnFn sinks poison-mode warnings (skipped fills during graph
// capture, fill failures). Default writes a single stderr line, like
// defaultArenaOverflowLog; tests swap it to capture.
var arenaPoisonWarnFn = func(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "ztensor arena poison: "+format+"\n", args...)
}

// arenaPoisonStageBytes is the host staging-buffer size for the fallback
// fill: large enough to amortize Memcpy launch overhead, small enough not to
// matter on the heap.
const arenaPoisonStageBytes = 4 << 20

// arenaPoisonFillHostStaged is the fallback fill: build the byte pattern in a
// host buffer and copy it to the device in chunks. Each Memcpy is synchronous
// (cudaMemcpy), so the poison is visible to all streams before the region is
// reused. Cost: ~one H2D copy per 4 MiB of reclaimed arena per Reset -- slow,
// but only reachable in the debug mode and only when the fill kernel has not
// been registered.
func arenaPoisonFillHostStaged(ptr unsafe.Pointer, byteLen int) error {
	stage := make([]byte, min(byteLen, arenaPoisonStageBytes))
	fillHostPoison(stage)
	for off := 0; off < byteLen; off += len(stage) {
		n := min(byteLen-off, len(stage))
		if err := Memcpy(unsafe.Add(ptr, off), unsafe.Pointer(&stage[0]), n, MemcpyHostToDevice); err != nil {
			return fmt.Errorf("arena poison fill (host-staged, %d bytes at +%d): %w", n, off, err)
		}
	}
	return nil
}

// fillHostPoison writes the ArenaPoisonWord byte pattern (little-endian)
// into a host buffer. Shared by the host-staged device fill and by tests
// that poison host-backed arenas directly.
func fillHostPoison(b []byte) {
	w := ArenaPoisonWord
	pat := [4]byte{byte(w), byte(w >> 8), byte(w >> 16), byte(w >> 24)}
	for i := range b {
		b[i] = pat[i&3]
	}
}

// poisonRegion fills [ptr, ptr+byteLen) with the poison pattern. Only called
// when arenaPoisonEnabled, at the moment a region becomes reusable, either
// with a.mu held (Reset) or while the caller still exclusively owns the
// region (FreeArena) -- so the fill cannot race a new owner's writes. The
// kernel fill runs on the legacy default stream and the fallback Memcpy is
// synchronous; both order before subsequent work on the engine's blocking
// streams, so a legitimate new owner's writes land after the poison.
//
// Capture interaction (ADR 004/005): issuing fills or synchronous copies
// while a CUDA graph capture is active would either be recorded into the
// graph or hang the GB10 driver, so the fill is skipped with a warning.
// Regions recycled mid-capture are NOT poisoned.
func (a *ArenaPool) poisonRegion(ptr unsafe.Pointer, byteLen int, site string) {
	if byteLen <= 0 {
		return
	}
	if CaptureActive() {
		arenaPoisonWarnFn("%s: skipping poison fill of %d bytes: CUDA graph capture active", site, byteLen)
		return
	}
	if err := arenaPoisonFillFn(ptr, byteLen); err != nil {
		arenaPoisonWarnFn("%s: poison fill of %d bytes failed: %v", site, byteLen, err)
	}
}

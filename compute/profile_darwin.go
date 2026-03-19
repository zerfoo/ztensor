//go:build darwin

package compute

import (
	"encoding/binary"
	"strings"
	"syscall"
)

// profileOS populates macOS-specific hardware information using sysctl.
func profileOS(p *HardwareProfile) error {
	if model, err := syscall.Sysctl("machdep.cpu.brand_string"); err == nil {
		p.CPUModel = strings.TrimSpace(model)
	}

	if v, err := sysctlUint64("hw.l1dcachesize"); err == nil {
		p.CacheL1 = int64(v)
	}
	if v, err := sysctlUint64("hw.l2cachesize"); err == nil {
		p.CacheL2 = int64(v)
	}
	if v, err := sysctlUint64("hw.l3cachesize"); err == nil {
		p.CacheL3 = int64(v)
	}
	if v, err := sysctlUint64("hw.memsize"); err == nil {
		p.TotalRAM = int64(v)
	}

	detectMetalGPU(p)
	return nil
}

// detectMetalGPU sets GPU fields for Metal on macOS.
func detectMetalGPU(p *HardwareProfile) {
	if p.HasNEON {
		// Apple Silicon: unified memory GPU always available.
		p.GPUAvailable = true
		p.GPUBackend = "metal"
		p.GPUName = p.CPUModel
		p.GPUMemory = p.TotalRAM
		p.GPUCount = 1
		return
	}

	// Intel Mac: best-effort detection.
	if v, err := syscall.SysctlUint32("hw.optional.metal"); err == nil && v > 0 {
		p.GPUAvailable = true
		p.GPUBackend = "metal"
		p.GPUName = "GPU (Metal)"
		p.GPUCount = 1
	}
}

// sysctlUint64 reads a sysctl numeric value as uint64.
// syscall.Sysctl returns the raw value bytes as a Go string, stripping
// trailing null bytes. For 8-byte values this means we get 7 bytes
// (or fewer if high bytes are zero), so we right-pad with zeros.
func sysctlUint64(name string) (uint64, error) {
	raw, err := syscall.Sysctl(name)
	if err != nil {
		return 0, err
	}
	b := []byte(raw)
	if len(b) == 0 {
		return 0, syscall.EINVAL
	}
	// Pad to 8 bytes (Sysctl strips trailing 0x00 bytes).
	for len(b) < 8 {
		b = append(b, 0)
	}
	return binary.LittleEndian.Uint64(b[:8]), nil
}

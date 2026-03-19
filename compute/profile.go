package compute

import (
	"fmt"
	"runtime"
	"strings"
)

// HardwareProfile describes the CPU and GPU capabilities of the current system.
// It is used by the auto-optimization framework to select the best engine,
// kernels, and quantization strategy for the detected hardware.
type HardwareProfile struct {
	// CPU
	CPUCores  int    // logical CPU count (GOMAXPROCS-visible)
	CPUModel  string // human-readable CPU model string
	HasNEON   bool   // ARM SIMD (Neon)
	HasAVX2   bool   // x86 SIMD (AVX2)
	HasAVX512 bool   // x86 advanced SIMD (AVX-512)
	CacheL1   int64  // L1 data cache size in bytes (0 if unknown)
	CacheL2   int64  // L2 cache size in bytes (0 if unknown)
	CacheL3   int64  // L3 cache size in bytes (0 if unknown)
	TotalRAM  int64  // total physical memory in bytes

	// GPU
	GPUAvailable  bool   // true if a usable GPU was detected
	GPUBackend    string // "cuda", "rocm", "metal", "opencl", or ""
	GPUName       string // human-readable GPU name
	GPUMemory     int64  // GPU memory in bytes (0 if unknown)
	GPUComputeCap string // e.g. "8.9" for CUDA compute capability
	MultiGPU      bool   // true if more than one GPU is available
	GPUCount      int    // number of GPUs (0 if none)
}

// ProfileHardware detects the hardware capabilities of the current system.
// CPU information is always populated. GPU fields are populated on a
// best-effort basis — they remain zero-valued when no GPU is detected.
func ProfileHardware() (*HardwareProfile, error) {
	p := &HardwareProfile{
		CPUCores: runtime.NumCPU(),
	}

	// Detect SIMD from architecture.
	switch runtime.GOARCH {
	case "arm64":
		p.HasNEON = true
	case "amd64":
		p.HasAVX2 = detectAVX2()
		p.HasAVX512 = detectAVX512()
	}

	// OS-specific: CPU model, caches, RAM, GPU.
	if err := profileOS(p); err != nil {
		return p, fmt.Errorf("hardware profile: %w", err)
	}

	return p, nil
}

// RecommendEngine returns the compute backend name that best fits the
// detected hardware: "cuda", "rocm", "metal", "opencl", or "cpu".
func (p *HardwareProfile) RecommendEngine() string {
	if !p.GPUAvailable || p.GPUBackend == "" {
		return "cpu"
	}
	return p.GPUBackend
}

// String returns a human-readable summary of the hardware profile.
func (p *HardwareProfile) String() string {
	var b strings.Builder

	fmt.Fprintf(&b, "CPU: %d cores", p.CPUCores)
	if p.CPUModel != "" {
		fmt.Fprintf(&b, " (%s)", p.CPUModel)
	}

	var simd []string
	if p.HasNEON {
		simd = append(simd, "NEON")
	}
	if p.HasAVX2 {
		simd = append(simd, "AVX2")
	}
	if p.HasAVX512 {
		simd = append(simd, "AVX-512")
	}
	if len(simd) > 0 {
		fmt.Fprintf(&b, ", SIMD: %s", strings.Join(simd, "+"))
	}

	if p.TotalRAM > 0 {
		fmt.Fprintf(&b, ", RAM: %s", formatBytes(p.TotalRAM))
	}

	if p.GPUAvailable {
		fmt.Fprintf(&b, "\nGPU: %s", p.GPUName)
		if p.GPUMemory > 0 {
			fmt.Fprintf(&b, " (%s)", formatBytes(p.GPUMemory))
		}
		if p.GPUComputeCap != "" {
			fmt.Fprintf(&b, " [compute %s]", p.GPUComputeCap)
		}
		fmt.Fprintf(&b, ", backend: %s", p.GPUBackend)
		if p.MultiGPU {
			fmt.Fprintf(&b, ", %d GPUs", p.GPUCount)
		}
	} else {
		b.WriteString("\nGPU: none")
	}

	return b.String()
}

// formatBytes formats a byte count as a human-readable string.
func formatBytes(b int64) string {
	const (
		kiB = 1024
		miB = 1024 * kiB
		giB = 1024 * miB
	)
	switch {
	case b >= giB:
		return fmt.Sprintf("%.1f GiB", float64(b)/float64(giB))
	case b >= miB:
		return fmt.Sprintf("%.1f MiB", float64(b)/float64(miB))
	case b >= kiB:
		return fmt.Sprintf("%.1f KiB", float64(b)/float64(kiB))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

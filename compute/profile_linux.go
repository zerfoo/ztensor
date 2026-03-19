//go:build linux

package compute

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// profileOS populates Linux-specific hardware information from /proc and sysfs.
func profileOS(p *HardwareProfile) error {
	p.CPUModel = readCPUModel()
	p.CacheL1 = readCacheSize(1)
	p.CacheL2 = readCacheSize(2)
	p.CacheL3 = readCacheSize(3)

	var info syscall.Sysinfo_t
	if err := syscall.Sysinfo(&info); err == nil {
		p.TotalRAM = int64(info.Totalram) * int64(info.Unit)
	}

	detectLinuxGPU(p)
	return nil
}

// readCPUModel reads the CPU model from /proc/cpuinfo.
func readCPUModel() string {
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		if strings.HasPrefix(line, "model name") || strings.HasPrefix(line, "Model") {
			if idx := strings.Index(line, ":"); idx >= 0 {
				return strings.TrimSpace(line[idx+1:])
			}
		}
	}
	return ""
}

// readCacheSize reads cache size from sysfs (cpu0) in bytes.
// level is 1, 2, or 3.
func readCacheSize(level int) int64 {
	// Scan cpu0 cache indices for the matching level.
	for i := 0; i < 10; i++ {
		base := "/sys/devices/system/cpu/cpu0/cache/index" + strconv.Itoa(i)

		lb, err := os.ReadFile(base + "/level")
		if err != nil {
			break
		}
		l, _ := strconv.Atoi(strings.TrimSpace(string(lb)))
		if l != level {
			continue
		}

		// Only read data/unified caches, not instruction caches.
		tb, _ := os.ReadFile(base + "/type")
		t := strings.TrimSpace(string(tb))
		if t == "Instruction" {
			continue
		}

		sb, err := os.ReadFile(base + "/size")
		if err != nil {
			continue
		}
		return parseCacheSize(strings.TrimSpace(string(sb)))
	}
	return 0
}

// parseCacheSize parses sysfs cache sizes like "32K", "256K", "8192K".
func parseCacheSize(s string) int64 {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return 0
	}

	multiplier := int64(1)
	switch {
	case strings.HasSuffix(s, "K"):
		multiplier = 1024
		s = s[:len(s)-1]
	case strings.HasSuffix(s, "M"):
		multiplier = 1024 * 1024
		s = s[:len(s)-1]
	case strings.HasSuffix(s, "G"):
		multiplier = 1024 * 1024 * 1024
		s = s[:len(s)-1]
	}

	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0
	}
	return n * multiplier
}

// detectLinuxGPU probes for CUDA and ROCm GPUs.
func detectLinuxGPU(p *HardwareProfile) {
	// Try NVIDIA (CUDA) first via /proc/driver/nvidia/gpus/.
	if detectNvidiaGPU(p) {
		return
	}
	// Try AMD (ROCm) via /sys/class/kfd/kfd/topology/nodes/.
	detectAMDGPU(p)
}

// detectNvidiaGPU checks for NVIDIA GPUs via the nvidia driver proc interface.
func detectNvidiaGPU(p *HardwareProfile) bool {
	entries, err := os.ReadDir("/proc/driver/nvidia/gpus")
	if err != nil || len(entries) == 0 {
		return false
	}

	p.GPUAvailable = true
	p.GPUBackend = "cuda"
	p.GPUCount = len(entries)
	p.MultiGPU = len(entries) > 1

	// Read name from first GPU's information file.
	for _, e := range entries {
		info, err := os.ReadFile("/proc/driver/nvidia/gpus/" + e.Name() + "/information")
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(info), "\n") {
			if strings.HasPrefix(line, "Model:") {
				p.GPUName = strings.TrimSpace(strings.TrimPrefix(line, "Model:"))
				return true
			}
		}
	}

	p.GPUName = "NVIDIA GPU"
	return true
}

// detectAMDGPU checks for AMD GPUs via the KFD topology.
func detectAMDGPU(p *HardwareProfile) {
	entries, err := os.ReadDir("/sys/class/kfd/kfd/topology/nodes")
	if err != nil {
		return
	}

	gpuCount := 0
	for _, e := range entries {
		props, err := os.ReadFile("/sys/class/kfd/kfd/topology/nodes/" + e.Name() + "/properties")
		if err != nil {
			continue
		}
		// GPU nodes have simd_count > 0.
		for _, line := range strings.Split(string(props), "\n") {
			if strings.HasPrefix(line, "simd_count") {
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					if n, _ := strconv.Atoi(parts[1]); n > 0 {
						gpuCount++
					}
				}
			}
		}
	}

	if gpuCount > 0 {
		p.GPUAvailable = true
		p.GPUBackend = "rocm"
		p.GPUName = "AMD GPU (ROCm)"
		p.GPUCount = gpuCount
		p.MultiGPU = gpuCount > 1
	}
}

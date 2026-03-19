package compute

import (
	"runtime"
	"strings"
	"testing"
)

func TestProfileHardware_DetectsCPU(t *testing.T) {
	p, err := ProfileHardware()
	if err != nil {
		t.Fatalf("ProfileHardware() error: %v", err)
	}
	if p == nil {
		t.Fatal("ProfileHardware() returned nil")
	}
	if p.CPUCores <= 0 {
		t.Errorf("CPUCores = %d, want > 0", p.CPUCores)
	}
}

func TestProfileHardware_CPUCores(t *testing.T) {
	p, err := ProfileHardware()
	if err != nil {
		t.Fatalf("ProfileHardware() error: %v", err)
	}
	if got, want := p.CPUCores, runtime.NumCPU(); got != want {
		t.Errorf("CPUCores = %d, want %d", got, want)
	}
}

func TestProfileHardware_SIMD(t *testing.T) {
	p, err := ProfileHardware()
	if err != nil {
		t.Fatalf("ProfileHardware() error: %v", err)
	}

	switch runtime.GOARCH {
	case "arm64":
		if !p.HasNEON {
			t.Error("HasNEON = false on arm64, want true")
		}
		if p.HasAVX2 || p.HasAVX512 {
			t.Error("HasAVX2/HasAVX512 should be false on arm64")
		}
	case "amd64":
		if p.HasNEON {
			t.Error("HasNEON = true on amd64, want false")
		}
		// AVX2 is available on most modern x86-64 CPUs but we only
		// verify the flags are not contradictory (AVX512 implies AVX2).
		if p.HasAVX512 && !p.HasAVX2 {
			t.Error("HasAVX512 = true but HasAVX2 = false; AVX-512 implies AVX2")
		}
	default:
		// On other architectures, no SIMD flags should be set.
		if p.HasNEON || p.HasAVX2 || p.HasAVX512 {
			t.Errorf("unexpected SIMD flags on %s", runtime.GOARCH)
		}
	}
}

func TestProfileHardware_RecommendEngine(t *testing.T) {
	tests := []struct {
		name    string
		profile HardwareProfile
		want    string
	}{
		{
			name:    "cpu only",
			profile: HardwareProfile{CPUCores: 8},
			want:    "cpu",
		},
		{
			name: "cuda gpu",
			profile: HardwareProfile{
				CPUCores:     8,
				GPUAvailable: true,
				GPUBackend:   "cuda",
				GPUName:      "RTX 4090",
			},
			want: "cuda",
		},
		{
			name: "metal gpu",
			profile: HardwareProfile{
				CPUCores:     10,
				GPUAvailable: true,
				GPUBackend:   "metal",
				GPUName:      "Apple M4 Max",
			},
			want: "metal",
		},
		{
			name: "rocm gpu",
			profile: HardwareProfile{
				CPUCores:     16,
				GPUAvailable: true,
				GPUBackend:   "rocm",
				GPUName:      "AMD Instinct MI300X",
			},
			want: "rocm",
		},
		{
			name: "gpu available but backend empty",
			profile: HardwareProfile{
				CPUCores:     4,
				GPUAvailable: true,
				GPUBackend:   "",
			},
			want: "cpu",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.profile.RecommendEngine(); got != tt.want {
				t.Errorf("RecommendEngine() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestProfileHardware_String(t *testing.T) {
	tests := []struct {
		name     string
		profile  HardwareProfile
		contains []string
	}{
		{
			name: "cpu only",
			profile: HardwareProfile{
				CPUCores: 8,
				CPUModel: "Intel Core i7-12700K",
				HasAVX2:  true,
				TotalRAM: 32 * 1024 * 1024 * 1024,
			},
			contains: []string{"8 cores", "Intel Core i7-12700K", "AVX2", "32.0 GiB", "GPU: none"},
		},
		{
			name: "with gpu",
			profile: HardwareProfile{
				CPUCores:      10,
				CPUModel:      "Apple M4 Max",
				HasNEON:       true,
				TotalRAM:      64 * 1024 * 1024 * 1024,
				GPUAvailable:  true,
				GPUBackend:    "metal",
				GPUName:       "Apple M4 Max",
				GPUMemory:     64 * 1024 * 1024 * 1024,
				GPUComputeCap: "",
				GPUCount:      1,
			},
			contains: []string{"10 cores", "NEON", "metal", "Apple M4 Max"},
		},
		{
			name: "multi gpu",
			profile: HardwareProfile{
				CPUCores:      64,
				HasAVX512:     true,
				HasAVX2:       true,
				GPUAvailable:  true,
				GPUBackend:    "cuda",
				GPUName:       "NVIDIA H100",
				GPUMemory:     80 * 1024 * 1024 * 1024,
				GPUComputeCap: "9.0",
				MultiGPU:      true,
				GPUCount:      8,
			},
			contains: []string{"64 cores", "AVX2", "AVX-512", "cuda", "NVIDIA H100", "compute 9.0", "8 GPUs"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := tt.profile.String()
			for _, substr := range tt.contains {
				if !strings.Contains(s, substr) {
					t.Errorf("String() missing %q in:\n%s", substr, s)
				}
			}
		})
	}
}

func TestProfileHardware_RAM(t *testing.T) {
	p, err := ProfileHardware()
	if err != nil {
		t.Fatalf("ProfileHardware() error: %v", err)
	}
	// On darwin and linux, TotalRAM should be detected.
	switch runtime.GOOS {
	case "darwin", "linux":
		if p.TotalRAM <= 0 {
			t.Errorf("TotalRAM = %d, want > 0 on %s", p.TotalRAM, runtime.GOOS)
		}
	}
}

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		input int64
		want  string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{1024, "1.0 KiB"},
		{1536, "1.5 KiB"},
		{1048576, "1.0 MiB"},
		{1073741824, "1.0 GiB"},
		{17179869184, "16.0 GiB"},
	}
	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := formatBytes(tt.input); got != tt.want {
				t.Errorf("formatBytes(%d) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

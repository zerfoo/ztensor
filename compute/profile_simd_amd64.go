//go:build amd64

package compute

// detectAVX2 checks for AVX2 support via CPUID.
// CPUID leaf 7, sub-leaf 0, EBX bit 5.
func detectAVX2() bool {
	_, ebx, _, _ := cpuid(7, 0)
	return ebx&(1<<5) != 0
}

// detectAVX512 checks for AVX-512 Foundation support via CPUID.
// CPUID leaf 7, sub-leaf 0, EBX bit 16.
func detectAVX512() bool {
	_, ebx, _, _ := cpuid(7, 0)
	return ebx&(1<<16) != 0
}

// cpuid executes the CPUID instruction with the given leaf and sub-leaf.
func cpuid(leaf, subLeaf uint32) (eax, ebx, ecx, edx uint32)

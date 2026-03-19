//go:build !darwin && !linux

package compute

// profileOS is a no-op on unsupported platforms.
// CPU cores and SIMD flags are set in the platform-independent code.
func profileOS(p *HardwareProfile) error {
	return nil
}

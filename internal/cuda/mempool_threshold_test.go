package cuda

import "testing"

// TestSetMemPoolReleaseThreshold_NoOpWithoutSymbols verifies the hardening call
// is a silent no-op when the CUDA runtime (or the mempool symbols) are absent,
// which is the case on a CUDA-less test host (issue #118). On a host with CUDA
// it still returns nil after setting the attribute for real.
func TestSetMemPoolReleaseThreshold_NoOpWithoutSymbols(t *testing.T) {
	if err := setMemPoolReleaseThreshold(0, 64<<30); err != nil {
		t.Fatalf("expected nil (no-op without symbols), got %v", err)
	}
}

// TestSetMemPoolReleaseThreshold_RoutesThroughIndirection verifies the public
// entry point dispatches through the swappable indirection with the device and
// byte count intact, so the engine wiring can be asserted without a device.
func TestSetMemPoolReleaseThreshold_RoutesThroughIndirection(t *testing.T) {
	var (
		gotDev   int
		gotBytes uint64
		called   int
	)
	orig := memPoolSetReleaseThresholdFn
	memPoolSetReleaseThresholdFn = func(dev int, b uint64) error {
		called++
		gotDev = dev
		gotBytes = b
		return nil
	}
	t.Cleanup(func() { memPoolSetReleaseThresholdFn = orig })

	if err := SetMemPoolReleaseThreshold(3, 12345); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if called != 1 || gotDev != 3 || gotBytes != 12345 {
		t.Fatalf("routing failed: called=%d dev=%d bytes=%d (want 1/3/12345)",
			called, gotDev, gotBytes)
	}
}

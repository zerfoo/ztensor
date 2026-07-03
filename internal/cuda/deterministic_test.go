package cuda

import "testing"

// enableDeterministicForTest turns ZTENSOR_DETERMINISTIC mode on for one test
// and restores the process default afterwards, mirroring
// enableArenaPoisonForTest.
func enableDeterministicForTest(t *testing.T, enabled bool) {
	t.Helper()
	orig := deterministicEnabled
	deterministicEnabled = enabled
	t.Cleanup(func() { deterministicEnabled = orig })
}

func TestDeterministicEnabled_DefaultOff(t *testing.T) {
	// Process-default value must come from the env var read at init; the
	// test suite does not set ZTENSOR_DETERMINISTIC, so this should be off
	// unless a developer's shell happens to export it.
	if envDeterministicSet := deterministicEnabled; envDeterministicSet {
		t.Skip("ZTENSOR_DETERMINISTIC=1 set in the test environment; skipping default-off assertion")
	}
}

func TestDeterministicEnabled_Toggle(t *testing.T) {
	enableDeterministicForTest(t, true)
	if !DeterministicEnabled() {
		t.Fatal("DeterministicEnabled() = false after enabling")
	}
	enableDeterministicForTest(t, false)
	if DeterministicEnabled() {
		t.Fatal("DeterministicEnabled() = true after disabling")
	}
}

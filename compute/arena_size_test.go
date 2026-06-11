package compute

import (
	"testing"

	"github.com/zerfoo/ztensor/log"
)

func TestArenaSizeBytes_Default(t *testing.T) {
	t.Setenv("ZERFOO_ARENA_SIZE_GB", "")
	got := arenaSizeBytes(log.Nop())
	want := int64(defaultArenaSizeGB) * 1024 * 1024 * 1024
	if got != want {
		t.Fatalf("default arena size: got %d, want %d", got, want)
	}
}

func TestArenaSizeBytes_EnvOverride(t *testing.T) {
	tests := []struct {
		name  string
		env   string
		wantG int64
	}{
		{"min", "1", 1},
		{"training-typical", "32", 32},
		{"max", "128", 128},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("ZERFOO_ARENA_SIZE_GB", tt.env)
			got := arenaSizeBytes(log.Nop())
			want := tt.wantG * 1024 * 1024 * 1024
			if got != want {
				t.Fatalf("env=%q: got %d, want %d", tt.env, got, want)
			}
		})
	}
}

func TestArenaSizeBytes_InvalidFallsBackToDefault(t *testing.T) {
	tests := []struct {
		name string
		env  string
	}{
		{"non-integer", "lots"},
		{"below-min", "0"},
		{"above-max", "256"},
		{"negative", "-5"},
	}
	wantDefault := int64(defaultArenaSizeGB) * 1024 * 1024 * 1024
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("ZERFOO_ARENA_SIZE_GB", tt.env)
			got := arenaSizeBytes(log.Nop())
			if got != wantDefault {
				t.Fatalf("env=%q: got %d, want default %d", tt.env, got, wantDefault)
			}
		})
	}
}

func TestArenaSizeBytes_TestingOverrideWinsAndRestores(t *testing.T) {
	// The override beats the env var and is not bound to GB granularity or
	// the [min, max] range -- it exists so the parity harness can build a
	// deliberately tiny arena (testing/parity).
	t.Setenv("ZERFOO_ARENA_SIZE_GB", "32")
	restore := SetArenaBytesForTesting(64 << 20)
	if got := arenaSizeBytes(log.Nop()); got != 64<<20 {
		restore()
		t.Fatalf("with override: got %d, want %d", got, int64(64<<20))
	}
	restore()
	if got, want := arenaSizeBytes(log.Nop()), int64(32)*1024*1024*1024; got != want {
		t.Fatalf("after restore: got %d, want env-driven %d", got, want)
	}
}

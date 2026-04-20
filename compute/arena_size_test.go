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

package stablehlo

import (
	"testing"
)

func TestFormatTensorType(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		dtype string
		want  string
	}{
		{"3D f32", []int{2, 3, 4}, "f32", "tensor<2x3x4xf32>"},
		{"2D f64", []int{8, 16}, "f64", "tensor<8x16xf64>"},
		{"1D i32", []int{10}, "i32", "tensor<10xi32>"},
		{"4D bf16", []int{1, 2, 3, 4}, "bf16", "tensor<1x2x3x4xbf16>"},
		{"scalar", []int{}, "f32", "tensor<f32>"},
		{"f16", []int{3, 5}, "f16", "tensor<3x5xf16>"},
		{"i64", []int{100}, "i64", "tensor<100xi64>"},
		{"f8E4M3FN", []int{4, 8}, "f8E4M3FN", "tensor<4x8xf8E4M3FN>"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FormatTensorType(tt.shape, tt.dtype)
			if got != tt.want {
				t.Errorf("FormatTensorType(%v, %q) = %q, want %q", tt.shape, tt.dtype, got, tt.want)
			}
		})
	}
}

func TestFormatScalarType(t *testing.T) {
	tests := []struct {
		dtype string
		want  string
	}{
		{"f32", "f32"},
		{"f64", "f64"},
		{"i32", "i32"},
		{"bf16", "bf16"},
	}
	for _, tt := range tests {
		t.Run(tt.dtype, func(t *testing.T) {
			got := FormatScalarType(tt.dtype)
			if got != tt.want {
				t.Errorf("FormatScalarType(%q) = %q, want %q", tt.dtype, got, tt.want)
			}
		})
	}
}

func TestSSANamer(t *testing.T) {
	n := &SSANamer{}

	if n.Count() != 0 {
		t.Fatalf("initial count = %d, want 0", n.Count())
	}

	expected := []string{"%v0", "%v1", "%v2", "%v3", "%v4"}
	for i, want := range expected {
		got := n.NextName()
		if got != want {
			t.Errorf("NextName() call %d = %q, want %q", i, got, want)
		}
	}

	if n.Count() != 5 {
		t.Errorf("count after 5 calls = %d, want 5", n.Count())
	}
}

func TestSSANamerConcurrent(t *testing.T) {
	n := &SSANamer{}
	const goroutines = 100

	done := make(chan string, goroutines)
	for range goroutines {
		go func() {
			done <- n.NextName()
		}()
	}

	seen := make(map[string]bool, goroutines)
	for range goroutines {
		name := <-done
		if seen[name] {
			t.Errorf("duplicate SSA name: %s", name)
		}
		seen[name] = true
	}

	if n.Count() != goroutines {
		t.Errorf("count = %d, want %d", n.Count(), goroutines)
	}
}

func TestGoDTypeToMLIR(t *testing.T) {
	tests := []struct {
		goType string
		want   string
		ok     bool
	}{
		{"float32", "f32", true},
		{"float64", "f64", true},
		{"float16", "f16", true},
		{"bfloat16", "bf16", true},
		{"float8", "f8E4M3FN", true},
		{"int8", "i8", true},
		{"int16", "i16", true},
		{"int32", "i32", true},
		{"int64", "i64", true},
		{"uint8", "ui8", true},
		{"uint32", "ui32", true},
		{"uint64", "ui64", true},
		{"complex64", "", false},
		{"string", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.goType, func(t *testing.T) {
			got, ok := GoDTypeToMLIR(tt.goType)
			if ok != tt.ok {
				t.Errorf("GoDTypeToMLIR(%q) ok = %v, want %v", tt.goType, ok, tt.ok)
			}
			if got != tt.want {
				t.Errorf("GoDTypeToMLIR(%q) = %q, want %q", tt.goType, got, tt.want)
			}
		})
	}
}

func TestOpConstants(t *testing.T) {
	// Verify key op constants have the expected stablehlo. prefix and op names.
	ops := map[string]string{
		"OpAdd":         OpAdd,
		"OpSubtract":    OpSubtract,
		"OpMultiply":    OpMultiply,
		"OpDivide":      OpDivide,
		"OpDotGeneral":  OpDotGeneral,
		"OpTranspose":   OpTranspose,
		"OpReshape":     OpReshape,
		"OpBroadcastIn": OpBroadcastIn,
		"OpReduce":      OpReduce,
		"OpExp":         OpExp,
		"OpLog":         OpLog,
		"OpTanh":        OpTanh,
		"OpConstant":    OpConstant,
	}
	for name, val := range ops {
		if len(val) < 12 || val[:10] != "stablehlo." {
			t.Errorf("%s = %q, expected stablehlo.* prefix", name, val)
		}
	}
}

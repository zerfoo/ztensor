package tensor

import "testing"

func TestTernaryStorage_RoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		values []int8
	}{
		{"all negatives", []int8{-1, -1, -1, -1}},
		{"all zeros", []int8{0, 0, 0, 0}},
		{"all ones", []int8{1, 1, 1, 1}},
		{"mixed", []int8{-1, 0, 1, 0}},
		{"single -1", []int8{-1}},
		{"single 0", []int8{0}},
		{"single 1", []int8{1}},
		{"not multiple of 4", []int8{-1, 0, 1}},
		{"five elements", []int8{1, -1, 0, 1, -1}},
		{"seven elements", []int8{-1, 1, 0, -1, 1, 0, 1}},
		{"empty", []int8{}},
		{"sixteen elements", []int8{-1, 0, 1, 0, 1, -1, -1, 0, 0, 1, 1, -1, 0, 0, -1, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewTernaryStorageFrom(tt.values)
			if s.Len() != len(tt.values) {
				t.Fatalf("Len() = %d, want %d", s.Len(), len(tt.values))
			}
			for i, want := range tt.values {
				got := s.Get(i)
				if got != want {
					t.Errorf("Get(%d) = %d, want %d", i, got, want)
				}
			}
		})
	}
}

func TestTernaryStorage_Set(t *testing.T) {
	s := NewTernaryStorage(4)
	vals := []int8{1, -1, 0, 1}
	for i, v := range vals {
		s.SetElement(i, v)
	}
	for i, want := range vals {
		got := s.Get(i)
		if got != want {
			t.Errorf("Get(%d) = %d, want %d", i, got, want)
		}
	}

	// Overwrite and verify
	s.SetElement(0, -1)
	s.SetElement(3, 0)
	if got := s.Get(0); got != -1 {
		t.Errorf("after overwrite Get(0) = %d, want -1", got)
	}
	if got := s.Get(3); got != 0 {
		t.Errorf("after overwrite Get(3) = %d, want 0", got)
	}
	// Ensure other values unchanged
	if got := s.Get(1); got != -1 {
		t.Errorf("Get(1) = %d, want -1", got)
	}
	if got := s.Get(2); got != 0 {
		t.Errorf("Get(2) = %d, want 0", got)
	}
}

func TestTernaryStorage_Slice(t *testing.T) {
	values := []int8{-1, 0, 1, 0, -1}
	s := NewTernaryStorageFrom(values)
	out := s.Slice()
	if len(out) != len(values) {
		t.Fatalf("Slice() len = %d, want %d", len(out), len(values))
	}
	for i, want := range values {
		if out[i] != float32(want) {
			t.Errorf("Slice()[%d] = %g, want %g", i, out[i], float32(want))
		}
	}
}

func TestTernaryStorage_SizeCompaction(t *testing.T) {
	// 1024 float32 values = 4096 bytes.
	// 1024 ternary values = 256 bytes (16x smaller than float32, 4x smaller than int8).
	n := 1024
	s := NewTernaryStorage(n)
	rawSize := len(s.RawBytes())
	float32Size := n * 4 // 4 bytes per float32
	int8Size := n        // 1 byte per int8

	if rawSize != 256 {
		t.Errorf("RawBytes size = %d, want 256", rawSize)
	}
	if float32Size/rawSize != 16 {
		t.Errorf("float32 compaction ratio = %d, want 16 (float32=%d, ternary=%d)",
			float32Size/rawSize, float32Size, rawSize)
	}
	if int8Size/rawSize != 4 {
		t.Errorf("int8 compaction ratio = %d, want 4 (int8=%d, ternary=%d)",
			int8Size/rawSize, int8Size, rawSize)
	}
}

func TestTernaryStorage_PanicOnInvalidValue(t *testing.T) {
	s := NewTernaryStorage(4)
	for _, val := range []int8{-2, 2, 5, -128, 127} {
		func(v int8) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("Set(0, %d) did not panic", v)
				}
			}()
			s.SetElement(0, v)
		}(val)
	}
}

func TestTernaryStorage_PanicOnOutOfRange(t *testing.T) {
	s := NewTernaryStorage(4)

	// Get out of range
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Get(4) did not panic")
			}
		}()
		s.Get(4)
	}()

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Get(-1) did not panic")
			}
		}()
		s.Get(-1)
	}()

	// Set out of range
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Set(4, 0) did not panic")
			}
		}()
		s.SetElement(4, 0)
	}()
}

func TestTernaryStorage_DeviceType(t *testing.T) {
	s := NewTernaryStorage(1)
	if dt := s.DeviceType(); dt != 0 { // device.CPU = 0
		t.Errorf("DeviceType() = %v, want CPU (0)", dt)
	}
}

func TestTernaryStorage_Empty(t *testing.T) {
	s := NewTernaryStorage(0)
	if s.Len() != 0 {
		t.Errorf("Len() = %d, want 0", s.Len())
	}
	if out := s.Slice(); len(out) != 0 {
		t.Errorf("Slice() len = %d, want 0", len(out))
	}
	if len(s.RawBytes()) != 0 {
		t.Errorf("RawBytes() len = %d, want 0", len(s.RawBytes()))
	}
}

func TestTernaryStorage_ByteAlignment(t *testing.T) {
	// Verify correct byte count for various sizes
	tests := []struct {
		n         int
		wantBytes int
	}{
		{0, 0},
		{1, 1},
		{2, 1},
		{3, 1},
		{4, 1},
		{5, 2},
		{8, 2},
		{9, 3},
		{100, 25},
	}
	for _, tt := range tests {
		s := NewTernaryStorage(tt.n)
		if got := len(s.RawBytes()); got != tt.wantBytes {
			t.Errorf("NewTernaryStorage(%d): RawBytes len = %d, want %d", tt.n, got, tt.wantBytes)
		}
	}
}

//go:build unix

package tensor

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMmapFile(t *testing.T) {
	// Create a temp file with known content.
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bin")
	content := []byte("hello mmap world! this is test data for the mmap helper.")
	if err := os.WriteFile(path, content, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	data, closer, err := MmapFile(path)
	if err != nil {
		t.Fatalf("MmapFile: %v", err)
	}
	defer func() {
		if err := closer(); err != nil {
			t.Errorf("closer: %v", err)
		}
	}()

	if len(data) != len(content) {
		t.Errorf("len(data) = %d, want %d", len(data), len(content))
	}

	for i, b := range content {
		if data[i] != b {
			t.Errorf("data[%d] = %d, want %d", i, data[i], b)
			break
		}
	}
}

func TestMmapFile_Empty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.bin")
	if err := os.WriteFile(path, nil, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, _, err := MmapFile(path)
	if err == nil {
		t.Fatal("expected error for empty file, got nil")
	}
}

func TestMmapFile_NonExistent(t *testing.T) {
	_, _, err := MmapFile("/nonexistent/path/file.bin")
	if err == nil {
		t.Fatal("expected error for non-existent file, got nil")
	}
}

func TestMmap_Munmap(t *testing.T) {
	// Create a temp file.
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bin")
	content := make([]byte, 4096)
	for i := range content {
		content[i] = byte(i % 256)
	}
	if err := os.WriteFile(path, content, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	data, err := Mmap(f.Fd(), 0, len(content))
	if err != nil {
		t.Fatalf("Mmap: %v", err)
	}

	// Verify content.
	for i := range 100 {
		if data[i] != content[i] {
			t.Errorf("data[%d] = %d, want %d", i, data[i], content[i])
			break
		}
	}

	// Munmap should succeed.
	if err := Munmap(data); err != nil {
		t.Errorf("Munmap: %v", err)
	}
}

func TestMmapFile_F32Tensor(t *testing.T) {
	// End-to-end: write F32 data to file, mmap it, create MmapStorage, verify.
	dir := t.TempDir()
	path := filepath.Join(dir, "tensor.bin")

	values := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	raw := makeF32Raw(values)
	if err := os.WriteFile(path, raw, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	data, closer, err := MmapFile(path)
	if err != nil {
		t.Fatalf("MmapFile: %v", err)
	}
	defer func() { _ = closer() }()

	s, err := NewMmapStorage(data, len(values), GGMLTypeF32)
	if err != nil {
		t.Fatalf("NewMmapStorage: %v", err)
	}

	tensor, err := NewWithStorage[float32]([]int{2, 3}, s)
	if err != nil {
		t.Fatalf("NewWithStorage: %v", err)
	}

	got := tensor.Data()
	for i, want := range values {
		if got[i] != want {
			t.Errorf("Data()[%d] = %v, want %v", i, got[i], want)
		}
	}
}

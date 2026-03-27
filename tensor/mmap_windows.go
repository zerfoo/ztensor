//go:build windows

package tensor

import (
	"fmt"
)

// MmapFile is not yet implemented on Windows. Use heap-based loading instead.
func MmapFile(path string) (data []byte, closer func() error, err error) {
	return nil, nil, fmt.Errorf("mmap: not supported on Windows; use heap loading")
}

// Mmap is not yet implemented on Windows.
func Mmap(fd uintptr, offset int64, length int) ([]byte, error) {
	return nil, fmt.Errorf("mmap: not supported on Windows")
}

// Munmap is not yet implemented on Windows.
func Munmap(data []byte) error {
	return fmt.Errorf("munmap: not supported on Windows")
}

// MadviseSequential is a no-op on Windows.
func MadviseSequential(_ []byte) error { return nil }

// MadviseRandom is a no-op on Windows.
func MadviseRandom(_ []byte) error { return nil }

// MadviseWillNeed is a no-op on Windows.
func MadviseWillNeed(_ []byte) error { return nil }

// MadviseDontNeed is a no-op on Windows.
func MadviseDontNeed(_ []byte) error { return nil }

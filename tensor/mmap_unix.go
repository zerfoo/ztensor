//go:build unix

package tensor

import (
	"fmt"
	"os"
	"syscall"
)

// MmapFile memory-maps the entire file at the given path for reading.
// It returns the mapped byte slice and a cleanup function that unmaps the region.
// The caller must call the cleanup function when done to release the mapping.
//
// On Unix (Linux, Darwin), this uses syscall.Mmap with PROT_READ and MAP_PRIVATE.
func MmapFile(path string) (data []byte, closer func() error, err error) {
	f, err := os.Open(path) //nolint:gosec // Path is caller-provided, used for model loading
	if err != nil {
		return nil, nil, fmt.Errorf("mmap open: %w", err)
	}

	fi, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("mmap stat: %w", err)
	}

	size := fi.Size()
	if size == 0 {
		_ = f.Close()
		return nil, nil, fmt.Errorf("mmap: file is empty")
	}
	if size > 1<<40 { // 1 TB sanity limit
		_ = f.Close()
		return nil, nil, fmt.Errorf("mmap: file too large (%d bytes)", size)
	}

	mapped, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("mmap syscall: %w", err)
	}

	// Close the file descriptor; the mapping remains valid after close.
	_ = f.Close()

	cleanup := func() error {
		return syscall.Munmap(mapped)
	}

	return mapped, cleanup, nil
}

// Mmap maps a region of a file descriptor into memory.
func Mmap(fd uintptr, offset int64, length int) ([]byte, error) {
	if length <= 0 {
		return nil, fmt.Errorf("mmap: length must be positive, got %d", length)
	}
	data, err := syscall.Mmap(int(fd), offset, length, syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}
	return data, nil
}

// Munmap releases a previously mapped memory region.
func Munmap(data []byte) error {
	return syscall.Munmap(data)
}

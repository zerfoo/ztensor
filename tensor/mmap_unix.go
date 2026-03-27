//go:build unix

package tensor

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
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

// MadviseSequential hints to the kernel that the mmap'd region will be
// accessed sequentially. This enables aggressive read-ahead, which is
// optimal during model loading when all tensors are read in order.
func MadviseSequential(data []byte) error {
	return madvise(data, syscall.MADV_SEQUENTIAL)
}

// MadviseRandom hints to the kernel that the mmap'd region will be
// accessed randomly. This disables read-ahead, which is optimal during
// inference when individual transformer layers are accessed in
// unpredictable patterns (especially with MoE or speculative decoding).
func MadviseRandom(data []byte) error {
	return madvise(data, syscall.MADV_RANDOM)
}

// MadviseWillNeed hints to the kernel that the specified region will be
// needed soon. The kernel may start paging in the data asynchronously.
// Use this to prefetch the next transformer layer's weights while the
// current layer is computing.
func MadviseWillNeed(data []byte) error {
	return madvise(data, syscall.MADV_WILLNEED)
}

// MadviseDontNeed hints to the kernel that the specified region is no
// longer needed. The kernel may free the physical pages, reducing RSS.
// Use after processing a layer to release pages back to the OS.
func MadviseDontNeed(data []byte) error {
	return madvise(data, syscall.MADV_DONTNEED)
}

func madvise(data []byte, advice int) error {
	if len(data) == 0 {
		return nil
	}
	_, _, errno := syscall.Syscall(syscall.SYS_MADVISE,
		uintptr(unsafe.Pointer(&data[0])),
		uintptr(len(data)),
		uintptr(advice))
	if errno != 0 {
		return fmt.Errorf("madvise(%d): %w", advice, errno)
	}
	return nil
}

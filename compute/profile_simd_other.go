//go:build !amd64

package compute

// detectAVX2 always returns false on non-x86 platforms.
func detectAVX2() bool { return false }

// detectAVX512 always returns false on non-x86 platforms.
func detectAVX512() bool { return false }

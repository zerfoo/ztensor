//go:build darwin

package metal

import "unsafe"

// strToPtr converts a Go string to a null-terminated C string pointer.
// The returned pointer is only valid for the duration of the call.
func strToPtr(s string) unsafe.Pointer {
	b := append([]byte(s), 0)
	return unsafe.Pointer(unsafe.SliceData(b))
}

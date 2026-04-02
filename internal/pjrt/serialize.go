package pjrt

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// Serialize serializes the compiled executable to bytes. The serialized
// form can be cached to disk and later restored with Client.DeserializeAndLoad,
// skipping recompilation on subsequent runs with the same model and hardware.
func (e *LoadedExecutable) Serialize() ([]byte, error) {
	if e.handle == 0 {
		return nil, fmt.Errorf("pjrt: cannot serialize closed executable")
	}
	if e.lib.PJRT_Executable_Serialize == 0 {
		return nil, fmt.Errorf("pjrt: plugin does not support PJRT_Executable_Serialize")
	}

	// PJRT_Executable_Serialize_Args:
	//   struct_size        uintptr
	//   executable         uintptr  (PJRT_LoadedExecutable*)
	//   serialized_bytes   uintptr  (out: const char*)
	//   serialized_bytes_size uintptr (out: size_t)
	type serializeArgs struct {
		structSize          uintptr
		executable          uintptr
		serializedBytes     uintptr
		serializedBytesSize uintptr
	}
	args := serializeArgs{
		structSize: unsafe.Sizeof(serializeArgs{}),
		executable: e.handle,
	}

	errPtr := cuda.Ccall(e.lib.PJRT_Executable_Serialize, uintptr(unsafe.Pointer(&args)))
	if err := e.lib.checkError(errPtr); err != nil {
		return nil, fmt.Errorf("PJRT_Executable_Serialize: %w", err)
	}
	if args.serializedBytes == 0 || args.serializedBytesSize == 0 {
		return nil, fmt.Errorf("pjrt: PJRT_Executable_Serialize returned empty result")
	}

	// Copy the serialized bytes into a Go-managed slice. The C pointer
	// is owned by the PJRT runtime and may be invalidated when the
	// executable is destroyed.
	n := int(args.serializedBytesSize)
	src := unsafe.Slice((*byte)(unsafe.Pointer(args.serializedBytes)), n)
	out := make([]byte, n)
	copy(out, src)

	return out, nil
}

// DeserializeAndLoad restores a previously serialized executable, returning
// a LoadedExecutable ready for execution. This skips the compilation step
// entirely, which can save significant time for large models.
//
// The serialized data must have been produced by Serialize() on the same
// plugin and hardware platform.
func (c *Client) DeserializeAndLoad(data []byte) (*LoadedExecutable, error) {
	if c.handle == 0 {
		return nil, fmt.Errorf("pjrt: cannot deserialize on closed client")
	}
	if c.lib.PJRT_Executable_DeserializeAndLoad == 0 {
		return nil, fmt.Errorf("pjrt: plugin does not support PJRT_Executable_DeserializeAndLoad")
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("pjrt: cannot deserialize empty data")
	}

	// PJRT_Executable_DeserializeAndLoad_Args:
	//   struct_size          uintptr
	//   client               uintptr  (PJRT_Client*)
	//   serialized_executable       uintptr  (const char*)
	//   serialized_executable_size  uintptr  (size_t)
	//   loaded_executable    uintptr  (out: PJRT_LoadedExecutable*)
	type deserializeArgs struct {
		structSize               uintptr
		client                   uintptr
		serializedExecutable     uintptr
		serializedExecutableSize uintptr
		loadedExecutable         uintptr
	}
	args := deserializeArgs{
		structSize:               unsafe.Sizeof(deserializeArgs{}),
		client:                   c.handle,
		serializedExecutable:     uintptr(unsafe.Pointer(&data[0])),
		serializedExecutableSize: uintptr(len(data)),
	}

	errPtr := cuda.Ccall(c.lib.PJRT_Executable_DeserializeAndLoad, uintptr(unsafe.Pointer(&args)))
	if err := c.lib.checkError(errPtr); err != nil {
		return nil, fmt.Errorf("PJRT_Executable_DeserializeAndLoad: %w", err)
	}
	if args.loadedExecutable == 0 {
		return nil, fmt.Errorf("pjrt: PJRT_Executable_DeserializeAndLoad returned null executable")
	}

	exec := &LoadedExecutable{lib: c.lib, handle: args.loadedExecutable}

	// Query and cache output metadata, same as after Compile.
	if err := exec.queryOutputMetadata(); err != nil {
		exec.Close()
		return nil, fmt.Errorf("pjrt: query output metadata after deserialize: %w", err)
	}

	return exec, nil
}

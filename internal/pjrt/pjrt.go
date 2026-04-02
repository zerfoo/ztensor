// Package pjrt provides purego bindings for the PJRT C API.
//
// PJRT (Portable JAX Runtime) is OpenXLA's hardware plugin API.
// A single PJRT integration gives access to every accelerator that
// ships a PJRT plugin (CPU, CUDA, TPU, Trainium, Metal) without
// per-backend kernel work.
//
// All bindings use dlopen/dlsym via the cuda package's exported
// helpers — zero CGo.
package pjrt

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// pjrtApiHeaderSize is the byte size of the PJRT_Api header:
// struct_size (8 bytes) + pjrt_api_version (8 bytes).
const pjrtApiHeaderSize = 16

// pjrtApiHeader mirrors the first two fields of the C PJRT_Api struct.
type pjrtApiHeader struct {
	StructSize     uintptr
	PjrtApiVersion uintptr
}

// pjrtApiTable mirrors the PJRT_Api struct function pointer slots
// after the header. Each field is a C function pointer.
type pjrtApiTable struct {
	PJRT_Error_Destroy                 uintptr // slot 0
	PJRT_Error_Message                 uintptr // slot 1
	PJRT_Plugin_Initialize             uintptr // slot 2
	PJRT_Client_Create                 uintptr // slot 3
	PJRT_Client_Destroy                uintptr // slot 4
	PJRT_Client_PlatformName           uintptr // slot 5
	PJRT_Client_PlatformVersion        uintptr // slot 6
	PJRT_Client_Devices                uintptr // slot 7
	PJRT_Client_AddressableDevices     uintptr // slot 8
	PJRT_Client_Compile                uintptr // slot 9
	PJRT_Client_BufferFromHostBuffer   uintptr // slot 10
	PJRT_Buffer_ToHostBuffer           uintptr // slot 11
	PJRT_Buffer_OnDeviceSizeInBytes    uintptr // slot 12
	PJRT_Buffer_Destroy                uintptr // slot 13
	PJRT_Buffer_Delete                 uintptr // slot 14
	PJRT_Buffer_ElementType            uintptr // slot 15
	PJRT_Buffer_Dimensions             uintptr // slot 16
	PJRT_Buffer_UnsafePointer          uintptr // slot 17
	PJRT_Buffer_ReadyEvent             uintptr // slot 18
	PJRT_Device_GetDescription         uintptr // slot 19
	PJRT_Device_IsAddressable          uintptr // slot 20
	PJRT_Device_LocalHardwareId        uintptr // slot 21
	PJRT_DeviceDescription_Id          uintptr // slot 22
	PJRT_DeviceDescription_Kind        uintptr // slot 23
	PJRT_LoadedExecutable_Execute      uintptr // slot 24
	PJRT_LoadedExecutable_Destroy      uintptr // slot 25
	PJRT_LoadedExecutable_Delete       uintptr // slot 26
	PJRT_Executable_NumOutputs         uintptr // slot 27
	PJRT_Executable_Serialize          uintptr // slot 28
	PJRT_Executable_DeserializeAndLoad uintptr // slot 29
	PJRT_Event_Await                   uintptr // slot 30
	PJRT_Event_Destroy                 uintptr // slot 31
}

// PJRTLib holds a dlopen handle for a PJRT plugin and the resolved
// function pointers extracted from the PJRT_Api struct returned by
// GetPjrtApi().
type PJRTLib struct {
	handle uintptr        // dlopen handle for the plugin .so
	api    unsafe.Pointer // pointer to the PJRT_Api struct

	// Version reported by the plugin.
	VersionMajor int
	VersionMinor int

	// Function pointers extracted from the PJRT_Api struct.
	pjrtApiTable
}

// pluginSearchPaths returns the ordered list of directories to search for
// a PJRT plugin shared library. $PJRT_PLUGIN_PATH is checked first.
func pluginSearchPaths() []string {
	var paths []string

	// 1. Explicit environment variable (colon-separated).
	if env := os.Getenv("PJRT_PLUGIN_PATH"); env != "" {
		paths = append(paths, strings.Split(env, ":")...)
	}

	// 2. Standard system paths.
	paths = append(paths,
		"/usr/lib",
		"/usr/local/lib",
	)

	// 3. AWS Neuron SDK (Trainium/Inferentia).
	paths = append(paths, "/opt/aws/neuron/lib")

	// 4. GoMLX convention.
	if home, err := os.UserHomeDir(); err == nil {
		paths = append(paths, filepath.Join(home, ".local", "lib", "go-xla"))
	}

	// 5. Common XLA/JAX pip install locations.
	paths = append(paths,
		"/usr/local/lib/python3/dist-packages/jaxlib",
		"/usr/lib/python3/dist-packages/jaxlib",
	)

	return paths
}

// Load opens a PJRT plugin shared library and extracts all function
// pointers from the PJRT_Api struct. pluginName is the bare library
// filename (e.g. "pjrt_c_api_cpu_plugin.so").
//
// Load searches $PJRT_PLUGIN_PATH, standard system directories,
// AWS Neuron paths, and Python site-packages.
//
// Returns a clean error if the plugin is not found or if the API
// version is incompatible.
func Load(pluginName string) (*PJRTLib, error) {
	candidates := []string{pluginName}
	for _, dir := range pluginSearchPaths() {
		candidates = append(candidates, filepath.Join(dir, pluginName))
	}

	lib := &PJRTLib{}
	var lastErr string
	for _, path := range candidates {
		h, err := dlopenPath(path)
		if err == nil {
			lib.handle = h
			break
		}
		lastErr = err.Error()
	}
	if lib.handle == 0 {
		return nil, fmt.Errorf("pjrt: plugin %q not found (last: %s)", pluginName, lastErr)
	}

	// Resolve the single entry point.
	getPjrtApi, err := dlsym(lib.handle, "GetPjrtApi")
	if err != nil {
		lib.Close()
		return nil, fmt.Errorf("pjrt: dlsym GetPjrtApi: %w", err)
	}

	// Call GetPjrtApi() -> *PJRT_Api.
	apiPtr := ccall(getPjrtApi)
	if apiPtr == 0 {
		lib.Close()
		return nil, fmt.Errorf("pjrt: GetPjrtApi returned null")
	}
	lib.api = unsafe.Pointer(apiPtr) //nolint:govet // apiPtr is a valid C pointer from GetPjrtApi

	// Read struct_size and pjrt_api_version from the header.
	header := (*pjrtApiHeader)(lib.api)
	structSize := header.StructSize
	versionField := header.PjrtApiVersion

	// pjrt_api_version packs major in high 32 bits, minor in low 32.
	lib.VersionMajor = int(versionField >> 32)
	lib.VersionMinor = int(versionField & 0xFFFFFFFF)

	if structSize < pjrtApiHeaderSize {
		lib.Close()
		return nil, fmt.Errorf("pjrt: PJRT_Api struct_size %d too small (need >= %d)",
			structSize, pjrtApiHeaderSize)
	}

	// Read the function pointer table. We overlay pjrtApiTable at the
	// offset after the header, reading only as many slots as the plugin
	// struct_size allows.
	tableBytes := structSize - pjrtApiHeaderSize
	goTableSize := unsafe.Sizeof(pjrtApiTable{})
	tablePtr := unsafe.Add(lib.api, pjrtApiHeaderSize)

	if tableBytes >= goTableSize {
		// Plugin has all slots we need — direct overlay.
		lib.pjrtApiTable = *(*pjrtApiTable)(tablePtr)
	} else {
		// Older plugin with fewer slots. Copy only available bytes.
		src := unsafe.Slice((*byte)(tablePtr), tableBytes)
		dst := unsafe.Slice((*byte)(unsafe.Pointer(&lib.pjrtApiTable)), goTableSize)
		copy(dst, src)
	}

	// Validate that essential function pointers are present.
	type requiredFn struct {
		name string
		ptr  uintptr
	}
	required := []requiredFn{
		{"PJRT_Error_Destroy", lib.PJRT_Error_Destroy},
		{"PJRT_Error_Message", lib.PJRT_Error_Message},
		{"PJRT_Client_Create", lib.PJRT_Client_Create},
		{"PJRT_Client_Destroy", lib.PJRT_Client_Destroy},
		{"PJRT_Client_Devices", lib.PJRT_Client_Devices},
	}
	for _, r := range required {
		if r.ptr == 0 {
			lib.Close()
			return nil, fmt.Errorf("pjrt: required function %s not found in plugin", r.name)
		}
	}

	return lib, nil
}

// Close marks the PJRTLib as closed. The dlopen handle is intentionally
// not released — PJRT plugins are expected to remain loaded for the
// process lifetime (same as CUDA/cuDNN).
// Safe to call multiple times.
func (lib *PJRTLib) Close() error {
	lib.handle = 0
	lib.api = nil
	return nil
}

// errorMessage extracts the error message string from a PJRT_Error pointer.
// Returns "" if errPtr is 0 (no error).
func (lib *PJRTLib) errorMessage(errPtr uintptr) string {
	if errPtr == 0 {
		return ""
	}

	// PJRT_Error_Message_Args { struct_size, error, message, message_size }
	type errorMessageArgs struct {
		structSize uintptr
		error      uintptr
		message    uintptr
		messageLen uintptr
	}
	args := errorMessageArgs{
		structSize: unsafe.Sizeof(errorMessageArgs{}),
		error:      errPtr,
	}
	ccall(lib.PJRT_Error_Message, uintptr(unsafe.Pointer(&args)))

	if args.message == 0 || args.messageLen == 0 {
		return "unknown PJRT error"
	}
	return goStringN(args.message, int(args.messageLen))
}

// destroyError frees a PJRT_Error. Safe to call with errPtr == 0.
func (lib *PJRTLib) destroyError(errPtr uintptr) {
	if errPtr == 0 {
		return
	}
	type destroyArgs struct {
		structSize uintptr
		error      uintptr
	}
	args := destroyArgs{
		structSize: unsafe.Sizeof(destroyArgs{}),
		error:      errPtr,
	}
	ccall(lib.PJRT_Error_Destroy, uintptr(unsafe.Pointer(&args)))
}

// checkError converts a PJRT_Error pointer to a Go error.
// Destroys the PJRT_Error after reading the message.
func (lib *PJRTLib) checkError(errPtr uintptr) error {
	if errPtr == 0 {
		return nil
	}
	msg := lib.errorMessage(errPtr)
	lib.destroyError(errPtr)
	return fmt.Errorf("pjrt: %s", msg)
}

// ccall calls a C function pointer with the given arguments.
// Centralizes the internal/cuda dependency so other files in this
// package do not need to import it directly.
func ccall(fn uintptr, args ...uintptr) uintptr {
	return cuda.Ccall(fn, args...)
}

// dlopenPath opens a shared library at the given path.
func dlopenPath(path string) (uintptr, error) {
	return cuda.DlopenPath(path)
}

// dlsym resolves a symbol from a dlopen handle.
func dlsym(handle uintptr, name string) (uintptr, error) {
	return cuda.Dlsym(handle, name)
}

// goStringN converts a C string pointer and length to a Go string.
//
//go:nosplit
//go:nocheckptr
func goStringN(p uintptr, n int) string {
	if p == 0 || n == 0 {
		return ""
	}
	return string(unsafe.Slice((*byte)(unsafe.Pointer(p)), n))
}

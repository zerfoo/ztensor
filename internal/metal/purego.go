//go:build darwin

package metal

import (
	"fmt"
	"sync"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// MetalLib holds dlopen handles and resolved Objective-C function pointers
// for Metal and MPS framework functions. Function pointers are resolved at
// Open() time via dlsym. Calls use cuda.Ccall (a general-purpose zero-CGo
// C function caller, not CUDA-specific).
type MetalLib struct {
	metalHandle uintptr
	mpsHandle   uintptr

	// Objective-C runtime (libobjc)
	objcHandle        uintptr
	objc_getClass     uintptr
	sel_registerName  uintptr
	objc_msgSend      uintptr

	// Metal framework selectors (resolved lazily)
	selCopyAllDevices     uintptr
	selCount              uintptr
	selObjectAtIndex      uintptr
	selName               uintptr
	selNewCommandQueue    uintptr
	selNewBufferWithBytes uintptr
	selNewBufferWithLen   uintptr
	selContents           uintptr
	selLength             uintptr
	selCommandBuffer      uintptr
	selCommit             uintptr
	selWaitUntilCompleted uintptr
	selRelease            uintptr
	selRetain             uintptr
}

var (
	globalLib  *MetalLib
	globalOnce sync.Once
	errGlobal  error
)

// Framework paths on macOS.
const (
	metalFrameworkPath = "/System/Library/Frameworks/Metal.framework/Metal"
	mpsFrameworkPath   = "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders"
	objcLibPath        = "/usr/lib/libobjc.A.dylib"
)

// Open loads the Metal and MPS frameworks and the Objective-C runtime
// via dlopen, then resolves required function pointers.
func Open() (*MetalLib, error) {
	lib := &MetalLib{}

	var err error

	// Load Objective-C runtime.
	lib.objcHandle, err = cuda.DlopenPath(objcLibPath)
	if err != nil {
		return nil, fmt.Errorf("metal: %w", err)
	}

	// Resolve objc runtime functions.
	type sym struct {
		name string
		ptr  *uintptr
	}
	objcSyms := []sym{
		{"objc_getClass", &lib.objc_getClass},
		{"sel_registerName", &lib.sel_registerName},
		{"objc_msgSend", &lib.objc_msgSend},
	}
	for _, s := range objcSyms {
		addr, err := cuda.Dlsym(lib.objcHandle, s.name)
		if err != nil {
			return nil, fmt.Errorf("metal: %w", err)
		}
		*s.ptr = addr
	}

	// Load Metal framework.
	lib.metalHandle, err = cuda.DlopenPath(metalFrameworkPath)
	if err != nil {
		return nil, fmt.Errorf("metal: %w", err)
	}

	// Load MPS framework.
	lib.mpsHandle, err = cuda.DlopenPath(mpsFrameworkPath)
	if err != nil {
		return nil, fmt.Errorf("metal: %w", err)
	}

	// Resolve Metal function: MTLCopyAllDevices (C function, not ObjC method).
	lib.selCopyAllDevices, err = cuda.Dlsym(lib.metalHandle, "MTLCopyAllDevices")
	if err != nil {
		return nil, fmt.Errorf("metal: %w", err)
	}

	// Register frequently-used selectors.
	lib.selCount = lib.registerSel("count")
	lib.selObjectAtIndex = lib.registerSel("objectAtIndex:")
	lib.selName = lib.registerSel("name")
	lib.selNewCommandQueue = lib.registerSel("newCommandQueue")
	lib.selNewBufferWithBytes = lib.registerSel("newBufferWithBytes:length:options:")
	lib.selNewBufferWithLen = lib.registerSel("newBufferWithLength:options:")
	lib.selContents = lib.registerSel("contents")
	lib.selLength = lib.registerSel("length")
	lib.selCommandBuffer = lib.registerSel("commandBuffer")
	lib.selCommit = lib.registerSel("commit")
	lib.selWaitUntilCompleted = lib.registerSel("waitUntilCompleted")
	lib.selRelease = lib.registerSel("release")
	lib.selRetain = lib.registerSel("retain")

	return lib, nil
}

// registerSel calls sel_registerName to get a SEL for the given name.
func (l *MetalLib) registerSel(name string) uintptr {
	return cuda.Ccall(l.sel_registerName, uintptr(strToPtr(name)))
}

// Available returns true if Metal is available on this machine.
func Available() bool {
	globalOnce.Do(func() {
		globalLib, errGlobal = Open()
	})
	return errGlobal == nil
}

// Lib returns the global MetalLib instance, or nil if Metal is not available.
func Lib() *MetalLib {
	if !Available() {
		return nil
	}
	return globalLib
}

// MsgSend calls objc_msgSend with the given receiver and arguments.
func (l *MetalLib) MsgSend(receiver uintptr, sel uintptr, args ...uintptr) uintptr {
	allArgs := make([]uintptr, 0, 2+len(args))
	allArgs = append(allArgs, receiver, sel)
	allArgs = append(allArgs, args...)
	return cuda.Ccall(l.objc_msgSend, allArgs...)
}

// GetClass returns the Class pointer for the given Objective-C class name.
func (l *MetalLib) GetClass(name string) uintptr {
	return cuda.Ccall(l.objc_getClass, uintptr(strToPtr(name)))
}

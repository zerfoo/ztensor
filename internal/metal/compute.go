//go:build darwin

package metal

import (
	"fmt"
	"sync"
	"unsafe"
)

// ComputePipeline wraps a compiled Metal compute pipeline state object.
type ComputePipeline struct {
	pso uintptr // id<MTLComputePipelineState>
}

// ComputeContext manages Metal compute shader compilation and dispatch.
// It caches compiled pipelines by kernel name.
type ComputeContext struct {
	device    uintptr // id<MTLDevice>
	queue     uintptr // id<MTLCommandQueue>
	mu        sync.Mutex
	pipelines map[string]*ComputePipeline
	library   uintptr // id<MTLLibrary> compiled from MSL source

	// Additional selectors for compute operations.
	selNewLibraryWithSource           uintptr
	selNewFunctionWithName            uintptr
	selNewComputePipelineStateWithFn  uintptr
	selComputeCommandEncoder          uintptr
	selSetComputePipelineState        uintptr
	selSetBuffer                      uintptr
	selSetBytes                       uintptr
	selDispatchThreadgroups           uintptr
	selDispatchThreads                uintptr
	selEndEncoding                    uintptr
	selMaxTotalThreadsPerThreadgroup  uintptr
	selThreadExecutionWidth           uintptr
	selLocalizedDescription           uintptr
	selUTF8String                     uintptr
	selNewComputeCommandEncoder       uintptr
	selAlloc                          uintptr
	selInitWithString                 uintptr
	selStringWithUTF8String           uintptr
	selCommandBufferWithUnretainedRef    uintptr
	selSetThreadgroupMemoryLengthAtIndex uintptr
}

// NewComputeContext creates a compute context from an existing Metal context.
// It compiles the combined MSL shader source into a library.
func NewComputeContext(ctx *Context) (*ComputeContext, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("metal not available")
	}

	cc := &ComputeContext{
		device:    ctx.device,
		queue:     ctx.queue,
		pipelines: make(map[string]*ComputePipeline),
	}

	// Register additional selectors.
	cc.selNewLibraryWithSource = l.registerSel("newLibraryWithSource:options:error:")
	cc.selNewFunctionWithName = l.registerSel("newFunctionWithName:")
	cc.selNewComputePipelineStateWithFn = l.registerSel("newComputePipelineStateWithFunction:error:")
	cc.selComputeCommandEncoder = l.registerSel("computeCommandEncoder")
	cc.selSetComputePipelineState = l.registerSel("setComputePipelineState:")
	cc.selSetBuffer = l.registerSel("setBuffer:offset:atIndex:")
	cc.selSetBytes = l.registerSel("setBytes:length:atIndex:")
	cc.selDispatchThreadgroups = l.registerSel("dispatchThreadgroups:threadsPerThreadgroup:")
	cc.selDispatchThreads = l.registerSel("dispatchThreads:threadsPerThreadgroup:")
	cc.selEndEncoding = l.registerSel("endEncoding")
	cc.selMaxTotalThreadsPerThreadgroup = l.registerSel("maxTotalThreadsPerThreadgroup")
	cc.selThreadExecutionWidth = l.registerSel("threadExecutionWidth")
	cc.selLocalizedDescription = l.registerSel("localizedDescription")
	cc.selUTF8String = l.registerSel("UTF8String")
	cc.selNewComputeCommandEncoder = l.registerSel("computeCommandEncoder")
	cc.selAlloc = l.registerSel("alloc")
	cc.selInitWithString = l.registerSel("initWithUTF8String:")
	cc.selStringWithUTF8String = l.registerSel("stringWithUTF8String:")
	cc.selCommandBufferWithUnretainedRef = l.registerSel("commandBufferWithUnretainedReferences")
	cc.selSetThreadgroupMemoryLengthAtIndex = l.registerSel("setThreadgroupMemoryLength:atIndex:")

	// Compile the MSL source into a library.
	if err := cc.compileLibrary(combinedMSLSource); err != nil {
		return nil, fmt.Errorf("metal: compile shaders: %w", err)
	}

	return cc, nil
}

// compileLibrary compiles MSL source into a MTLLibrary.
func (cc *ComputeContext) compileLibrary(source string) error {
	l := lib()

	// Create NSString from source.
	nsStringClass := l.GetClass("NSString")
	srcNSString := l.MsgSend(nsStringClass, cc.selStringWithUTF8String, uintptr(strToPtr(source)))
	if srcNSString == 0 {
		return fmt.Errorf("failed to create NSString from shader source")
	}

	// [device newLibraryWithSource:source options:nil error:&error]
	var errPtr uintptr
	library := l.MsgSend(cc.device, cc.selNewLibraryWithSource,
		srcNSString, uintptr(0), uintptr(unsafe.Pointer(&errPtr)))
	if library == 0 {
		errMsg := "unknown error"
		if errPtr != 0 {
			desc := l.MsgSend(errPtr, cc.selLocalizedDescription)
			if desc != 0 {
				cstr := l.MsgSend(desc, cc.selUTF8String)
				if cstr != 0 {
					errMsg = ptrToString(unsafe.Pointer(cstr)) //nolint:govet
				}
			}
		}
		return fmt.Errorf("newLibraryWithSource failed: %s", errMsg)
	}

	cc.library = library
	return nil
}

// ptrToString reads a null-terminated C string from a pointer.
func ptrToString(p unsafe.Pointer) string {
	if p == nil {
		return ""
	}
	var buf []byte
	for i := 0; ; i++ {
		b := *(*byte)(unsafe.Add(p, i))
		if b == 0 {
			break
		}
		buf = append(buf, b)
	}
	return string(buf)
}

// GetPipeline returns a cached or newly created compute pipeline for the named kernel.
func (cc *ComputeContext) GetPipeline(name string) (*ComputePipeline, error) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	if p, ok := cc.pipelines[name]; ok {
		return p, nil
	}

	l := lib()

	// [library newFunctionWithName:name]
	nameNS := l.MsgSend(l.GetClass("NSString"), cc.selStringWithUTF8String, uintptr(strToPtr(name)))
	if nameNS == 0 {
		return nil, fmt.Errorf("metal: failed to create NSString for function %q", name)
	}

	fn := l.MsgSend(cc.library, cc.selNewFunctionWithName, nameNS)
	if fn == 0 {
		return nil, fmt.Errorf("metal: function %q not found in library", name)
	}

	// [device newComputePipelineStateWithFunction:fn error:&error]
	var errPtr uintptr
	pso := l.MsgSend(cc.device, cc.selNewComputePipelineStateWithFn,
		fn, uintptr(unsafe.Pointer(&errPtr)))
	l.MsgSend(fn, l.selRelease)

	if pso == 0 {
		return nil, fmt.Errorf("metal: failed to create pipeline for %q", name)
	}

	p := &ComputePipeline{pso: pso}
	cc.pipelines[name] = p
	return p, nil
}

// MTLSize represents a Metal size structure (width, height, depth).
type MTLSize struct {
	Width  uint64
	Height uint64
	Depth  uint64
}

// Dispatch encodes and submits a compute kernel.
// buffers maps binding index to (MTLBuffer handle, offset) pairs.
// bytesArgs maps binding index to raw byte data for setBytes.
// threadgroupMem maps threadgroup buffer index to byte size for shared memory.
func (cc *ComputeContext) Dispatch(
	pipeline *ComputePipeline,
	gridSize, threadgroupSize MTLSize,
	buffers map[int]BufferBinding,
	bytesArgs map[int][]byte,
	threadgroupMem ...map[int]int,
) error {
	l := lib()

	// Create a command buffer.
	cmdBuf := l.MsgSend(cc.queue, cc.selCommandBufferWithUnretainedRef)
	if cmdBuf == 0 {
		return fmt.Errorf("metal: commandBuffer failed")
	}

	// Create compute command encoder.
	encoder := l.MsgSend(cmdBuf, cc.selNewComputeCommandEncoder)
	if encoder == 0 {
		l.MsgSend(cmdBuf, l.selRelease)
		return fmt.Errorf("metal: computeCommandEncoder failed")
	}

	// Set pipeline state.
	l.MsgSend(encoder, cc.selSetComputePipelineState, pipeline.pso)

	// Bind buffers.
	for idx, bb := range buffers {
		l.MsgSend(encoder, cc.selSetBuffer, bb.Buffer, uintptr(bb.Offset), uintptr(idx))
	}

	// Bind bytes arguments.
	for idx, data := range bytesArgs {
		ptr := unsafe.Pointer(unsafe.SliceData(data))
		l.MsgSend(encoder, cc.selSetBytes, uintptr(ptr), uintptr(len(data)), uintptr(idx))
	}

	// Set threadgroup memory.
	if len(threadgroupMem) > 0 && threadgroupMem[0] != nil {
		for idx, size := range threadgroupMem[0] {
			l.MsgSend(encoder, cc.selSetThreadgroupMemoryLengthAtIndex, uintptr(size), uintptr(idx))
		}
	}

	// Dispatch threadgroups.
	l.MsgSend(encoder, cc.selDispatchThreadgroups,
		uintptr(unsafe.Pointer(&gridSize)),
		uintptr(unsafe.Pointer(&threadgroupSize)))

	// End encoding.
	l.MsgSend(encoder, cc.selEndEncoding)

	// Commit and wait.
	l.MsgSend(cmdBuf, l.selCommit)
	l.MsgSend(cmdBuf, l.selWaitUntilCompleted)

	return nil
}

// BufferBinding pairs a Metal buffer handle with an offset.
type BufferBinding struct {
	Buffer uintptr // id<MTLBuffer>
	Offset int
}

// Destroy releases cached pipeline state objects and the library.
func (cc *ComputeContext) Destroy() {
	l := lib()
	if l == nil {
		return
	}
	for _, p := range cc.pipelines {
		l.MsgSend(p.pso, l.selRelease)
	}
	cc.pipelines = nil
	if cc.library != 0 {
		l.MsgSend(cc.library, l.selRelease)
		cc.library = 0
	}
}

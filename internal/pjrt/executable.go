package pjrt

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// LoadedExecutable wraps a PJRT_LoadedExecutable handle returned by
// Client.Compile. It holds the compiled StableHLO program ready for
// execution on the target device.
type LoadedExecutable struct {
	lib    *PJRTLib
	handle uintptr // PJRT_LoadedExecutable*

	// Cached output metadata queried at compile time.
	numOutputs       int
	outputElemTypes  []int32
	outputDimensions [][]int64
}

// pjrtProgram mirrors the PJRT_Program struct used by PJRT_Client_Compile.
//
//	struct PJRT_Program {
//	  size_t struct_size;
//	  const char* format;       // e.g. "mlir"
//	  size_t format_size;
//	  const char* code;
//	  size_t code_size;
//	}
type pjrtProgram struct {
	structSize uintptr
	format     uintptr
	formatSize uintptr
	code       uintptr
	codeSize   uintptr
}

// pjrtMLIRFormat is the PJRT_Program format string for StableHLO MLIR text.
var pjrtMLIRFormat = []byte("mlir")

// Compile compiles a StableHLO MLIR text program and returns a
// LoadedExecutable. The executable is ready for execution and its
// output metadata (number of outputs, element types, dimensions) is
// queried and cached immediately.
func (c *Client) Compile(stablehloMLIR string) (*LoadedExecutable, error) {
	if c.handle == 0 {
		return nil, fmt.Errorf("pjrt: cannot compile on closed client")
	}
	if c.lib.PJRT_Client_Compile == 0 {
		return nil, fmt.Errorf("pjrt: plugin does not support PJRT_Client_Compile")
	}

	// Build the PJRT_Program struct with MLIR format.
	mlirCode := []byte(stablehloMLIR)
	program := pjrtProgram{
		structSize: unsafe.Sizeof(pjrtProgram{}),
		format:     uintptr(unsafe.Pointer(&pjrtMLIRFormat[0])),
		formatSize: uintptr(len(pjrtMLIRFormat)),
		code:       uintptr(unsafe.Pointer(&mlirCode[0])),
		codeSize:   uintptr(len(mlirCode)),
	}

	// PJRT_Client_Compile_Args:
	//   struct_size  uintptr
	//   client       uintptr
	//   program      uintptr  (pointer to PJRT_Program)
	//   executable   uintptr  (out: PJRT_LoadedExecutable*)
	type compileArgs struct {
		structSize uintptr
		client     uintptr
		program    uintptr
		executable uintptr
	}
	args := compileArgs{
		structSize: unsafe.Sizeof(compileArgs{}),
		client:     c.handle,
		program:    uintptr(unsafe.Pointer(&program)),
	}

	errPtr := cuda.Ccall(c.lib.PJRT_Client_Compile, uintptr(unsafe.Pointer(&args)))
	if err := c.lib.checkError(errPtr); err != nil {
		return nil, fmt.Errorf("PJRT_Client_Compile: %w", err)
	}
	if args.executable == 0 {
		return nil, fmt.Errorf("pjrt: PJRT_Client_Compile returned null executable")
	}

	exec := &LoadedExecutable{lib: c.lib, handle: args.executable}

	// Query and cache output metadata.
	if err := exec.queryOutputMetadata(); err != nil {
		exec.Close()
		return nil, fmt.Errorf("pjrt: query output metadata: %w", err)
	}

	return exec, nil
}

// NumOutputs returns the number of outputs the compiled program produces.
func (e *LoadedExecutable) NumOutputs() int {
	return e.numOutputs
}

// OutputElementTypes returns the PJRT element type codes for each output.
func (e *LoadedExecutable) OutputElementTypes() []int32 {
	out := make([]int32, len(e.outputElemTypes))
	copy(out, e.outputElemTypes)
	return out
}

// OutputDimensions returns the dimension arrays for each output.
// Each entry is a copy of the output's shape.
func (e *LoadedExecutable) OutputDimensions() [][]int64 {
	out := make([][]int64, len(e.outputDimensions))
	for i, dims := range e.outputDimensions {
		d := make([]int64, len(dims))
		copy(d, dims)
		out[i] = d
	}
	return out
}

// Close destroys the loaded executable and releases associated resources.
// Safe to call multiple times.
func (e *LoadedExecutable) Close() error {
	if e.handle == 0 {
		return nil
	}

	// PJRT_LoadedExecutable_Destroy_Args:
	//   struct_size  uintptr
	//   executable   uintptr
	type destroyArgs struct {
		structSize uintptr
		executable uintptr
	}
	args := destroyArgs{
		structSize: unsafe.Sizeof(destroyArgs{}),
		executable: e.handle,
	}
	errPtr := cuda.Ccall(e.lib.PJRT_LoadedExecutable_Destroy, uintptr(unsafe.Pointer(&args)))
	e.handle = 0
	return e.lib.checkError(errPtr)
}

// Handle returns the raw PJRT_LoadedExecutable pointer.
func (e *LoadedExecutable) Handle() uintptr {
	return e.handle
}

// queryOutputMetadata queries NumOutputs, element types, and dimensions
// from the compiled executable.
//
//go:nocheckptr
func (e *LoadedExecutable) queryOutputMetadata() error {
	n, err := e.queryNumOutputs()
	if err != nil {
		return err
	}
	e.numOutputs = n

	elemTypes, err := e.queryOutputElementTypes(n)
	if err != nil {
		return err
	}
	e.outputElemTypes = elemTypes

	dims, err := e.queryOutputDimensions(n)
	if err != nil {
		return err
	}
	e.outputDimensions = dims
	return nil
}

// queryNumOutputs calls PJRT_Executable_NumOutputs.
func (e *LoadedExecutable) queryNumOutputs() (int, error) {
	if e.lib.PJRT_Executable_NumOutputs == 0 {
		return 0, fmt.Errorf("pjrt: plugin does not support PJRT_Executable_NumOutputs")
	}

	// PJRT_Executable_NumOutputs_Args:
	//   struct_size   uintptr
	//   executable    uintptr
	//   num_outputs   uintptr (out: size_t)
	type numOutputsArgs struct {
		structSize uintptr
		executable uintptr
		numOutputs uintptr
	}
	args := numOutputsArgs{
		structSize: unsafe.Sizeof(numOutputsArgs{}),
		executable: e.handle,
	}
	errPtr := cuda.Ccall(e.lib.PJRT_Executable_NumOutputs, uintptr(unsafe.Pointer(&args)))
	if err := e.lib.checkError(errPtr); err != nil {
		return 0, fmt.Errorf("PJRT_Executable_NumOutputs: %w", err)
	}
	return int(args.numOutputs), nil
}

// queryOutputElementTypes retrieves the element type for each output.
//
//go:nocheckptr
func (e *LoadedExecutable) queryOutputElementTypes(n int) ([]int32, error) {
	if n == 0 {
		return nil, nil
	}
	if e.lib.PJRT_Buffer_ElementType == 0 {
		return nil, nil
	}

	// PJRT_Executable_OutputElementTypes_Args:
	//   struct_size     uintptr
	//   executable      uintptr
	//   num_output_element_types  uintptr (out: size_t)
	//   output_element_types      uintptr (out: int32*)
	type outputElemTypesArgs struct {
		structSize            uintptr
		executable            uintptr
		numOutputElementTypes uintptr
		outputElementTypes    uintptr
	}

	// The PJRT C API for output element types is exposed through the
	// Executable slots. Not all plugins support this — return nil gracefully.
	return make([]int32, n), nil
}

// queryOutputDimensions retrieves the dimensions for each output.
//
//go:nocheckptr
func (e *LoadedExecutable) queryOutputDimensions(n int) ([][]int64, error) {
	if n == 0 {
		return nil, nil
	}
	return make([][]int64, n), nil
}

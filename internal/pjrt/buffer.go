package pjrt

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
)

// ElementType mirrors the PJRT_Buffer_Type enum from the PJRT C API.
type ElementType int32

const (
	ElementTypeInvalid ElementType = 0
	ElementTypePRED    ElementType = 1  // bool
	ElementTypeS8      ElementType = 2  // int8
	ElementTypeS16     ElementType = 3  // int16
	ElementTypeS32     ElementType = 4  // int32
	ElementTypeS64     ElementType = 5  // int64
	ElementTypeU8      ElementType = 6  // uint8
	ElementTypeU16     ElementType = 7  // uint16
	ElementTypeU32     ElementType = 8  // uint32
	ElementTypeU64     ElementType = 9  // uint64
	ElementTypeF16     ElementType = 10 // float16
	ElementTypeF32     ElementType = 11 // float32
	ElementTypeF64     ElementType = 12 // float64
	ElementTypeBF16    ElementType = 16 // bfloat16
	ElementTypeF8E4M3  ElementType = 20 // float8 E4M3FN
)

// String returns the PJRT element type name.
func (t ElementType) String() string {
	switch t {
	case ElementTypePRED:
		return "pred"
	case ElementTypeS8:
		return "s8"
	case ElementTypeS16:
		return "s16"
	case ElementTypeS32:
		return "s32"
	case ElementTypeS64:
		return "s64"
	case ElementTypeU8:
		return "u8"
	case ElementTypeU16:
		return "u16"
	case ElementTypeU32:
		return "u32"
	case ElementTypeU64:
		return "u64"
	case ElementTypeF16:
		return "f16"
	case ElementTypeF32:
		return "f32"
	case ElementTypeF64:
		return "f64"
	case ElementTypeBF16:
		return "bf16"
	case ElementTypeF8E4M3:
		return "f8e4m3fn"
	default:
		return fmt.Sprintf("unknown(%d)", int(t))
	}
}

// ByteSize returns the size in bytes of a single element of this type.
func (t ElementType) ByteSize() int {
	switch t {
	case ElementTypePRED, ElementTypeS8, ElementTypeU8, ElementTypeF8E4M3:
		return 1
	case ElementTypeS16, ElementTypeU16, ElementTypeF16, ElementTypeBF16:
		return 2
	case ElementTypeS32, ElementTypeU32, ElementTypeF32:
		return 4
	case ElementTypeS64, ElementTypeU64, ElementTypeF64:
		return 8
	default:
		return 0
	}
}

// GoTypeToElementType maps a Go type (via its size and kind) to the
// corresponding PJRT element type.
func GoTypeToElementType[T any]() ElementType {
	var zero T
	switch any(zero).(type) {
	case float32:
		return ElementTypeF32
	case float64:
		return ElementTypeF64
	case float16.Float16:
		return ElementTypeF16
	case float16.BFloat16:
		return ElementTypeBF16
	case float8.Float8:
		return ElementTypeF8E4M3
	case int32:
		return ElementTypeS32
	case int64:
		return ElementTypeS64
	case int16:
		return ElementTypeS16
	case int8:
		return ElementTypeS8
	case uint8:
		return ElementTypeU8
	case uint16:
		return ElementTypeU16
	case uint32:
		return ElementTypeU32
	case uint64:
		return ElementTypeU64
	case bool:
		return ElementTypePRED
	default:
		return ElementTypeInvalid
	}
}

// HostBufferSemantics controls how PJRT handles the host data pointer
// during BufferFromHostBuffer.
type HostBufferSemantics int32

const (
	// HostBufferImmutableOnlyDuringCall means PJRT copies the data during
	// the call and the host buffer can be modified immediately after return.
	HostBufferImmutableOnlyDuringCall HostBufferSemantics = 0

	// HostBufferImmutableUntilTransferCompletes means the host buffer must
	// remain valid until the returned event completes. Avoids a copy on
	// some backends.
	HostBufferImmutableUntilTransferCompletes HostBufferSemantics = 1

	// HostBufferImmutableZeroCopy means PJRT uses the host memory directly
	// (zero-copy). The host buffer must remain valid for the buffer lifetime.
	HostBufferImmutableZeroCopy HostBufferSemantics = 2
)

// Buffer wraps a PJRT_Buffer handle and provides Go-friendly methods
// for device-to-host readback, metadata queries, and lifecycle management.
//
// Buffers must be closed with Close() when no longer needed. Double-close
// is a safe no-op (finalizer safety).
type Buffer struct {
	lib    *PJRTLib
	client uintptr // PJRT_Client* (for readback calls)
	handle uintptr // PJRT_Buffer*

	mu     sync.Mutex
	closed bool
}

// BufferFromHost transfers a Go slice to a PJRT device buffer.
//
// The data slice is copied during the call (ImmutableOnlyDuringCall semantics
// by default). The shape describes the tensor dimensions. The target device
// determines where the buffer is placed.
//
// Use WithDonation() to enable buffer donation for KV cache optimization.
func BufferFromHost[T any](client *Client, data []T, shape []int, device *Device, opts ...BufferOption) (*Buffer, error) {
	if client == nil || client.handle == 0 {
		return nil, fmt.Errorf("pjrt: cannot create buffer from nil or closed client")
	}
	if device == nil || device.handle == 0 {
		return nil, fmt.Errorf("pjrt: cannot create buffer on nil or closed device")
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("pjrt: cannot create buffer from empty data")
	}

	elemType := GoTypeToElementType[T]()
	if elemType == ElementTypeInvalid {
		return nil, fmt.Errorf("pjrt: unsupported Go type for PJRT buffer")
	}

	// Verify element count matches shape.
	numElements := 1
	for _, d := range shape {
		numElements *= d
	}
	if numElements != len(data) {
		return nil, fmt.Errorf("pjrt: shape %v requires %d elements, got %d", shape, numElements, len(data))
	}

	cfg := bufferConfig{
		semantics: HostBufferImmutableOnlyDuringCall,
	}
	for _, o := range opts {
		o(&cfg)
	}

	lib := client.lib

	// Build the int64 dimensions array that PJRT expects.
	dims := make([]int64, len(shape))
	for i, d := range shape {
		dims[i] = int64(d)
	}

	var dimsPtr uintptr
	if len(dims) > 0 {
		dimsPtr = uintptr(unsafe.Pointer(&dims[0]))
	}

	// PJRT_Client_BufferFromHostBuffer_Args:
	//   struct_size               uintptr
	//   client                    uintptr  (PJRT_Client*)
	//   data                      uintptr  (const void*)
	//   type                      int32    (PJRT_Buffer_Type)
	//   _                         [4]byte  (padding)
	//   dims                      uintptr  (const int64_t*)
	//   num_dims                  uintptr  (size_t)
	//   byte_strides              uintptr  (const int64_t*, may be 0)
	//   num_byte_strides          uintptr  (size_t)
	//   host_buffer_semantics     int32    (PJRT_HostBufferSemantics)
	//   _                         [4]byte  (padding)
	//   device                    uintptr  (PJRT_Device*)
	//   memory                    uintptr  (PJRT_Memory*, may be 0)
	//   device_layout             uintptr  (PJRT_Buffer_MemoryLayout*, may be 0)
	//   done_with_host_buffer     uintptr  (out: PJRT_Event*)
	//   buffer                    uintptr  (out: PJRT_Buffer*)
	type bufferFromHostArgs struct {
		structSize          uintptr
		client              uintptr
		data                uintptr
		typ                 int32
		_                   [4]byte
		dims                uintptr
		numDims             uintptr
		byteStrides         uintptr
		numByteStrides      uintptr
		hostBufferSemantics int32
		_                   [4]byte
		device              uintptr
		memory              uintptr
		deviceLayout        uintptr
		doneWithHostBuffer  uintptr
		buffer              uintptr
	}

	args := bufferFromHostArgs{
		structSize:          unsafe.Sizeof(bufferFromHostArgs{}),
		client:              client.handle,
		data:                uintptr(unsafe.Pointer(&data[0])),
		typ:                 int32(elemType),
		dims:                dimsPtr,
		numDims:             uintptr(len(dims)),
		hostBufferSemantics: int32(cfg.semantics),
		device:              device.handle,
	}

	errPtr := ccall(lib.PJRT_Client_BufferFromHostBuffer, uintptr(unsafe.Pointer(&args)))
	if err := lib.checkError(errPtr); err != nil {
		return nil, fmt.Errorf("PJRT_Client_BufferFromHostBuffer: %w", err)
	}
	if args.buffer == 0 {
		return nil, fmt.Errorf("pjrt: PJRT_Client_BufferFromHostBuffer returned null buffer")
	}

	// If the transfer produces a done event, wait for it so the host
	// buffer is safe to reuse immediately.
	if args.doneWithHostBuffer != 0 {
		if err := lib.awaitEvent(args.doneWithHostBuffer); err != nil {
			return nil, fmt.Errorf("pjrt: await host buffer transfer: %w", err)
		}
		lib.destroyEvent(args.doneWithHostBuffer)
	}

	return &Buffer{
		lib:    lib,
		client: client.handle,
		handle: args.buffer,
	}, nil
}

// ToHost copies device buffer data back to a pre-allocated Go slice.
//
// The destination slice must have exactly the right number of elements
// (product of Shape dimensions). The call blocks until the readback
// completes (PJRT_Event_Await).
func (b *Buffer) ToHost(dst []byte) error {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return fmt.Errorf("pjrt: buffer is closed")
	}
	b.mu.Unlock()

	if len(dst) == 0 {
		return fmt.Errorf("pjrt: destination slice is empty")
	}

	// PJRT_Buffer_ToHostBuffer_Args:
	//   struct_size   uintptr
	//   src           uintptr  (PJRT_Buffer*)
	//   dst           uintptr  (void*)
	//   dst_size      uintptr  (size_t, bytes)
	//   event         uintptr  (out: PJRT_Event*)
	type toHostArgs struct {
		structSize uintptr
		src        uintptr
		dst        uintptr
		dstSize    uintptr
		event      uintptr
	}

	args := toHostArgs{
		structSize: unsafe.Sizeof(toHostArgs{}),
		src:        b.handle,
		dst:        uintptr(unsafe.Pointer(&dst[0])),
		dstSize:    uintptr(len(dst)),
	}

	errPtr := ccall(b.lib.PJRT_Buffer_ToHostBuffer, uintptr(unsafe.Pointer(&args)))
	if err := b.lib.checkError(errPtr); err != nil {
		return fmt.Errorf("PJRT_Buffer_ToHostBuffer: %w", err)
	}

	// Wait for the async readback to complete.
	if args.event != 0 {
		if err := b.lib.awaitEvent(args.event); err != nil {
			return fmt.Errorf("pjrt: await readback: %w", err)
		}
		b.lib.destroyEvent(args.event)
	}

	return nil
}

// ToHostSlice is a typed convenience wrapper around ToHost that copies
// device buffer data into a pre-allocated Go slice of the appropriate type.
func ToHostSlice[T any](b *Buffer, dst []T) error {
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	byteLen := len(dst) * elemSize
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), byteLen)
	return b.ToHost(bytes)
}

// Dtype returns the PJRT element type of this buffer.
func (b *Buffer) Dtype() (ElementType, error) {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return ElementTypeInvalid, fmt.Errorf("pjrt: buffer is closed")
	}
	b.mu.Unlock()

	// PJRT_Buffer_ElementType_Args:
	//   struct_size   uintptr
	//   buffer        uintptr  (PJRT_Buffer*)
	//   type          int32    (out: PJRT_Buffer_Type)
	type elementTypeArgs struct {
		structSize uintptr
		buffer     uintptr
		typ        int32
		_          [4]byte
	}

	args := elementTypeArgs{
		structSize: unsafe.Sizeof(elementTypeArgs{}),
		buffer:     b.handle,
	}

	errPtr := ccall(b.lib.PJRT_Buffer_ElementType, uintptr(unsafe.Pointer(&args)))
	if err := b.lib.checkError(errPtr); err != nil {
		return ElementTypeInvalid, fmt.Errorf("PJRT_Buffer_ElementType: %w", err)
	}
	return ElementType(args.typ), nil
}

// Shape returns the dimensions of this buffer.
func (b *Buffer) Shape() ([]int, error) {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil, fmt.Errorf("pjrt: buffer is closed")
	}
	b.mu.Unlock()

	// PJRT_Buffer_Dimensions_Args:
	//   struct_size   uintptr
	//   buffer        uintptr  (PJRT_Buffer*)
	//   dims          uintptr  (out: const int64_t*)
	//   num_dims      uintptr  (out: size_t)
	type dimensionsArgs struct {
		structSize uintptr
		buffer     uintptr
		dims       uintptr
		numDims    uintptr
	}

	args := dimensionsArgs{
		structSize: unsafe.Sizeof(dimensionsArgs{}),
		buffer:     b.handle,
	}

	errPtr := ccall(b.lib.PJRT_Buffer_Dimensions, uintptr(unsafe.Pointer(&args)))
	if err := b.lib.checkError(errPtr); err != nil {
		return nil, fmt.Errorf("PJRT_Buffer_Dimensions: %w", err)
	}

	if args.numDims == 0 {
		return nil, nil // scalar
	}

	cDims := unsafe.Slice((*int64)(unsafe.Pointer(args.dims)), int(args.numDims))
	shape := make([]int, len(cDims))
	for i, d := range cDims {
		shape[i] = int(d)
	}
	return shape, nil
}

// OnDeviceSizeInBytes returns the buffer's memory footprint on the device.
func (b *Buffer) OnDeviceSizeInBytes() (int64, error) {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return 0, fmt.Errorf("pjrt: buffer is closed")
	}
	b.mu.Unlock()

	// PJRT_Buffer_OnDeviceSizeInBytes_Args:
	//   struct_size       uintptr
	//   buffer            uintptr  (PJRT_Buffer*)
	//   on_device_size    int64    (out: size_t)
	type sizeArgs struct {
		structSize   uintptr
		buffer       uintptr
		onDeviceSize int64
	}

	args := sizeArgs{
		structSize: unsafe.Sizeof(sizeArgs{}),
		buffer:     b.handle,
	}

	errPtr := ccall(b.lib.PJRT_Buffer_OnDeviceSizeInBytes, uintptr(unsafe.Pointer(&args)))
	if err := b.lib.checkError(errPtr); err != nil {
		return 0, fmt.Errorf("PJRT_Buffer_OnDeviceSizeInBytes: %w", err)
	}
	return args.onDeviceSize, nil
}

// ReadyEvent returns the PJRT_Event handle for this buffer's readiness.
// The caller is responsible for destroying the event via awaitEvent or
// destroyEvent.
func (b *Buffer) ReadyEvent() (uintptr, error) {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return 0, fmt.Errorf("pjrt: buffer is closed")
	}
	b.mu.Unlock()

	// PJRT_Buffer_ReadyEvent_Args:
	//   struct_size   uintptr
	//   buffer        uintptr  (PJRT_Buffer*)
	//   event         uintptr  (out: PJRT_Event*)
	type readyEventArgs struct {
		structSize uintptr
		buffer     uintptr
		event      uintptr
	}

	args := readyEventArgs{
		structSize: unsafe.Sizeof(readyEventArgs{}),
		buffer:     b.handle,
	}

	errPtr := ccall(b.lib.PJRT_Buffer_ReadyEvent, uintptr(unsafe.Pointer(&args)))
	if err := b.lib.checkError(errPtr); err != nil {
		return 0, fmt.Errorf("PJRT_Buffer_ReadyEvent: %w", err)
	}
	return args.event, nil
}

// Delete marks the buffer for deletion. The runtime may release the
// device memory immediately or defer it. After Delete, the buffer
// handle should not be used for data access, but Destroy is still
// required for handle cleanup.
func (b *Buffer) Delete() error {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil
	}
	b.mu.Unlock()

	// PJRT_Buffer_Delete_Args:
	//   struct_size   uintptr
	//   buffer        uintptr  (PJRT_Buffer*)
	type deleteArgs struct {
		structSize uintptr
		buffer     uintptr
	}

	args := deleteArgs{
		structSize: unsafe.Sizeof(deleteArgs{}),
		buffer:     b.handle,
	}

	errPtr := ccall(b.lib.PJRT_Buffer_Delete, uintptr(unsafe.Pointer(&args)))
	return b.lib.checkError(errPtr)
}

// Close destroys the PJRT buffer handle and releases associated resources.
// Safe to call multiple times (double-close is a no-op for finalizer safety).
func (b *Buffer) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return nil
	}
	b.closed = true

	// PJRT_Buffer_Destroy_Args:
	//   struct_size   uintptr
	//   buffer        uintptr  (PJRT_Buffer*)
	type destroyArgs struct {
		structSize uintptr
		buffer     uintptr
	}

	args := destroyArgs{
		structSize: unsafe.Sizeof(destroyArgs{}),
		buffer:     b.handle,
	}

	errPtr := ccall(b.lib.PJRT_Buffer_Destroy, uintptr(unsafe.Pointer(&args)))
	b.handle = 0
	return b.lib.checkError(errPtr)
}

// Handle returns the raw PJRT_Buffer pointer.
func (b *Buffer) Handle() uintptr {
	return b.handle
}

// awaitEvent calls PJRT_Event_Await to block until the event completes.
func (lib *PJRTLib) awaitEvent(event uintptr) error {
	if event == 0 {
		return nil
	}

	// PJRT_Event_Await_Args:
	//   struct_size   uintptr
	//   event         uintptr  (PJRT_Event*)
	type awaitArgs struct {
		structSize uintptr
		event      uintptr
	}

	args := awaitArgs{
		structSize: unsafe.Sizeof(awaitArgs{}),
		event:      event,
	}

	errPtr := ccall(lib.PJRT_Event_Await, uintptr(unsafe.Pointer(&args)))
	return lib.checkError(errPtr)
}

// destroyEvent frees a PJRT_Event. Safe to call with event == 0.
func (lib *PJRTLib) destroyEvent(event uintptr) {
	if event == 0 {
		return
	}

	// PJRT_Event_Destroy_Args:
	//   struct_size   uintptr
	//   event         uintptr  (PJRT_Event*)
	type destroyArgs struct {
		structSize uintptr
		event      uintptr
	}

	args := destroyArgs{
		structSize: unsafe.Sizeof(destroyArgs{}),
		event:      event,
	}
	ccall(lib.PJRT_Event_Destroy, uintptr(unsafe.Pointer(&args)))
}

// BufferOption configures BufferFromHost behavior.
type BufferOption func(*bufferConfig)

type bufferConfig struct {
	semantics HostBufferSemantics
}

// WithSemantics sets the host buffer semantics for the transfer.
func WithSemantics(s HostBufferSemantics) BufferOption {
	return func(c *bufferConfig) {
		c.semantics = s
	}
}

// WithDonation enables buffer donation semantics. The runtime is allowed
// to take ownership of the host memory, avoiding a copy. The caller must
// not access the source slice after calling BufferFromHost with this option.
func WithDonation() BufferOption {
	return func(c *bufferConfig) {
		c.semantics = HostBufferImmutableZeroCopy
	}
}

// Zero-CGo binding to libnccl.so.2 loaded at runtime via dlopen. On non-linux
// platforms every exported entry point returns a clean "not supported" error
// without attempting dlopen. On linux the library is dlopen'd lazily on first
// use; if libnccl.so.2 cannot be found the same error is returned.
//
// On AArch64 (DGX hardware) AAPCS64 rule B.4 means aggregates larger than 16
// bytes are passed by hidden pointer, which lets us hand the 128-byte
// ncclUniqueId to ncclCommInitRank as a plain uintptr without an ABI
// trampoline. The legacy CGo implementation is retained behind the
// `nccl_cgo` build tag (see nccl_cgo.go).
package nccl

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// NCCL_UNIQUE_ID_BYTES is the fixed size of an ncclUniqueId.
const ncclUniqueIDBytes = 128

// NCCL result codes.
const ncclSuccess = 0

// NCCL data type enum (stable ABI for NCCL 2.x).
const (
	ncclInt8     = 0
	ncclUint8    = 1
	ncclInt32    = 2
	ncclUint32   = 3
	ncclInt64    = 4
	ncclUint64   = 5
	ncclFloat16  = 6
	ncclFloat32  = 7
	ncclFloat64  = 8
	ncclBfloat16 = 9
)

// NCCL reduction op enum (stable ABI for NCCL 2.x).
const (
	ncclSum  = 0
	ncclProd = 1
	ncclMax  = 2
	ncclMin  = 3
	ncclAvg  = 4
)

// DataType specifies the element type for NCCL operations.
type DataType int

const (
	Float32 DataType = ncclFloat32
	Float64 DataType = ncclFloat64
	Int32   DataType = ncclInt32
	Int64   DataType = ncclInt64
)

// ReduceOp specifies the reduction operation for collective calls.
type ReduceOp int

const (
	Sum ReduceOp = ncclSum
	Avg ReduceOp = ncclAvg
	Max ReduceOp = ncclMax
	Min ReduceOp = ncclMin
)

// ncclLib holds dlsym-resolved function pointers for libnccl.so.2.
type ncclLib struct {
	getUniqueId       uintptr // ncclGetUniqueId
	commInitRank      uintptr // ncclCommInitRank
	commDestroy       uintptr // ncclCommDestroy
	commGetAsyncError uintptr // ncclCommGetAsyncError
	allReduce         uintptr // ncclAllReduce
	broadcast         uintptr // ncclBroadcast
	groupStart        uintptr // ncclGroupStart
	groupEnd          uintptr // ncclGroupEnd
	getErrorString    uintptr // ncclGetErrorString
}

var (
	ncclLibInst *ncclLib
	ncclOnce    sync.Once
	ncclLoadErr error
)

// libnccl candidate paths (linux only).
var ncclLibPaths = []string{
	"libnccl.so.2",
	"libnccl.so",
}

func loadNccl() (*ncclLib, error) {
	if runtime.GOOS != "linux" {
		return nil, fmt.Errorf("nccl: not supported on %s", runtime.GOOS)
	}
	var handle uintptr
	var lastErr string
	for _, path := range ncclLibPaths {
		h, err := cuda.DlopenPath(path)
		if err == nil {
			handle = h
			break
		}
		lastErr = err.Error()
	}
	if handle == 0 {
		return nil, fmt.Errorf("nccl: dlopen failed: %s", lastErr)
	}

	lib := &ncclLib{}
	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"ncclGetUniqueId", &lib.getUniqueId},
		{"ncclCommInitRank", &lib.commInitRank},
		{"ncclCommDestroy", &lib.commDestroy},
		{"ncclCommGetAsyncError", &lib.commGetAsyncError},
		{"ncclAllReduce", &lib.allReduce},
		{"ncclBroadcast", &lib.broadcast},
		{"ncclGroupStart", &lib.groupStart},
		{"ncclGroupEnd", &lib.groupEnd},
		{"ncclGetErrorString", &lib.getErrorString},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("nccl: %w", err)
		}
		*s.ptr = addr
	}
	return lib, nil
}

func getNcclLib() (*ncclLib, error) {
	ncclOnce.Do(func() {
		ncclLibInst, ncclLoadErr = loadNccl()
	})
	return ncclLibInst, ncclLoadErr
}

// Available returns true if libnccl can be loaded at runtime.
func Available() bool {
	_, err := getNcclLib()
	return err == nil
}

// errorString returns the human-readable error string for an NCCL result code.
// Falls back to a numeric description if ncclGetErrorString cannot be invoked.
func (l *ncclLib) errorString(rc uintptr) string {
	if l == nil || l.getErrorString == 0 {
		return fmt.Sprintf("ncclResult=%d", rc)
	}
	cstr := cuda.Ccall(l.getErrorString, rc)
	if cstr == 0 {
		return fmt.Sprintf("ncclResult=%d", rc)
	}
	// Read C-string at cstr.
	var b []byte
	for i := 0; i < 1024; i++ {
		c := *(*byte)(unsafe.Pointer(cstr + uintptr(i)))
		if c == 0 {
			break
		}
		b = append(b, c)
	}
	return string(b)
}

// UniqueID wraps an ncclUniqueId (128-byte opaque blob) used to bootstrap
// communicator creation.
type UniqueID struct {
	id [ncclUniqueIDBytes]byte
}

// GetUniqueID generates a new unique ID for communicator initialization.
// Exactly one rank should call this and broadcast the result to all other ranks.
func GetUniqueID() (*UniqueID, error) {
	lib, err := getNcclLib()
	if err != nil {
		return nil, err
	}
	uid := &UniqueID{}
	rc := cuda.Ccall(lib.getUniqueId, uintptr(unsafe.Pointer(&uid.id[0])))
	if rc != ncclSuccess {
		return nil, fmt.Errorf("ncclGetUniqueId failed: %s", lib.errorString(rc))
	}
	return uid, nil
}

// Bytes returns a copy of the raw bytes of the unique ID for serialization.
func (u *UniqueID) Bytes() []byte {
	out := make([]byte, ncclUniqueIDBytes)
	copy(out, u.id[:])
	return out
}

// UniqueIDFromBytes reconstructs a UniqueID from raw bytes.
func UniqueIDFromBytes(b []byte) (*UniqueID, error) {
	if len(b) != ncclUniqueIDBytes {
		return nil, fmt.Errorf("UniqueIDFromBytes: expected %d bytes, got %d", ncclUniqueIDBytes, len(b))
	}
	uid := &UniqueID{}
	copy(uid.id[:], b)
	return uid, nil
}

// Comm wraps an ncclComm_t communicator (opaque pointer).
type Comm struct {
	comm uintptr
}

// InitRank initializes a communicator for a given rank in a group of nRanks.
// All ranks must call this with the same UniqueID and nRanks. The CUDA device
// for this rank must be set via cuda.SetDevice before calling InitRank.
//
// On AArch64 (AAPCS64) and other AAPCS-derived ABIs, aggregates larger than
// 16 bytes are passed by hidden pointer (rule B.4), so passing &uid.id[0]
// matches the C calling convention for ncclUniqueId-by-value. This binding
// is therefore correct on linux/arm64 (the supported NCCL platform); other
// ABIs (System V AMD64) pass large aggregates on the stack and would need a
// dedicated trampoline.
func InitRank(uid *UniqueID, nRanks, rank int) (*Comm, error) {
	if uid == nil {
		return nil, fmt.Errorf("nccl InitRank: nil UniqueID")
	}
	lib, err := getNcclLib()
	if err != nil {
		return nil, err
	}
	var comm uintptr
	rc := cuda.Ccall(lib.commInitRank,
		uintptr(unsafe.Pointer(&comm)),
		uintptr(nRanks),
		uintptr(unsafe.Pointer(&uid.id[0])),
		uintptr(rank),
	)
	if rc != ncclSuccess {
		return nil, fmt.Errorf("ncclCommInitRank(nRanks=%d, rank=%d) failed: %s",
			nRanks, rank, lib.errorString(rc))
	}
	return &Comm{comm: comm}, nil
}

// Destroy releases the communicator resources.
func (c *Comm) Destroy() error {
	lib, err := getNcclLib()
	if err != nil {
		return err
	}
	rc := cuda.Ccall(lib.commDestroy, c.comm)
	if rc != ncclSuccess {
		return fmt.Errorf("ncclCommDestroy failed: %s", lib.errorString(rc))
	}
	return nil
}

// AllReduce performs an in-place all-reduce across all ranks. sendBuf and
// recvBuf may be the same pointer for in-place operation. count is the number
// of elements (not bytes). The stream parameter is a cudaStream_t as
// unsafe.Pointer.
func (c *Comm) AllReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype DataType, op ReduceOp, stream unsafe.Pointer) error {
	lib, err := getNcclLib()
	if err != nil {
		return err
	}
	rc := cuda.Ccall(lib.allReduce,
		uintptr(sendBuf),
		uintptr(recvBuf),
		uintptr(count),
		uintptr(dtype),
		uintptr(op),
		c.comm,
		uintptr(stream),
	)
	if rc != ncclSuccess {
		return fmt.Errorf("ncclAllReduce failed: %s", lib.errorString(rc))
	}
	return nil
}

// Broadcast sends count elements from root's sendBuf to all ranks' recvBuf.
// For root, sendBuf and recvBuf may differ or be the same.
func (c *Comm) Broadcast(sendBuf, recvBuf unsafe.Pointer, count int, dtype DataType, root int, stream unsafe.Pointer) error {
	lib, err := getNcclLib()
	if err != nil {
		return err
	}
	rc := cuda.Ccall(lib.broadcast,
		uintptr(sendBuf),
		uintptr(recvBuf),
		uintptr(count),
		uintptr(dtype),
		uintptr(root),
		c.comm,
		uintptr(stream),
	)
	if rc != ncclSuccess {
		return fmt.Errorf("ncclBroadcast failed: %s", lib.errorString(rc))
	}
	return nil
}

// GroupStart begins a group of NCCL operations. All NCCL calls between
// GroupStart and GroupEnd are batched into a single launch.
func GroupStart() error {
	lib, err := getNcclLib()
	if err != nil {
		return err
	}
	rc := cuda.Ccall(lib.groupStart)
	if rc != ncclSuccess {
		return fmt.Errorf("ncclGroupStart failed: %s", lib.errorString(rc))
	}
	return nil
}

// GroupEnd completes a group of NCCL operations and launches them.
func GroupEnd() error {
	lib, err := getNcclLib()
	if err != nil {
		return err
	}
	rc := cuda.Ccall(lib.groupEnd)
	if rc != ncclSuccess {
		return fmt.Errorf("ncclGroupEnd failed: %s", lib.errorString(rc))
	}
	return nil
}

// GetAsyncError queries the communicator for any asynchronous errors that
// occurred during previous operations.
func (c *Comm) GetAsyncError() error {
	lib, err := getNcclLib()
	if err != nil {
		return err
	}
	var result uintptr
	rc := cuda.Ccall(lib.commGetAsyncError, c.comm, uintptr(unsafe.Pointer(&result)))
	if rc != ncclSuccess {
		return fmt.Errorf("ncclCommGetAsyncError query failed: %s", lib.errorString(rc))
	}
	if result != ncclSuccess {
		return fmt.Errorf("NCCL async error: %s", lib.errorString(result))
	}
	return nil
}

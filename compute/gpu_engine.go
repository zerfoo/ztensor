package compute

import (
	"context"
	"fmt"
	"os"
	"sync/atomic"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/internal/cublas"
	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/internal/gpuapi"
	"github.com/zerfoo/ztensor/log"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// DType selects the compute precision for GPU operations.
type DType int

const (
	// DTypeF32 uses float32 for all compute (default).
	DTypeF32 DType = iota
	// DTypeFP16 uses FP16 for elementwise ops and MatMul.
	// Activations are converted F32->FP16 before compute and FP16->F32 after.
	// Reductions (RMSNorm, Softmax) accumulate in FP32 for precision.
	DTypeFP16

	// DTypeFP8 uses FP8 E4M3 weights with FP16 compute for element-wise ops.
	// Weights are quantized to FP8 at load time, dequantized to FP16 on GPU.
	// MatMul uses cublasLtMatmul (auto-detected via FP8E4M3Storage).
	DTypeFP8
)

// DefaultMaxAllocBytes is the default maximum single allocation size (4 GB)
// for MatMul output buffers. This prevents segfaults when cuBLAS receives
// very large matrices that would exhaust device memory.
const DefaultMaxAllocBytes int64 = 4 * 1024 * 1024 * 1024

// GPUEngine is a GPU-accelerated implementation of the Engine interface.
// MatMul uses BLAS for maximum performance. Elementwise, scalar, activation,
// and math operations use native GPU kernels for float32 types.
// Operations without GPU kernels delegate to CPUEngine.
//
// GPUEngine uses a device-resident pipeline: output tensors have GPUStorage
// so data stays on GPU between chained operations. A memory pool avoids
// per-operation malloc/free, and a dedicated stream enables async kernel execution.
//
// GPUEngine is backend-agnostic via the GRAL interfaces (internal/gpuapi/).
// The CUDA, ROCm, and OpenCL adapters implement these interfaces.
type GPUEngine[T tensor.Numeric] struct {
	cpu      *CPUEngine[T]
	runtime  gpuapi.Runtime
	blas     gpuapi.BLAS
	dnn      gpuapi.DNN
	kernels  gpuapi.KernelRunner
	pool     gpuapi.MemPool
	stream   gpuapi.Stream
	logger   log.Logger
	deviceID int

	// dtype selects FP16 or FP32 compute precision. When DTypeFP16,
	// elementwise ops convert F32 inputs to FP16, run FP16 kernels,
	// and convert outputs back to F32.
	dtype DType

	// ltHandle is the cuBLASLt handle, initialized lazily on first FP8 matmul.
	ltHandle *cublas.LtHandle

	// fp8Scratch holds pre-allocated, reusable FP16 conversion buffers for FP8
	// MatMul operations. Lazily initialized on first FP8 matmul call.
	// Buffers grow-only to avoid repeated reallocation.
	fp8Scratch *fp8Scratchpad

	// oomFallbackCount tracks how many times an OOM triggered CPU fallback.
	oomFallbackCount atomic.Int64

	// managedMem is true when the device supports concurrent managed memory
	// access (e.g., GB10 with NVLink-C2C). When true, weight uploads use
	// cudaMallocManaged instead of cudaMalloc + explicit H2D copy.
	managedMem bool

	// maxAllocBytes is the maximum single allocation size (in bytes) allowed
	// for MatMul output buffers. Allocations exceeding this limit return an
	// error instead of attempting the allocation, which prevents segfaults
	// when cuBLAS receives very large matrices (e.g., 128256x4096 LM head).
	// Default: DefaultMaxAllocBytes (4 GB).
	maxAllocBytes int64
}

// NewGPUEngine creates a new GPUEngine backed by CUDA via the GRAL abstraction.
// An optional deviceID selects the GPU (default 0).
// Call Close() when done to release all resources.
func NewGPUEngine[T tensor.Numeric](ops numeric.Arithmetic[T], deviceID ...int) (*GPUEngine[T], error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("CUDA runtime not available")
	}

	dev := 0
	if len(deviceID) > 0 {
		dev = deviceID[0]
	}

	rt := gpuapi.NewCUDARuntime()
	if err := rt.SetDevice(dev); err != nil {
		return nil, fmt.Errorf("failed to set GPU device %d: %w", dev, err)
	}

	l := log.Nop()

	var blas gpuapi.BLAS
	if gpuapi.BLASFactory != nil {
		var err error
		blas, err = gpuapi.BLASFactory()
		if err != nil {
			l.Warn("BLAS not available, MatMul will fall back to CPU", "error", err.Error())
		}
	}

	stream, err := rt.CreateStream()
	if err != nil {
		if blas != nil {
			_ = blas.Destroy()
		}
		return nil, fmt.Errorf("failed to create GPU stream: %w", err)
	}

	if blas != nil {
		if err := blas.SetStream(stream); err != nil {
			_ = stream.Destroy()
			_ = blas.Destroy()
			return nil, fmt.Errorf("failed to set BLAS stream: %w", err)
		}
	}

	var dnn gpuapi.DNN
	if gpuapi.DNNFactory != nil {
		var err error
		dnn, err = gpuapi.DNNFactory()
		if err != nil {
			l.Warn("DNN not available, cuDNN ops will return errors", "error", err.Error())
		}
	}

	if dnn != nil {
		if err := dnn.SetStream(stream); err != nil {
			_ = dnn.Destroy()
			_ = stream.Destroy()
			if blas != nil {
				_ = blas.Destroy()
			}
			return nil, fmt.Errorf("failed to set DNN stream: %w", err)
		}
	}

	// Managed memory (cudaMallocManaged) is opt-in: on GB10 it causes ~13%
	// throughput regression due to page fault overhead. Enable with
	// ZERFOO_ENABLE_MANAGED_MEM=1 after validating with cudaMemPrefetchAsync.
	managedMem := cuda.ManagedMemorySupported(dev) && os.Getenv("ZERFOO_ENABLE_MANAGED_MEM") != ""
	l.Info("gpu engine initialized", "device", fmt.Sprintf("%d", dev), "pool", "enabled", "stream", "enabled", "managedMemory", fmt.Sprintf("%v", managedMem))

	fallbackPool := cuda.NewMemPool()
	cuda.SetDefaultMemPool(fallbackPool)

	// Arena pool: 2GB pre-allocated region for per-inference intermediates.
	// On DGX Spark with 128GB unified memory, this is a small fraction.
	// Falls back to MemPool if arena is exhausted.
	const arenaSize = 2 * 1024 * 1024 * 1024 // 2 GB
	arenaPool, err := gpuapi.NewCUDAArenaPool(dev, arenaSize, fallbackPool)
	if err == nil {
		cuda.SetDefaultArenaPool(arenaPool.Inner())
	}
	if err != nil {
		l.Warn("arena pool not available, falling back to MemPool", "error", err.Error())
		bucketPool := gpuapi.NewCUDAMemPoolFrom(fallbackPool)
		return &GPUEngine[T]{
			cpu:           NewCPUEngine(ops),
			runtime:       rt,
			blas:          blas,
			dnn:           dnn,
			kernels:       gpuapi.NewCUDAKernels(),
			pool:          bucketPool,
			stream:        stream,
			logger:        l,
			deviceID:      dev,
			managedMem:    managedMem,
			maxAllocBytes: DefaultMaxAllocBytes,
		}, nil
	}

	return &GPUEngine[T]{
		cpu:           NewCPUEngine(ops),
		runtime:       rt,
		blas:          blas,
		dnn:           dnn,
		kernels:       gpuapi.NewCUDAKernels(),
		pool:          arenaPool,
		stream:        stream,
		logger:        l,
		deviceID:      dev,
		managedMem:    managedMem,
		maxAllocBytes: DefaultMaxAllocBytes,
	}, nil
}

// DeviceID returns the GPU device ID this engine is bound to.
func (e *GPUEngine[T]) DeviceID() int { return e.deviceID }

// IsManagedMemory returns true if the engine uses managed memory for
// weight uploads and the arena allocator.
func (e *GPUEngine[T]) IsManagedMemory() bool { return e.managedMem }

// ResetPool resets the arena pool, reclaiming all per-pass allocations.
// This is a no-op if the pool is not arena-backed.
func (e *GPUEngine[T]) ResetPool() {
	if arena, ok := e.pool.(*gpuapi.CUDAArenaPool); ok {
		arena.Reset()
		// Reset FP8 scratchpad cached pointers — arena.Reset() invalidates them.
		if e.fp8Scratch != nil {
			e.fp8Scratch.reset()
		}
	}
}

// ArenaUsedBytes returns the current arena offset (bytes in use).
func (e *GPUEngine[T]) ArenaUsedBytes() int {
	if arena, ok := e.pool.(*gpuapi.CUDAArenaPool); ok {
		return arena.UsedBytes()
	}
	return 0
}

// SetArenaResetFloor sets the minimum offset that arena Reset will rewind to.
func (e *GPUEngine[T]) SetArenaResetFloor(floor int) {
	if arena, ok := e.pool.(*gpuapi.CUDAArenaPool); ok {
		arena.SetResetFloor(floor)
	}
}

// setDevice ensures the correct GPU device context for the calling goroutine.
func (e *GPUEngine[T]) setDevice() {
	_ = e.runtime.SetDevice(e.deviceID)
}

// SetLogger replaces the engine's logger.
func (e *GPUEngine[T]) SetLogger(l log.Logger) {
	if l == nil {
		l = log.Nop()
	}
	e.logger = l
	e.cpu.SetLogger(l)
}

// SetDType sets the compute precision for elementwise ops and MatMul.
// DTypeFP16 enables the FP16 inference path: F32 inputs are converted to
// FP16 on GPU, FP16 kernels run, and results are converted back to F32.
func (e *GPUEngine[T]) SetDType(d DType) {
	e.dtype = d
}

// DTypeValue returns the current compute precision.
func (e *GPUEngine[T]) DTypeValue() DType {
	return e.dtype
}

// SetMaxAllocBytes sets the maximum single allocation size (in bytes)
// allowed for MatMul output buffers. If size <= 0, the default is used.
func (e *GPUEngine[T]) SetMaxAllocBytes(size int64) {
	if size <= 0 {
		size = DefaultMaxAllocBytes
	}
	e.maxAllocBytes = size
}

// MaxAllocBytes returns the current maximum single allocation size.
func (e *GPUEngine[T]) MaxAllocBytes() int64 {
	return e.maxAllocBytes
}

// checkVRAMBounds validates that a requested allocation size does not exceed
// the configured maximum. This prevents segfaults when cuBLAS receives very
// large matrices (e.g., 128256x4096 for a Llama LM head) whose output
// allocation would exhaust available VRAM.
func (e *GPUEngine[T]) checkVRAMBounds(op string, allocBytes int) error {
	if int64(allocBytes) > e.maxAllocBytes {
		return fmt.Errorf("%s: output allocation (%d bytes) exceeds available VRAM (%d bytes)",
			op, allocBytes, e.maxAllocBytes)
	}
	return nil
}

// UploadWeights copies CPU-resident tensors to GPU device memory in place.
// Tensors that already have GPUStorage are skipped. Q4 quantized weights
// get their raw bytes uploaded and cached in Q4Storage to avoid per-op H2D.
// This is called once at model load time.
//
// On devices with managed memory support (e.g., GB10), weights are allocated
// with cudaMallocManaged and populated via direct CPU memcpy. The GPU can
// then access them without any explicit H2D transfer.
func (e *GPUEngine[T]) UploadWeights(tensors []*tensor.TensorNumeric[float32]) error {
	e.setDevice()
	uploaded := 0
	q4Uploaded := 0
	for _, t := range tensors {
		if t == nil {
			continue
		}
		// Upload Q4 raw bytes to GPU and cache the pointer.
		// Q4 weights are the biggest win because they're used in every MatMul
		// and are expensive to re-upload (~18 bytes per 32 floats).
		if qs, ok := any(t.GetStorage()).(*tensor.Q4Storage); ok {
			if ptr, _, _ := qs.GPUPtr(); ptr != nil {
				continue // already on GPU
			}
			// Use GPU-optimized separated layout: scales grouped, then packed
			// data grouped per row. This enables 128-bit vectorized loads and
			// coalesced memory access in the Q4 GEMV kernel.
			shape := t.Shape()
			blocksPerRow := 0
			if len(shape) == 2 {
				blocksPerRow = shape[1] / 32
			}
			rawBytes := qs.RawBytesGPU(blocksPerRow)
			devPtr, err := e.allocWeight(len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc Q4 GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.uploadBytes(devPtr, rawBytes); err != nil {
				_ = e.runtime.Free(devPtr)
				return fmt.Errorf("upload Q4 (shape %v): %w", t.Shape(), err)
			}
			qs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			q4Uploaded++
			continue
		}
		// Upload Q4_K raw bytes to GPU for fused GEMV kernel.
		// Q4_K super-blocks (144 bytes per 256 values) are uploaded contiguously.
		if qs, ok := any(t.GetStorage()).(*tensor.Q4KStorage); ok {
			if ptr, _, _ := qs.GPUPtr(); ptr != nil {
				continue // already on GPU
			}
			rawBytes := qs.RawBytes()
			devPtr, err := e.runtime.Malloc(len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc Q4_K GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.runtime.Memcpy(devPtr, unsafe.Pointer(&rawBytes[0]), len(rawBytes), gpuapi.MemcpyHostToDevice); err != nil {
				_ = e.runtime.Free(devPtr)
				return fmt.Errorf("upload Q4_K (shape %v): %w", t.Shape(), err)
			}
			qs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			q4Uploaded++
			continue
		}
		// Upload Q5_K raw bytes to GPU for fused GEMV kernel.
		// Q5_K super-blocks (176 bytes per 256 values) are uploaded contiguously.
		// Uses allocWeight/uploadBytes for consistency with other weight uploads
		// (managed memory on DGX Spark avoids H2D memcpy alignment issues).
		if qs, ok := any(t.GetStorage()).(*tensor.Q5KStorage); ok {
			if ptr, _, _ := qs.GPUPtr(); ptr != nil {
				continue // already on GPU
			}
			rawBytes := qs.RawBytes()
			devPtr, err := e.allocWeight(len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc Q5_K GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.uploadBytes(devPtr, rawBytes); err != nil {
				_ = e.runtime.Free(devPtr)
				return fmt.Errorf("upload Q5_K (shape %v): %w", t.Shape(), err)
			}
			qs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			q4Uploaded++
			continue
		}
		// Upload Q6_K raw bytes to GPU for fused GEMV kernel.
		// Q6_K super-blocks (210 bytes per 256 values) are uploaded contiguously.
		if qs, ok := any(t.GetStorage()).(*tensor.Q6KStorage); ok {
			if ptr, _, _ := qs.GPUPtr(); ptr != nil {
				continue // already on GPU
			}
			rawBytes := qs.RawBytes()
			devPtr, err := e.allocWeight(len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc Q6_K GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.uploadBytes(devPtr, rawBytes); err != nil {
				_ = e.runtime.Free(devPtr)
				return fmt.Errorf("upload Q6_K (shape %v): %w", t.Shape(), err)
			}
			qs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			q4Uploaded++
			continue
		}
		// Upload Q5_0 in separated GPU layout (scales | qh | qs) for fast GEMV.
		// The separated layout aligns fp16 and uint32 fields naturally, avoiding
		// byte-wise loads on ARM64 Grace Hopper.
		if qs, ok := any(t.GetStorage()).(*tensor.Q5_0Storage); ok {
			if ptr, _, _ := qs.GPUPtr(); ptr != nil {
				continue // already on GPU
			}
			rawBytes := qs.RawBytesGPU()
			devPtr, err := e.allocWeight(len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc Q5_0 GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.uploadBytes(devPtr, rawBytes); err != nil {
				_ = e.runtime.Free(devPtr)
				return fmt.Errorf("upload Q5_0 (shape %v): %w", t.Shape(), err)
			}
			qs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			q4Uploaded++
			continue
		}
		// Skip MmapStorage in UploadWeights entirely. MmapStorage tensors
		// are handled per-operation by matMulMmap which dequantizes to float32
		// (via Slice()) before GPU upload. Pre-uploading raw quantized bytes
		// causes misaligned address errors on ARM64 (GB10/Grace Hopper) that
		// cascade through CUDA and break graph capture. The per-op path is
		// slower but correct. TODO: investigate ARM64 mmap+cudaMemcpy alignment.
		if _, ok := any(t.GetStorage()).(*tensor.MmapStorage); ok {
			continue
		}
		// Upload FP8 E4M3 raw bytes to GPU and cache the pointer.
		// FP8 weights (1 byte/element) are used with cublasLtMatmul for
		// mixed-precision MatMul (FP8 weights × FP16 activations → FP32 output).
		if fs, ok := any(t.GetStorage()).(*tensor.FP8E4M3Storage); ok {
			if ptr, _, _ := fs.GPUPtr(); ptr != nil {
				continue // already on GPU
			}
			rawBytes := fs.RawBytes()
			devPtr, err := e.allocWeight(len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc FP8 GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.uploadBytes(devPtr, rawBytes); err != nil {
				_ = e.runtime.Free(devPtr)
				return fmt.Errorf("upload FP8 (shape %v): %w", t.Shape(), err)
			}
			fs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			// Upload per-tensor scale factor (single float32) to GPU.
			scale := fs.Scale()
			scaleBytes := unsafe.Slice((*byte)(unsafe.Pointer(&scale)), f32Size)
			scalePtr, err := e.allocWeight(f32Size)
			if err != nil {
				return fmt.Errorf("alloc FP8 scale GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.uploadBytes(scalePtr, scaleBytes); err != nil {
				_ = e.runtime.Free(scalePtr)
				return fmt.Errorf("upload FP8 scale (shape %v): %w", t.Shape(), err)
			}
			fs.SetScaleGPUPtr(scalePtr)
			uploaded++
			continue
		}
		// Upload BFloat16 raw bytes to GPU and cache the pointer.
		// BF16 weights (2 bytes/element) are used with cublasGemmEx for
		// mixed-precision MatMul (BF16 weights × FP32 activations → FP32 output).
		if bs, ok := any(t.GetStorage()).(*tensor.BFloat16Storage); ok {
			if ptr, _, _ := bs.GPUPtr(); ptr != nil {
				continue // already on GPU
			}
			rawBytes := bs.RawBytes()
			devPtr, err := e.allocWeight(len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc BF16 GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.uploadBytes(devPtr, rawBytes); err != nil {
				_ = e.runtime.Free(devPtr)
				return fmt.Errorf("upload BF16 (shape %v): %w", t.Shape(), err)
			}
			bs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			uploaded++
			continue
		}
		// Upload float32 weights to GPU. With Pow, binary ops, and
		// Split/Concat now running on GPU, float32 weights benefit from
		// staying on device (eliminates per-op H2D copies for norm weights).
		if _, ok := t.GetStorage().(*tensor.GPUStorage[float32]); ok {
			continue // already on GPU
		}
		if _, ok := any(t.GetStorage()).(*tensor.Float16Storage); ok {
			continue // already converted to FP16 on GPU
		}
		// Skip Q8 tensors -- they are uploaded as raw bytes in the Q8 loop below.
		if _, ok := any(t.GetStorage()).(*tensor.Q8Storage); ok {
			continue
		}
		// Skip quantized types with dedicated GPU upload handlers above.
		// Without this skip, the F32 loop dequantizes and REPLACES the
		// storage, destroying the native format and blocking GEMV dispatch.
		if _, ok := any(t.GetStorage()).(*tensor.Q4Storage); ok {
			continue
		}
		if _, ok := any(t.GetStorage()).(*tensor.Q4KStorage); ok {
			continue
		}
		if _, ok := any(t.GetStorage()).(*tensor.Q5_0Storage); ok {
			continue
		}
		if _, ok := any(t.GetStorage()).(*tensor.Q5KStorage); ok {
			continue
		}
		if _, ok := any(t.GetStorage()).(*tensor.Q6KStorage); ok {
			continue
		}
		data := t.Data()
		n := len(data)
		if n == 0 {
			continue
		}
		if debugGPU || os.Getenv("UPLOAD_TRACE") == "1" {
			fmt.Fprintf(os.Stderr, "[UPLOAD_F32] shape=%v len=%d storage=%T\n", t.Shape(), n, t.GetStorage())
		}
		byteSize := n * f32Size
		devPtr, err := e.allocWeight(byteSize)
		if err != nil {
			return fmt.Errorf("alloc f32 GPU (shape %v): %w", t.Shape(), err)
		}
		src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), byteSize)
		if err := e.uploadBytes(devPtr, src); err != nil {
			_ = e.runtime.Free(devPtr)
			return fmt.Errorf("upload f32 (shape %v): %w", t.Shape(), err)
		}
		gs, err := tensor.NewGPUStorageFromPtr[float32](devPtr, n, e.deviceID)
		if err != nil {
			_ = e.runtime.Free(devPtr)
			return fmt.Errorf("create GPU storage (shape %v): %w", t.Shape(), err)
		}
		t.SetStorage(gs)
		uploaded++
	}
	// Upload Q8 quantized weights as raw Q8 bytes (36 bytes per 32 values).
	// This keeps Q8 data compressed on GPU (325 MB vs 1.2 GB as F32) and uses
	// the Q8 dequant-GEMM kernel to dequantize during matmul.
	q8Uploaded := 0
	for _, t := range tensors {
		if t == nil {
			continue
		}
		qs, ok := any(t.GetStorage()).(*tensor.Q8Storage)
		if !ok {
			continue
		}
		if ptr, _, _ := qs.GPUPtr(); ptr != nil {
			continue // already on GPU
		}
		rawBytes := qs.RawBytes()
		devPtr, err := e.allocWeight(len(rawBytes))
		if err != nil {
			return fmt.Errorf("alloc Q8 GPU (shape %v): %w", t.Shape(), err)
		}
		if err := e.uploadBytes(devPtr, rawBytes); err != nil {
			_ = e.runtime.Free(devPtr)
			return fmt.Errorf("upload Q8 (shape %v): %w", t.Shape(), err)
		}
		qs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
		q8Uploaded++
	}
	method := "H2D"
	if e.managedMem {
		method = "managed"
	}
	if uploaded > 0 || q4Uploaded > 0 || q8Uploaded > 0 {
		e.logger.Info("weights uploaded to GPU",
			"f32", fmt.Sprintf("%d", uploaded),
			"q4", fmt.Sprintf("%d", q4Uploaded),
			"q8", fmt.Sprintf("%d", q8Uploaded),
			"device", fmt.Sprintf("%d", e.deviceID),
			"method", method)
	}
	return nil
}

// captureStatusFn is the indirection point for cuda.StreamCaptureStatus used
// by ensureNotCapturing. Tests swap it to inject synthetic capture state
// without requiring real CUDA hardware.
var captureStatusFn = cuda.StreamCaptureStatus

// ensureNotCapturing returns ErrCaptureIncompatibleAllocation if the
// engine's stream is currently capturing a CUDA graph. On CPU-only
// runtimes or when the stream handle is nil, returns nil (no capture
// is possible). If querying capture status itself fails, returns
// that error (do not assume safety on probe failure).
func (e *GPUEngine[T]) ensureNotCapturing() error {
	if e.stream == nil {
		return nil
	}
	ptr := e.stream.Ptr()
	if ptr == nil {
		return nil
	}
	s := cuda.StreamFromPtr(ptr)
	status, err := captureStatusFn(s)
	if err != nil {
		return fmt.Errorf("ensureNotCapturing: %w", err)
	}
	if status == cuda.CaptureStatusActive {
		return ErrCaptureIncompatibleAllocation
	}
	return nil
}

// allocWeight allocates permanent memory for a weight tensor.
// Uses cudaMallocManaged on devices with managed memory support,
// otherwise uses cudaMalloc. Returns ErrCaptureIncompatibleAllocation
// if invoked while a CUDA graph capture is active on the engine's stream.
func (e *GPUEngine[T]) allocWeight(byteSize int) (unsafe.Pointer, error) {
	if err := e.ensureNotCapturing(); err != nil {
		return nil, err
	}
	if e.managedMem {
		return cuda.MallocManaged(byteSize)
	}
	return e.runtime.Malloc(byteSize)
}

// uploadBytes copies src bytes into a device (or managed) pointer.
// With managed memory, this is a direct CPU memcpy (no H2D needed).
// Without managed memory, this uses cudaMemcpy H2D. Returns
// ErrCaptureIncompatibleAllocation if invoked while a CUDA graph capture
// is active on the engine's stream.
func (e *GPUEngine[T]) uploadBytes(devPtr unsafe.Pointer, src []byte) error {
	if err := e.ensureNotCapturing(); err != nil {
		return err
	}
	if e.managedMem {
		dst := unsafe.Slice((*byte)(devPtr), len(src))
		copy(dst, src)
		return nil
	}
	return e.runtime.Memcpy(devPtr, unsafe.Pointer(&src[0]), len(src), gpuapi.MemcpyHostToDevice)
}

// Stream returns the engine's GPU stream as an unsafe.Pointer (cudaStream_t).
func (e *GPUEngine[T]) Stream() unsafe.Pointer {
	if e.stream == nil {
		return nil
	}
	return e.stream.Ptr()
}

// GPUStream returns the engine's gpuapi.Stream for async memory operations.
func (e *GPUEngine[T]) GPUStream() gpuapi.Stream {
	return e.stream
}

// BeginCapture starts recording GPU operations into a CUDA graph.
// All subsequent engine ops are recorded until EndCapture is called.
// If the engine's memory pool implements CaptureAwareAllocator, it is
// switched to use cudaMallocAsync on the capture stream so that any
// allocations during capture are recorded as graph nodes instead of
// calling cudaMalloc on the default stream.
func (e *GPUEngine[T]) BeginCapture() error {
	if cap, ok := e.pool.(gpuapi.CaptureAwareAllocator); ok {
		cap.SetCaptureStream(e.Stream())
	}
	s := cuda.StreamFromPtr(e.Stream())
	if err := cuda.StreamBeginCapture(s); err != nil {
		// Roll back capture-aware mode on failure.
		if cap, ok := e.pool.(gpuapi.CaptureAwareAllocator); ok {
			cap.ClearCaptureStream()
		}
		return err
	}
	return nil
}

// EndCapture stops recording and returns a handle to the captured graph.
// The handle can be replayed via ReplayGraph. Capture-aware allocation
// mode is cleared regardless of whether capture succeeds or fails.
func (e *GPUEngine[T]) EndCapture() (GraphHandle, error) {
	// Always clear capture-aware allocation when leaving the capture region.
	if cap, ok := e.pool.(gpuapi.CaptureAwareAllocator); ok {
		defer cap.ClearCaptureStream()
	}
	s := cuda.StreamFromPtr(e.Stream())
	graph, err := cuda.StreamEndCapture(s)
	if err != nil {
		return GraphHandle{}, err
	}
	exec, err := cuda.GraphInstantiate(graph)
	if err != nil {
		cuda.GraphDestroy(graph)
		return GraphHandle{}, err
	}
	// The Graph object is no longer needed after instantiation.
	cuda.GraphDestroy(graph)
	return GraphHandle{ptr: exec}, nil
}

// ReplayGraph executes a previously captured graph on the engine's stream.
func (e *GPUEngine[T]) ReplayGraph(handle GraphHandle) error {
	exec, ok := handle.ptr.(*cuda.GraphExec)
	if !ok || exec == nil {
		return fmt.Errorf("ReplayGraph: invalid graph handle")
	}
	s := cuda.StreamFromPtr(e.Stream())
	if err := cuda.GraphLaunch(exec, s); err != nil {
		return err
	}
	return s.Synchronize()
}

// DestroyGraph releases resources associated with a captured graph.
func (e *GPUEngine[T]) DestroyGraph(handle GraphHandle) error {
	exec, ok := handle.ptr.(*cuda.GraphExec)
	if !ok || exec == nil {
		return nil
	}
	return cuda.GraphExecDestroy(exec)
}

// Close releases the BLAS handle, DNN handle, GPU stream, and drains the memory pool.
// The engine must not be used after Close.
func (e *GPUEngine[T]) Close() error {
	var firstErr error

	// Free FP8 scratch buffers before draining the pool.
	if e.fp8Scratch != nil {
		e.fp8Scratch.free(e.pool, e.deviceID)
		e.fp8Scratch = nil
	}

	if e.pool != nil {
		if err := e.pool.Drain(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if e.stream != nil {
		if err := e.stream.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if e.dnn != nil {
		if err := e.dnn.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if e.blas != nil {
		if err := e.blas.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if e.ltHandle != nil {
		if err := e.ltHandle.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}

// OOMFallbackCount returns the number of times GPU OOM triggered CPU fallback.
func (e *GPUEngine[T]) OOMFallbackCount() int64 {
	return e.oomFallbackCount.Load()
}

// Ops returns the arithmetic ops for this engine.
func (e *GPUEngine[T]) Ops() numeric.Arithmetic[T] { return e.cpu.Ops() }

// MatMul performs matrix multiplication using GPU BLAS for float32 and BFloat16
// tensors. For Q4_0 quantized tensors, uses the Q4 dequant-GEMM kernel.
// For other types, it falls back to the CPU implementation.
// Supports 2D matrices and batched matmul (3D+ tensors).
func (e *GPUEngine[T]) MatMul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("MatMul: input tensors must not be nil")
	}

	// VRAM bounds check: estimate output size before dispatching to any path.
	// For A [..., m, k] x B [..., k, n], output is [..., m, n] * sizeof(float32).
	// This catches oversized allocations early (e.g., 128256x4096 LM head) and
	// returns a clear error instead of segfaulting inside cuBLAS.
	{
		aShape := a.Shape()
		bShape := b.Shape()
		if len(aShape) >= 2 && len(bShape) >= 2 {
			m := aShape[len(aShape)-2]
			n := bShape[len(bShape)-1]
			batchSize := 1
			for _, d := range aShape[:len(aShape)-2] {
				batchSize *= d
			}
			// Use float32 (4 bytes) as the element size for the output estimate,
			// since all MatMul paths produce float32 outputs.
			estimatedBytes := batchSize * m * n * 4
			if err := e.checkVRAMBounds("MatMul", estimatedBytes); err != nil {
				return nil, err
			}
		}
	}

	// Quantized MatMul dispatch chain. Order matters for performance:
	// check the most common quantization types first to minimize failing
	// type assertions on the hot path. Q4_K and Q4 are checked first
	// because GGUF Q4_K_M models (the primary benchmark target) use these.

	// Check for Q4_K quantized storage on A.
	if qs, ok := any(a.GetStorage()).(*tensor.Q4KStorage); ok {
		return e.matMulQ4K(ctx, qs, a, b, dst...)
	}

	// Check for Q4_K quantized storage on B (virtual-transposed weights).
	if qs, ok := any(b.GetStorage()).(*tensor.Q4KStorage); ok {
		return e.matMulQ4KBWeight(ctx, a, qs, b, dst...)
	}

	// Check for Q4 quantized storage on A.
	// Use any() to avoid impossible type assertion when T != float32.
	if qs, ok := any(a.GetStorage()).(*tensor.Q4Storage); ok {
		return e.matMulQ4(ctx, qs, a, b, dst...)
	}

	// Check for Q4 quantized storage on B (virtual-transposed weights).
	if qs, ok := any(b.GetStorage()).(*tensor.Q4Storage); ok {
		return e.matMulQ4BWeight(ctx, a, qs, b, dst...)
	}

	// Check for Q8 quantized storage on A.
	if qs, ok := any(a.GetStorage()).(*tensor.Q8Storage); ok {
		return e.matMulQ8(ctx, qs, a, b, dst...)
	}

	// Check for Q8 quantized storage on B (virtual-transposed weights).
	if qs, ok := any(b.GetStorage()).(*tensor.Q8Storage); ok {
		return e.matMulQ8BWeight(ctx, a, qs, b, dst...)
	}

	// Less common quantization types checked after the fast path.

	// Check for Q6_K quantized storage on A.
	if qs, ok := any(a.GetStorage()).(*tensor.Q6KStorage); ok {
		return e.matMulQ6K(ctx, qs, a, b, dst...)
	}

	// Check for Q6_K quantized storage on B (virtual-transposed weights).
	if qs, ok := any(b.GetStorage()).(*tensor.Q6KStorage); ok {
		return e.matMulQ6KBWeight(ctx, a, qs, b, dst...)
	}

	// Check for Q5_K quantized storage on A.
	if qs, ok := any(a.GetStorage()).(*tensor.Q5KStorage); ok {
		return e.matMulQ5K(ctx, qs, a, b, dst...)
	}

	// Check for Q5_K quantized storage on B (virtual-transposed weights).
	if qs, ok := any(b.GetStorage()).(*tensor.Q5KStorage); ok {
		return e.matMulQ5KBWeight(ctx, a, qs, b, dst...)
	}

	// Check for Q5_0 quantized storage on A.
	if qs, ok := any(a.GetStorage()).(*tensor.Q5_0Storage); ok {
		return e.matMulQ5_0(ctx, qs, a, b, dst...)
	}

	// Check for Q5_0 quantized storage on B (virtual-transposed weights).
	if qs, ok := any(b.GetStorage()).(*tensor.Q5_0Storage); ok {
		return e.matMulQ5_0BWeight(ctx, a, qs, b, dst...)
	}

	// Check for MmapStorage on A -- route to the appropriate quantized kernel
	// based on the GGML type stored in the mmap'd region.
	if ms, ok := any(a.GetStorage()).(*tensor.MmapStorage); ok {
		return e.matMulMmap(ctx, ms, a, b, dst...)
	}

	// Check for MmapStorage on B (virtual-transposed weights).
	if ms, ok := any(b.GetStorage()).(*tensor.MmapStorage); ok {
		return e.matMulMmapB(ctx, a, ms, b, dst...)
	}

	// FP16/FP8/BF16 storage checks — skip entirely for F32 compute.
	if e.dtype != DTypeF32 {
		// Check for Float16Storage on A or B — pass FP16 device pointers directly.
		aFP16, aIsFP16 := any(a.GetStorage()).(*tensor.Float16Storage)
		bFP16, bIsFP16 := any(b.GetStorage()).(*tensor.Float16Storage)
		if aIsFP16 || bIsFP16 {
			return fp16MatMulNative(e, ctx, a, b, aFP16, bFP16, aIsFP16, bIsFP16, dst...)
		}

		// Check for FP8 E4M3 storage on both A and B (both-FP8 native GEMM).
		fsA, aIsFP8 := any(a.GetStorage()).(*tensor.FP8E4M3Storage)
		fsB, bIsFP8 := any(b.GetStorage()).(*tensor.FP8E4M3Storage)
		if aIsFP8 && bIsFP8 {
			return e.matMulFP8Both(ctx, fsA, fsB, a, b, dst...)
		}

		// Check for FP8 E4M3 storage on A (FP8 weights × FP32 activations).
		if aIsFP8 {
			return e.matMulFP8(ctx, fsA, a, b, dst...)
		}

		// Check for FP8 E4M3 storage on B (FP8 weights as B operand).
		if bIsFP8 {
			return e.matMulFP8BWeight(ctx, a, fsB, b, dst...)
		}

		// Check for BFloat16Storage on A (BF16 weights × FP32 activations).
		if bs, ok := any(a.GetStorage()).(*tensor.BFloat16Storage); ok {
			return e.matMulBF16(ctx, bs, a, b, dst...)
		}

		// Check for BFloat16Storage on B (BF16 weights as B operand).
		if bs, ok := any(b.GetStorage()).(*tensor.BFloat16Storage); ok {
			return e.matMulBF16BWeight(ctx, a, bs, b, dst...)
		}
	}

	// FP16 compute path: convert F32 inputs to FP16, run mixed-precision GEMM.
	if (e.dtype == DTypeFP16 || e.dtype == DTypeFP8) && isFloat32[T]() {
		aShape := a.Shape()
		bShape := b.Shape()
		if len(aShape) >= 2 && len(bShape) >= 2 {
			return fp16MatMul(e, ctx, a, b, dst...)
		}
	}

	// float32 and BFloat16 have GPU BLAS paths; fall back for other types.
	var zero T
	_, isFloat32 := any(zero).(float32)
	_, isBFloat16 := any(zero).(float16.BFloat16)
	if !isFloat32 && !isBFloat16 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if e.blas == nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return nil, fmt.Errorf("MatMul: tensors must have at least 2 dimensions, got %d and %d", len(aShape), len(bShape))
	}

	// Extract matrix dimensions from the last two axes.
	aRows := aShape[len(aShape)-2]
	aCols := aShape[len(aShape)-1]
	bRows := bShape[len(bShape)-2]
	bCols := bShape[len(bShape)-1]

	if aCols != bRows {
		return nil, fmt.Errorf("MatMul: incompatible shapes %v and %v (inner dimensions %d != %d)", aShape, bShape, aCols, bRows)
	}

	m, k, n := aRows, aCols, bCols

	// Compute batch dimensions.
	aBatch := aShape[:len(aShape)-2]
	bBatch := bShape[:len(bShape)-2]

	aBatchSize := 1
	for _, d := range aBatch {
		aBatchSize *= d
	}

	bBatchSize := 1
	for _, d := range bBatch {
		bBatchSize *= d
	}

	// For batched matmul: a has batch dims, b may be unbatched (broadcast).
	if bBatchSize != 1 && aBatchSize != bBatchSize {
		return nil, fmt.Errorf("MatMul: batch dimensions %v and %v are incompatible", aBatch, bBatch)
	}

	batchSize := aBatchSize
	if bBatchSize > batchSize {
		batchSize = bBatchSize
	}

	// Build output shape.
	outShape := make([]int, 0, len(aShape))
	outShape = append(outShape, aBatch...)
	outShape = append(outShape, m, n)

	// Get device pointers for inputs.
	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		e.oomFallbackCount.Add(1)
		e.logger.Warn("MatMul: GPU alloc failed, falling back to CPU", "error", err.Error())

		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	defer cleanupA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		e.oomFallbackCount.Add(1)
		e.logger.Warn("MatMul: GPU alloc failed, falling back to CPU", "error", err.Error())

		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	defer cleanupB()

	elemSize := int(unsafe.Sizeof(zero))
	aMatSize := m * k
	bMatSize := k * n
	cMatSize := m * n

	// VRAM bounds check: reject allocations that would exceed device memory
	// and cause a segfault (e.g., 128256x4096 LM head output).
	outputBytes := batchSize * cMatSize * elemSize
	if err := e.checkVRAMBounds("MatMul", outputBytes); err != nil {
		return nil, err
	}

	// Reuse dst's existing GPU memory when possible (#84).
	devCTotal, reusedC := tryReuseDstPtr[T](batchSize*cMatSize, dst)
	if !reusedC {
		devCTotal, err = e.pool.Alloc(e.deviceID, outputBytes)
		if err != nil {
			e.oomFallbackCount.Add(1)
			e.logger.Warn("MatMul: GPU output alloc failed, falling back to CPU", "error", err.Error())

			return e.cpu.MatMul(ctx, a, b, dst...)
		}
	}

	// Use strided batched GEMM when available for float32 with batch > 1.
	// This replaces N sequential Sgemm calls with a single cuBLAS call.
	// When bBatchSize == 1, strideB = 0 broadcasts B across all batches
	// (supported by cuBLAS). This is critical for GQA attention where K/V
	// have fewer heads than Q, replacing N individual Sgemm calls with 1.
	if batchSize > 1 && isFloat32 {
		if batched, ok := e.blas.(gpuapi.BLASBatched); ok {
			strideA := int64(aMatSize)
			strideBVal := int64(bMatSize)
			if bBatchSize == 1 {
				strideBVal = 0
			}
			strideC := int64(cMatSize)
			if debugGPU {
				e.logger.Debug("MatMul: SgemmStridedBatched call",
					"m", fmt.Sprintf("%d", m),
					"n", fmt.Sprintf("%d", n),
					"k", fmt.Sprintf("%d", k),
					"batchSize", fmt.Sprintf("%d", batchSize),
					"devA", fmt.Sprintf("%p", devA),
					"devB", fmt.Sprintf("%p", devB),
					"devC", fmt.Sprintf("%p", devCTotal))
			}
			if err := batched.SgemmStridedBatched(m, n, k, 1.0,
				devA, strideA, devB, strideBVal, 0.0,
				devCTotal, strideC, batchSize); err != nil {
				if !reusedC {
					e.pool.Free(e.deviceID, devCTotal, outputBytes)
				}
				return nil, fmt.Errorf("MatMul: batched GEMM: %w", err)
			}
			if reusedC {
				return finishReusedDst[T](dst[0], outShape), nil
			}
			return makeGPUResult[T](e, outShape, devCTotal, batchSize*cMatSize, dst...)
		}
	}

	for batch := range batchSize {
		aOff := batch * aMatSize * elemSize
		bOff := 0
		if bBatchSize > 1 {
			bOff = batch * bMatSize * elemSize
		}

		cOff := batch * cMatSize * elemSize

		batchDevA := unsafe.Add(devA, aOff)
		batchDevB := unsafe.Add(devB, bOff)
		batchDevC := unsafe.Add(devCTotal, cOff)

		if debugGPU {
			e.logger.Debug("MatMul: Sgemm call",
				"batch", fmt.Sprintf("%d/%d", batch, batchSize),
				"m", fmt.Sprintf("%d", m),
				"n", fmt.Sprintf("%d", n),
				"k", fmt.Sprintf("%d", k),
				"devA", fmt.Sprintf("%p", batchDevA),
				"devB", fmt.Sprintf("%p", batchDevB),
				"devC", fmt.Sprintf("%p", batchDevC))
		}

		var blasErr error
		if isBFloat16 {
			blasErr = e.blas.BFloat16Gemm(m, n, k, 1.0, batchDevA, batchDevB, 0.0, batchDevC)
		} else {
			blasErr = e.blas.Sgemm(m, n, k, 1.0, batchDevA, batchDevB, 0.0, batchDevC)
		}

		if blasErr != nil {
			if !reusedC {
				e.pool.Free(e.deviceID, devCTotal, outputBytes)
			}

			return nil, fmt.Errorf("MatMul: BLAS batch %d: %w", batch, blasErr)
		}
	}

	if reusedC {
		return finishReusedDst[T](dst[0], outShape), nil
	}
	return makeGPUResult[T](e, outShape, devCTotal, batchSize*cMatSize, dst...)
}

// MatMulTransposeB computes C = A * B^T using cuBLAS SgemmNT, avoiding
// an explicit Transpose allocation and kernel launch.
// A is [...batch, m, k], B is [...batch, n, k], result is [...batch, m, n].
// Supports batch broadcasting (bBatch=1 broadcasts B across A's batch).
func (e *GPUEngine[T]) MatMulTransposeB(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("MatMulTransposeB: input tensors must not be nil")
	}

	// Check for SgemmNT support.
	ntBLAS, ok := e.blas.(gpuapi.BLASTransposeB)
	if !ok {
		// Fall back: explicit transpose + standard MatMul.
		kT, err := e.Transpose(ctx, b, []int{0, 2, 1})
		if err != nil {
			return nil, err
		}
		return e.MatMul(ctx, a, kT, dst...)
	}

	var zero T
	_, isFloat32 := any(zero).(float32)
	if !isFloat32 {
		kT, err := e.Transpose(ctx, b, []int{0, 2, 1})
		if err != nil {
			return nil, err
		}
		return e.MatMul(ctx, a, kT, dst...)
	}

	e.setDevice()

	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return nil, fmt.Errorf("MatMulTransposeB: tensors must have at least 2 dimensions")
	}

	// A: [..., m, k], B: [..., n, k] → C: [..., m, n]
	m := aShape[len(aShape)-2]
	k := aShape[len(aShape)-1]
	n := bShape[len(bShape)-2] // B's second-to-last dim is n (B^T would be [k, n])
	bk := bShape[len(bShape)-1]

	if k != bk {
		return nil, fmt.Errorf("MatMulTransposeB: k dimensions must match: A[..., %d] vs B[..., %d]", k, bk)
	}

	aBatchSize := 1
	for _, d := range aShape[:len(aShape)-2] {
		aBatchSize *= d
	}
	bBatchSize := 1
	for _, d := range bShape[:len(bShape)-2] {
		bBatchSize *= d
	}
	if bBatchSize != 1 && aBatchSize != bBatchSize {
		return nil, fmt.Errorf("MatMulTransposeB: batch dimensions %v and %v are incompatible", aShape[:len(aShape)-2], bShape[:len(bShape)-2])
	}

	batchSize := aBatchSize
	if bBatchSize > batchSize {
		batchSize = bBatchSize
	}

	outShape := make([]int, 0, len(aShape))
	outShape = append(outShape, aShape[:len(aShape)-2]...)
	outShape = append(outShape, m, n)

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		kT, tErr := e.Transpose(ctx, b, []int{0, 2, 1})
		if tErr != nil {
			return nil, tErr
		}
		return e.MatMul(ctx, a, kT, dst...)
	}
	defer cleanupA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		kT, tErr := e.Transpose(ctx, b, []int{0, 2, 1})
		if tErr != nil {
			return nil, tErr
		}
		return e.MatMul(ctx, a, kT, dst...)
	}
	defer cleanupB()

	elemSize := int(unsafe.Sizeof(zero))
	aMatSize := m * k
	bMatSize := n * k
	cMatSize := m * n

	// VRAM bounds check: reject allocations that would exceed device memory.
	outputBytes := batchSize * cMatSize * elemSize
	if err := e.checkVRAMBounds("MatMulTransposeB", outputBytes); err != nil {
		return nil, err
	}

	devC, err := e.pool.Alloc(e.deviceID, outputBytes)
	if err != nil {
		kT, tErr := e.Transpose(ctx, b, []int{0, 2, 1})
		if tErr != nil {
			return nil, tErr
		}
		return e.MatMul(ctx, a, kT, dst...)
	}

	// Use strided batched NT GEMM when available for batch > 1.
	// This replaces N sequential SgemmNT calls with a single cuBLAS call.
	// When bBatchSize == 1, strideB = 0 broadcasts B across all batches
	// (supported by cuBLAS). Critical for GQA attention decode performance.
	if batchSize > 1 {
		if batchedNT, ok := e.blas.(gpuapi.BLASBatchedTransposeB); ok {
			strideBVal := int64(bMatSize)
			if bBatchSize == 1 {
				strideBVal = 0
			}
			if err := batchedNT.SgemmNTStridedBatched(m, n, k, 1.0,
				devA, int64(aMatSize), devB, strideBVal, 0.0,
				devC, int64(cMatSize), batchSize); err != nil {
				e.pool.Free(e.deviceID, devC, batchSize*cMatSize*elemSize)
				return nil, fmt.Errorf("MatMulTransposeB: batched NT GEMM: %w", err)
			}
			return makeGPUResult[T](e, outShape, devC, batchSize*cMatSize, dst...)
		}
	}

	for batch := range batchSize {
		aOff := batch * aMatSize * elemSize
		bOff := 0
		if bBatchSize > 1 {
			bOff = batch * bMatSize * elemSize
		}
		cOff := batch * cMatSize * elemSize

		if err := ntBLAS.SgemmNT(m, n, k, 1.0,
			unsafe.Add(devA, aOff),
			unsafe.Add(devB, bOff),
			0.0,
			unsafe.Add(devC, cOff),
		); err != nil {
			e.pool.Free(e.deviceID, devC, batchSize*cMatSize*elemSize)
			return nil, fmt.Errorf("MatMulTransposeB: BLAS batch %d: %w", batch, err)
		}
	}

	return makeGPUResult[T](e, outShape, devC, batchSize*cMatSize, dst...)
}

// matMulQ4 handles GPU Q4_0 dequant-GEMM: C = dequant(A_q4) * B.
// Only supports unbatched 2D for now; batched Q4 falls back to CPU.
func (e *GPUEngine[T]) matMulQ4(ctx context.Context, qs *tensor.Q4Storage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, fallback := aShapeCheck2D(a.Shape(), b.Shape(), 32)
	if fallback {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	bpr := k / 32
	ptr, _, _ := qs.GPUPtr()
	devA, freeA, err := e.uploadRawBytes(ptr, qs.RawBytesGPU(bpr))
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupB()

	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	totalBlocks := m * (k / 32)
	dataOff := tensor.Q4GPUDataOffset(totalBlocks)
	if err := e.kernels.GemmQ4F32(devA, devB, devC, m, k, n, dataOff, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	return makeGPUResult[T](e, []int{m, n}, devC, m*n, dst...)
}

// matMulQ4BWeight handles MatMul where B has Q4 storage (virtual-transposed weight).
// B's shape after virtual transpose is [K, N], but the Q4 data is laid out as [N, K].
// We compute C[M, N] = A[M, K] * dequant(B)^T by reformulating as:
//
//	C_temp[N, M] = gemm_q4(B_q4[N, K], A^T[K, M])
//
// For GEMV (M=1), A^T[K,1] is just A's data as a column, and C_temp[N,1]
// can be reshaped to [1, N] without a physical transpose.
func (e *GPUEngine[T]) matMulQ4BWeight(ctx context.Context, a *tensor.TensorNumeric[T], qs *tensor.Q4Storage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if debugGPU {
		fmt.Fprintf(os.Stderr, "matMulQ4BWeight: aShape=%v bShape=%v GPUPtr=%v\n", a.Shape(), b.Shape(), func() bool { p, _, _ := qs.GPUPtr(); return p != nil }())
	}

	m, k, n, outShape, fallback := bweightShapeMKN(a.Shape(), b.Shape(), 32)
	if fallback {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	bpr := k / 32
	ptr, _, _ := qs.GPUPtr()
	devQ4, freeQ4, err := e.uploadRawBytes(ptr, qs.RawBytesGPU(bpr))
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeQ4()

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupA()

	q4DataOff := tensor.Q4GPUDataOffset(qs.NumBlocks())

	if m == 1 {
		// GEMV fast path: C_temp[N, 1] = gemm_q4(B_q4[N,K], A^T[K,1])
		result, err := e.quantGemvResult(outShape, n, func(devC unsafe.Pointer) error {
			return e.kernels.GemmQ4F32(devQ4, devA, devC, n, k, 1, q4DataOff, e.stream)
		}, dst...)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		return result, nil
	}

	// General GEMM: C_temp[N, M] = gemm_q4(B_q4[N,K], A^T[K,M])
	aFlat, err := e.Reshape(ctx, a, []int{m, k})
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	aT, err := e.Transpose(ctx, aFlat, []int{1, 0})
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	devAT, cleanupAT, err := getDevicePtr(e, aT)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupAT()

	cTempSize := n * m * f32Size
	devCTemp, err := e.pool.Alloc(e.deviceID, cTempSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.kernels.GemmQ4F32(devQ4, devAT, devCTemp, n, k, m, q4DataOff, e.stream); err != nil {
		e.pool.Free(e.deviceID, devCTemp, cTempSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// C_temp is [N, M], transpose to [M, N], then reshape to outShape.
	cTempTensor, err := makeGPUResult[T](e, []int{n, m}, devCTemp, n*m)
	if err != nil {
		e.pool.Free(e.deviceID, devCTemp, cTempSize)
		return nil, err
	}

	cFlat, err := e.Transpose(ctx, cTempTensor, []int{1, 0})
	if err != nil {
		return nil, err
	}
	return e.Reshape(ctx, cFlat, outShape, dst...)
}

// matMulQ4K handles GPU Q4_K dequant-GEMM when Q4_K storage is on A.
// For GEMV (n==1, single-column B), uses fused dequant+GEMV kernel.
// For general GEMM (n>1), dequantizes Q4_K to F32 on GPU then calls cuBLAS Sgemm.
func (e *GPUEngine[T]) matMulQ4K(ctx context.Context, qs *tensor.Q4KStorage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, fallback := aShapeCheck2D(a.Shape(), b.Shape(), 0)
	if fallback {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	ptr, _, _ := qs.GPUPtr()
	devW, freeW, err := e.uploadRawBytes(ptr, qs.RawBytes())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeW()

	// Fused GEMV path: y = dequant(W_q4k) * x, when n==1 and K is 256-aligned.
	if n == 1 && k%256 == 0 {
		devX, cleanupX, err := getDevicePtr(e, b)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		defer cleanupX()

		result, err := e.quantGemvResult([]int{m, n}, m*n, func(devY unsafe.Pointer) error {
			return e.kernels.GemvQ4KF32(devW, devX, devY, m, k, e.stream)
		}, dst...)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		return result, nil
	}

	// General GEMM: dequantize Q4_K to F32 on GPU, then cuBLAS Sgemm.
	var gpuDequant func(src, dst unsafe.Pointer, rows, cols int) error
	if k%256 == 0 {
		gpuDequant = func(src, dst unsafe.Pointer, rows, cols int) error {
			return e.kernels.DequantQ4KF32(src, dst, rows, cols, e.stream)
		}
	}
	result, err := e.dequantSgemm(devW, m, k, n, gpuDequant, qs.Dequantize, b, []int{m, n}, "matMulQ4K", dst...)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	return result, nil
}

// matMulQ4KBWeight handles MatMul where B has Q4_K storage (virtual-transposed weight).
// B's shape after virtual transpose is [K, N], but the Q4_K data is laid out as [N, K].
// For GEMV (m==1, single-token decode), uses fused dequant+GEMV kernel directly
// on the Q4_K weight data, halving memory bandwidth vs separate dequant + GEMM.
// For general GEMM (m>1), dequantizes Q4_K to F32 on GPU then calls cuBLAS SgemmNT.
func (e *GPUEngine[T]) matMulQ4KBWeight(ctx context.Context, a *tensor.TensorNumeric[T], qs *tensor.Q4KStorage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, outShape, fallback := bweightShapeMKN(a.Shape(), b.Shape(), 0)
	if fallback {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	ptr, _, _ := qs.GPUPtr()
	devQ4K, freeQ4K, err := e.uploadRawBytes(ptr, qs.RawBytes())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeQ4K()

	// Fused GEMV path: y[n] = sum_k dequant(B_q4k[n, k]) * x[k], when m==1.
	if m == 1 && k%256 == 0 {
		devX, cleanupX, err := getDevicePtr(e, a)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		defer cleanupX()

		result, err := e.quantGemvResult(outShape, n, func(devY unsafe.Pointer) error {
			return e.kernels.GemvQ4KF32(devQ4K, devX, devY, n, k, e.stream)
		}, dst...)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		return result, nil
	}

	// General GEMM: dequantize Q4_K to F32, then SgemmNT.
	var gpuDequant func(src, dst unsafe.Pointer, rows, cols int) error
	if k%256 == 0 {
		gpuDequant = func(src, dst unsafe.Pointer, rows, cols int) error {
			return e.kernels.DequantQ4KF32(src, dst, rows, cols, e.stream)
		}
	}
	result, err := e.sgemmNTOrFallback(devQ4K, m, k, n, gpuDequant, qs.Dequantize, a, outShape, "matMulQ4KBWeight", dst...)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	return result, nil
}

// matMulQ6K handles GPU Q6_K dequant-GEMM when Q6_K storage is on A.
// For GEMV (n==1, single-column B), uses fused dequant+GEMV kernel.
// For general GEMM (n>1), dequantizes Q6_K to F32 on CPU then calls cuBLAS Sgemm.
func (e *GPUEngine[T]) matMulQ6K(ctx context.Context, qs *tensor.Q6KStorage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, fallback := aShapeCheck2D(a.Shape(), b.Shape(), 256)
	if fallback {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	ptr, _, _ := qs.GPUPtr()
	devW, freeW, err := e.uploadRawBytes(ptr, qs.RawBytes())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeW()

	// Fused GEMV path: y = dequant(W_q6k) * x, when n==1.
	if n == 1 {
		devX, cleanupX, err := getDevicePtr(e, b)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		defer cleanupX()

		result, err := e.quantGemvResult([]int{m, n}, m*n, func(devY unsafe.Pointer) error {
			return e.kernels.GemvQ6KF32(devW, devX, devY, m, k, e.stream)
		}, dst...)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		return result, nil
	}

	// General GEMM: dequantize Q6_K to F32 on CPU, upload, cuBLAS Sgemm.
	result, err := e.dequantSgemm(devW, m, k, n, nil, qs.Dequantize, b, []int{m, n}, "matMulQ6K", dst...)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	return result, nil
}

// matMulQ6KBWeight handles MatMul where B has Q6_K storage (virtual-transposed weight).
// B's shape after virtual transpose is [K, N], but the Q6_K data is laid out as [N, K].
// For GEMV (m==1, single-token decode), uses fused dequant+GEMV kernel directly
// on the Q6_K weight data, halving memory bandwidth vs separate dequant + GEMM.
// For general GEMM (m>1), dequantizes Q6_K to F32 on CPU then calls cuBLAS SgemmNT.
func (e *GPUEngine[T]) matMulQ6KBWeight(ctx context.Context, a *tensor.TensorNumeric[T], qs *tensor.Q6KStorage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, outShape, fallback := bweightShapeMKN(a.Shape(), b.Shape(), 256)
	if fallback {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	ptr, _, _ := qs.GPUPtr()
	devQ6K, freeQ6K, err := e.uploadRawBytes(ptr, qs.RawBytes())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeQ6K()

	// Fused GEMV path: y[n] = sum_k dequant(B_q6k[n, k]) * x[k], when m==1.
	if m == 1 {
		devX, cleanupX, err := getDevicePtr(e, a)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		defer cleanupX()

		result, err := e.quantGemvResult(outShape, n, func(devY unsafe.Pointer) error {
			return e.kernels.GemvQ6KF32(devQ6K, devX, devY, n, k, e.stream)
		}, dst...)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		return result, nil
	}

	// General GEMM: dequantize Q6_K to F32, then SgemmNT.
	gpuDequant := func(src, dst unsafe.Pointer, rows, cols int) error {
		return e.kernels.DequantQ6KF32(src, dst, rows, cols, e.stream)
	}
	result, err := e.sgemmNTOrFallback(devQ6K, m, k, n, gpuDequant, qs.Dequantize, a, outShape, "matMulQ6KBWeight", dst...)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	return result, nil
}

// matMulQ5K handles GPU Q5_K dequant-GEMV when Q5_K storage is on A.
// For GEMV (n==1, single-column B), uses fused dequant+GEMV kernel.
// For general GEMM (n>1), falls back to CPU.
func (e *GPUEngine[T]) matMulQ5K(ctx context.Context, qs *tensor.Q5KStorage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, fallback := aShapeCheck2D(a.Shape(), b.Shape(), 256)
	if fallback || n != 1 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	ptr, _, _ := qs.GPUPtr()
	devW, freeW, err := e.uploadRawBytes(ptr, qs.RawBytes())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeW()

	devX, cleanupX, err := getDevicePtr(e, b)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupX()

	result, err := e.quantGemvResult([]int{m, n}, m*n, func(devY unsafe.Pointer) error {
		return e.kernels.GemvQ5KF32(devW, devX, devY, m, k, e.stream)
	}, dst...)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	return result, nil
}

// matMulQ5KBWeight handles MatMul where B has Q5_K storage (virtual-transposed weight).
// B's shape after virtual transpose is [K, N], but the Q5_K data is laid out as [N, K].
// For GEMV (m==1, single-token decode), uses fused dequant+GEMV kernel.
// For general GEMM (m>1), dequantizes Q5_K to F32 on GPU then calls cuBLAS SgemmNT.
func (e *GPUEngine[T]) matMulQ5KBWeight(ctx context.Context, a *tensor.TensorNumeric[T], qs *tensor.Q5KStorage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, outShape, fallback := bweightShapeMKN(a.Shape(), b.Shape(), 256)
	if fallback {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	ptr, _, _ := qs.GPUPtr()
	devQ5K, freeQ5K, err := e.uploadRawBytes(ptr, qs.RawBytes())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeQ5K()

	// Fused GEMV path: y[n] = sum_k dequant(B_q5k[n, k]) * x[k], when m==1.
	if m == 1 {
		devX, cleanupX, err := getDevicePtr(e, a)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		defer cleanupX()

		result, err := e.quantGemvResult(outShape, n, func(devY unsafe.Pointer) error {
			return e.kernels.GemvQ5KF32(devQ5K, devX, devY, n, k, e.stream)
		}, dst...)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		return result, nil
	}

	// General GEMM (M>1): dequantize Q5_K to F32 on GPU, then SgemmNT.
	gpuDequant := func(src, dst unsafe.Pointer, rows, cols int) error {
		return e.kernels.DequantQ5KF32(src, dst, rows, cols, e.stream)
	}
	result, err := e.sgemmNTOrFallback(devQ5K, m, k, n, gpuDequant, nil, a, outShape, "matMulQ5KBWeight", dst...)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	return result, nil
}

// matMulQ5_0 handles GPU Q5_0 dequant-GEMM when Q5_0 storage is on A.
// For GEMV (n==1, single-column B), uses fused dequant+GEMV kernel.
// For general GEMM (n>1), dequantizes Q5_0 to F32 on CPU then calls cuBLAS Sgemm.
func (e *GPUEngine[T]) matMulQ5_0(ctx context.Context, qs *tensor.Q5_0Storage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, fallback := aShapeCheck2D(a.Shape(), b.Shape(), 32)
	if fallback {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	nBlocks := qs.NumBlocks()
	ptr, _, _ := qs.GPUPtr()
	devW, freeW, err := e.uploadRawBytes(ptr, qs.RawBytesGPU())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeW()

	qhOff := tensor.Q5_0GPUQhOffset(nBlocks)
	qsOff := tensor.Q5_0GPUQsOffset(nBlocks)

	// Fused GEMV path.
	if n == 1 {
		devX, cleanupX, err := getDevicePtr(e, b)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		defer cleanupX()

		result, err := e.quantGemvResult([]int{m, n}, m*n, func(devY unsafe.Pointer) error {
			return e.kernels.GemvQ5_0F32(devW, devX, devY, m, k, qhOff, qsOff, e.stream)
		}, dst...)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		return result, nil
	}

	// General GEMM: dequantize Q5_0 to F32 on CPU, then cuBLAS Sgemm.
	result, err := e.dequantSgemm(devW, m, k, n, nil, qs.Dequantize, b, []int{m, n}, "matMulQ5_0", dst...)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	return result, nil
}

// matMulQ5_0BWeight handles MatMul where B has Q5_0 storage (virtual-transposed weight).
func (e *GPUEngine[T]) matMulQ5_0BWeight(ctx context.Context, a *tensor.TensorNumeric[T], qs *tensor.Q5_0Storage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, outShape, fallback := bweightShapeMKN(a.Shape(), b.Shape(), 32)
	if fallback {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	nBlocks := qs.NumBlocks()
	ptr, _, _ := qs.GPUPtr()
	devQ5_0, freeQ5_0, err := e.uploadRawBytes(ptr, qs.RawBytesGPU())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeQ5_0()

	qhOff := tensor.Q5_0GPUQhOffset(nBlocks)
	qsOff := tensor.Q5_0GPUQsOffset(nBlocks)

	// Fused GEMV path.
	if m == 1 {
		devX, cleanupX, err := getDevicePtr(e, a)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		defer cleanupX()

		result, err := e.quantGemvResult(outShape, n, func(devY unsafe.Pointer) error {
			return e.kernels.GemvQ5_0F32(devQ5_0, devX, devY, n, k, qhOff, qsOff, e.stream)
		}, dst...)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		return result, nil
	}

	// General GEMM (M>1): dequantize Q5_0 to F32, then SgemmNT.
	gpuDequant := func(src, dst unsafe.Pointer, rows, cols int) error {
		return e.kernels.DequantQ5_0F32(src, dst, rows, cols, e.stream)
	}
	result, err := e.sgemmNTOrFallback(devQ5_0, m, k, n, gpuDequant, qs.Dequantize, a, outShape, "matMulQ5_0BWeight", dst...)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	return result, nil
}

// matMulQ8 handles GPU Q8_0 dequant-GEMM: C = dequant(A_q8) * B.
// Only supports unbatched 2D for now; batched Q8 falls back to CPU.
func (e *GPUEngine[T]) matMulQ8(ctx context.Context, qs *tensor.Q8Storage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, fallback := aShapeCheck2D(a.Shape(), b.Shape(), 32)
	if fallback {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	ptr, _, _ := qs.GPUPtr()
	devA, freeA, err := e.uploadRawBytes(ptr, qs.RawBytes())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupB()

	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.kernels.GemmQ8F32(devA, devB, devC, m, k, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	return makeGPUResult[T](e, []int{m, n}, devC, m*n, dst...)
}

// matMulQ8BWeight handles MatMul where B has Q8 storage (virtual-transposed weight).
// B's shape after virtual transpose is [K, N], but the Q8 data is laid out as [N, K].
// We compute C[M, N] = A[M, K] * dequant(B)^T by reformulating as:
//
//	C_temp[N, M] = gemm_q8(B_q8[N, K], A^T[K, M])
//
// For GEMV (M=1), A^T[K,1] is just A's data as a column, and C_temp[N,1]
// can be reshaped to [1, N] without a physical transpose.
func (e *GPUEngine[T]) matMulQ8BWeight(ctx context.Context, a *tensor.TensorNumeric[T], qs *tensor.Q8Storage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, outShape, fallback := bweightShapeMKN(a.Shape(), b.Shape(), 32)
	if fallback {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	ptr, _, _ := qs.GPUPtr()
	devQ8, freeQ8, err := e.uploadRawBytes(ptr, qs.RawBytes())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeQ8()

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupA()

	if m == 1 {
		// GEMV fast path: C_temp[N, 1] = gemm_q8(B_q8[N,K], A^T[K,1])
		result, err := e.quantGemvResult(outShape, n, func(devC unsafe.Pointer) error {
			return e.kernels.GemmQ8F32(devQ8, devA, devC, n, k, 1, e.stream)
		}, dst...)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		return result, nil
	}

	// General GEMM: C_temp[N, M] = gemm_q8(B_q8[N,K], A^T[K,M])
	aFlat, err := e.Reshape(ctx, a, []int{m, k})
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	aT, err := e.Transpose(ctx, aFlat, []int{1, 0})
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	devAT, cleanupAT, err := getDevicePtr(e, aT)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupAT()

	cTempSize := n * m * f32Size
	devCTemp, err := e.pool.Alloc(e.deviceID, cTempSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.kernels.GemmQ8F32(devQ8, devAT, devCTemp, n, k, m, e.stream); err != nil {
		e.pool.Free(e.deviceID, devCTemp, cTempSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	cTempTensor, err := makeGPUResult[T](e, []int{n, m}, devCTemp, n*m)
	if err != nil {
		e.pool.Free(e.deviceID, devCTemp, cTempSize)
		return nil, err
	}

	cFlat, err := e.Transpose(ctx, cTempTensor, []int{1, 0})
	if err != nil {
		return nil, err
	}
	return e.Reshape(ctx, cFlat, outShape, dst...)
}

// matMulBF16 handles MatMul where A has BFloat16Storage.
// A is [M, K] in BF16, B is [K, N] in FP32 -> C is [M, N] in FP32.
// B's FP32 data is converted to BF16 on the fly, then MixedBF16Gemm
// computes with BF16 inputs and FP32 output via cublasGemmEx.
func (e *GPUEngine[T]) matMulBF16(ctx context.Context, bs *tensor.BFloat16Storage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, fallback := aShapeCheck2D(a.Shape(), b.Shape(), 0)
	if fallback || e.blas == nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	ptr, _, _ := bs.GPUPtr()
	devA, freeA, err := e.uploadRawBytes(ptr, bs.RawBytes())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeA()

	// Convert B from FP32 to BF16 and upload.
	bData := b.Data()
	bF32 := *(*[]float32)(unsafe.Pointer(&bData))
	bBF16 := tensor.NewBFloat16Storage(bF32)
	bBytes := bBF16.RawBytes()
	devB, err := e.pool.Alloc(e.deviceID, len(bBytes))
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer e.pool.Free(e.deviceID, devB, len(bBytes))

	if err := e.runtime.Memcpy(devB, unsafe.Pointer(&bBytes[0]), len(bBytes), gpuapi.MemcpyHostToDevice); err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.blas.MixedBF16Gemm(m, n, k, 1.0, devA, devB, 0.0, devC); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	return makeGPUResult[T](e, []int{m, n}, devC, m*n, dst...)
}

// matMulBF16BWeight handles MatMul where B has BFloat16Storage.
// A is [M, K] in FP32, B is [K, N] in BF16 -> C is [M, N] in FP32.
// A's FP32 data is converted to BF16 on the fly, then MixedBF16Gemm
// computes with BF16 inputs and FP32 output via cublasGemmEx.
func (e *GPUEngine[T]) matMulBF16BWeight(ctx context.Context, a *tensor.TensorNumeric[T], bs *tensor.BFloat16Storage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	m, k, n, outShape, fallback := bweightShapeMKN(a.Shape(), b.Shape(), 0)
	if fallback || e.blas == nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	// Convert A from FP32 to BF16 and upload.
	aData := a.Data()
	aF32 := *(*[]float32)(unsafe.Pointer(&aData))
	aBF16 := tensor.NewBFloat16Storage(aF32)
	aBytes := aBF16.RawBytes()
	devA, err := e.pool.Alloc(e.deviceID, len(aBytes))
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer e.pool.Free(e.deviceID, devA, len(aBytes))

	if err := e.runtime.Memcpy(devA, unsafe.Pointer(&aBytes[0]), len(aBytes), gpuapi.MemcpyHostToDevice); err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	ptr, _, _ := bs.GPUPtr()
	devB, freeB, err := e.uploadRawBytes(ptr, bs.RawBytes())
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer freeB()

	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.blas.MixedBF16Gemm(m, n, k, 1.0, devA, devB, 0.0, devC); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	return makeGPUResult[T](e, outShape, devC, m*n, dst...)
}

// cpuMatMulToGPU runs MatMul on the CPU engine then uploads the result to GPU.
// This ensures callers always receive a GPU-resident tensor, maintaining device
// consistency when the GPU engine falls back to CPU for unsupported quant types.
func (e *GPUEngine[T]) cpuMatMulToGPU(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := e.cpu.MatMul(ctx, a, b, dst...)
	if err != nil {
		return nil, err
	}
	// If already GPU-resident (e.g., dst was provided with GPUStorage), return as-is.
	if _, ok := result.GetStorage().(*tensor.GPUStorage[T]); ok {
		return result, nil
	}
	// Upload CPU result to GPU.
	data := result.Data()
	byteSize := len(data) * int(unsafe.Sizeof(*new(T)))
	devPtr, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return result, nil // fallback: return CPU tensor if GPU alloc fails
	}
	if err := e.runtime.Memcpy(devPtr, unsafe.Pointer(&data[0]), byteSize, gpuapi.MemcpyHostToDevice); err != nil {
		e.pool.Free(e.deviceID, devPtr, byteSize)
		return result, nil
	}
	return makeGPUResult[T](e, result.Shape(), devPtr, len(data), dst...)
}

// matMulMmap handles MatMul where A has MmapStorage. Routes to the appropriate
// quantized kernel based on QType, using the pre-uploaded GPU pointer from
// UploadWeights or uploading raw bytes on the fly.
func (e *GPUEngine[T]) matMulMmap(ctx context.Context, ms *tensor.MmapStorage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()
	if len(aShape) < 2 || len(bShape) < 2 || len(aShape) > 2 || len(bShape) > 2 {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}

	m := aShape[0]
	k := aShape[1]
	n := bShape[1]
	e.setDevice()

	// Acquire GPU pointer for the quantized weight data.
	devW, freeW, err := e.mmapDevicePtr(ms)
	if err != nil {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}
	defer freeW()

	qtype := ms.QType()

	// GEMV fast path (single-token decode: n==1).
	if n == 1 {
		devX, cleanupX, err := getDevicePtr(e, b)
		if err != nil {
			return e.cpuMatMulToGPU(ctx, a, b, dst...)
		}
		defer cleanupX()

		f32Size := int(unsafe.Sizeof(float32(0)))
		cSize := m * f32Size
		devY, err := e.pool.Alloc(e.deviceID, cSize)
		if err != nil {
			return e.cpuMatMulToGPU(ctx, a, b, dst...)
		}

		var kerr error
		switch qtype {
		case tensor.GGMLTypeQ4_K:
			if k%256 != 0 {
				e.pool.Free(e.deviceID, devY, cSize)
				return e.cpuMatMulToGPU(ctx, a, b, dst...)
			}
			kerr = e.kernels.GemvQ4KF32(devW, devX, devY, m, k, e.stream)
		case tensor.GGMLTypeQ4_0:
			if k%32 != 0 {
				e.pool.Free(e.deviceID, devY, cSize)
				return e.cpuMatMulToGPU(ctx, a, b, dst...)
			}
			totalBlocks := (m * k) / 32
			dataOff := tensor.Q4GPUDataOffset(totalBlocks)
			kerr = e.kernels.GemmQ4F32(devW, devX, devY, m, k, 1, dataOff, e.stream)
		case tensor.GGMLTypeQ8_0:
			if k%32 != 0 {
				e.pool.Free(e.deviceID, devY, cSize)
				return e.cpuMatMulToGPU(ctx, a, b, dst...)
			}
			kerr = e.kernels.GemmQ8F32(devW, devX, devY, m, k, 1, e.stream)
		case tensor.GGMLTypeQ6_K:
			if k%256 != 0 {
				e.pool.Free(e.deviceID, devY, cSize)
				return e.cpuMatMulToGPU(ctx, a, b, dst...)
			}
			kerr = e.kernels.GemvQ6KF32(devW, devX, devY, m, k, e.stream)
		case tensor.GGMLTypeQ5_K:
			if k%256 != 0 {
				e.pool.Free(e.deviceID, devY, cSize)
				return e.cpuMatMulToGPU(ctx, a, b, dst...)
			}
			kerr = e.kernels.GemvQ5KF32(devW, devX, devY, m, k, e.stream)
		default:
			e.pool.Free(e.deviceID, devY, cSize)
			return e.cpuMatMulToGPU(ctx, a, b, dst...)
		}
		if kerr != nil {
			e.pool.Free(e.deviceID, devY, cSize)
			return e.cpuMatMulToGPU(ctx, a, b, dst...)
		}
		return makeGPUResult[T](e, []int{m, n}, devY, m*n, dst...)
	}

	// General GEMM: dequantize Q4_K on GPU, then cuBLAS Sgemm.
	// Only Q4_K has a GPU dequant kernel; others fall back to CPU.
	if qtype != tensor.GGMLTypeQ4_K {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}

	f32Size := int(unsafe.Sizeof(float32(0)))
	dequantSize := m * k * f32Size
	devAF32, err := e.pool.Alloc(e.deviceID, dequantSize)
	if err != nil {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}
	defer e.pool.Free(e.deviceID, devAF32, dequantSize)

	if err := e.kernels.DequantQ4KF32(devW, devAF32, m, k, e.stream); err != nil {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}
	defer cleanupB()

	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}

	if err := e.blas.Sgemm(m, n, k, 1.0, devAF32, devB, 0.0, devC); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return nil, fmt.Errorf("matMulMmap: Sgemm: %w", err)
	}

	return makeGPUResult[T](e, []int{m, n}, devC, m*n, dst...)
}

// matMulMmapB handles MatMul where B has MmapStorage (virtual-transposed weight).
// Routes to the appropriate quantized kernel based on QType.
func (e *GPUEngine[T]) matMulMmapB(ctx context.Context, a *tensor.TensorNumeric[T], ms *tensor.MmapStorage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()
	if len(aShape) < 2 || len(bShape) < 2 || len(bShape) > 2 {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}

	// B is virtual-transposed: logical [K, N], physical [N, K].
	k := aShape[len(aShape)-1]
	n := bShape[1]
	m := 1
	for i := 0; i < len(aShape)-1; i++ {
		m *= aShape[i]
	}

	e.setDevice()

	devW, freeW, err := e.mmapDevicePtr(ms)
	if err != nil {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}
	defer freeW()

	qtype := ms.QType()
	nPhys := bShape[0] // physical rows = N (output dim)

	// GEMV fast path (m==1, single-token decode).
	if m == 1 {
		devX, cleanupX, err := getDevicePtr(e, a)
		if err != nil {
			return e.cpuMatMulToGPU(ctx, a, b, dst...)
		}
		defer cleanupX()

		f32Size := int(unsafe.Sizeof(float32(0)))
		cSize := n * f32Size
		devY, err := e.pool.Alloc(e.deviceID, cSize)
		if err != nil {
			return e.cpuMatMulToGPU(ctx, a, b, dst...)
		}

		var kerr error
		switch qtype {
		case tensor.GGMLTypeQ4_K:
			if k%256 != 0 {
				e.pool.Free(e.deviceID, devY, cSize)
				return e.cpuMatMulToGPU(ctx, a, b, dst...)
			}
			kerr = e.kernels.GemvQ4KF32(devW, devX, devY, nPhys, k, e.stream)
		case tensor.GGMLTypeQ4_0:
			if k%32 != 0 {
				e.pool.Free(e.deviceID, devY, cSize)
				return e.cpuMatMulToGPU(ctx, a, b, dst...)
			}
			totalBlocks := (nPhys * k) / 32
			dataOff := tensor.Q4GPUDataOffset(totalBlocks)
			kerr = e.kernels.GemmQ4F32(devW, devX, devY, nPhys, k, 1, dataOff, e.stream)
		case tensor.GGMLTypeQ8_0:
			if k%32 != 0 {
				e.pool.Free(e.deviceID, devY, cSize)
				return e.cpuMatMulToGPU(ctx, a, b, dst...)
			}
			kerr = e.kernels.GemmQ8F32(devW, devX, devY, nPhys, k, 1, e.stream)
		case tensor.GGMLTypeQ6_K:
			if k%256 != 0 {
				e.pool.Free(e.deviceID, devY, cSize)
				return e.cpuMatMulToGPU(ctx, a, b, dst...)
			}
			kerr = e.kernels.GemvQ6KF32(devW, devX, devY, nPhys, k, e.stream)
		case tensor.GGMLTypeQ5_K:
			if k%256 != 0 {
				e.pool.Free(e.deviceID, devY, cSize)
				return e.cpuMatMulToGPU(ctx, a, b, dst...)
			}
			kerr = e.kernels.GemvQ5KF32(devW, devX, devY, nPhys, k, e.stream)
		default:
			e.pool.Free(e.deviceID, devY, cSize)
			return e.cpuMatMulToGPU(ctx, a, b, dst...)
		}
		if kerr != nil {
			e.pool.Free(e.deviceID, devY, cSize)
			return e.cpuMatMulToGPU(ctx, a, b, dst...)
		}

		outShape := make([]int, len(aShape))
		copy(outShape, aShape[:len(aShape)-1])
		outShape[len(outShape)-1] = n
		return makeGPUResult[T](e, outShape, devY, m*n, dst...)
	}

	// General GEMM: dequantize Q4_K on GPU, then cuBLAS SgemmNT.
	// Only Q4_K has a GPU dequant kernel; others fall back to CPU.
	if qtype != tensor.GGMLTypeQ4_K {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}

	f32Size := int(unsafe.Sizeof(float32(0)))
	dequantSize := nPhys * k * f32Size
	devBF32, err := e.pool.Alloc(e.deviceID, dequantSize)
	if err != nil {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}
	defer e.pool.Free(e.deviceID, devBF32, dequantSize)

	if err := e.kernels.DequantQ4KF32(devW, devBF32, nPhys, k, e.stream); err != nil {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}
	defer cleanupA()

	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return e.cpuMatMulToGPU(ctx, a, b, dst...)
	}

	outShape := make([]int, len(aShape))
	copy(outShape, aShape[:len(aShape)-1])
	outShape[len(outShape)-1] = n

	// Use SgemmNT if available (avoids explicit transpose).
	if ntBLAS, ok := e.blas.(gpuapi.BLASTransposeB); ok {
		if err := ntBLAS.SgemmNT(m, n, k, 1.0, devA, devBF32, 0.0, devC); err != nil {
			e.pool.Free(e.deviceID, devC, cSize)
			return nil, fmt.Errorf("matMulMmapB: SgemmNT: %w", err)
		}
		return makeGPUResult[T](e, outShape, devC, m*n, dst...)
	}

	// Fallback: CPU MatMul.
	e.pool.Free(e.deviceID, devC, cSize)
	return e.cpuMatMulToGPU(ctx, a, b, dst...)
}

// mmapDevicePtr returns the GPU device pointer for MmapStorage data. If the data
// has been pre-uploaded via UploadWeights, returns the cached pointer. Otherwise,
// uploads the raw bytes to a temporary allocation from the pool.
func (e *GPUEngine[T]) mmapDevicePtr(ms *tensor.MmapStorage) (unsafe.Pointer, func(), error) {
	if ptr, _, _ := ms.GPUPtr(); ptr != nil {
		return ptr, func() {}, nil
	}
	// Fallback: upload on the fly.
	// Copy mmap bytes to heap first -- cudaMemcpy from mmap'd pages
	// can fail with misaligned address on ARM64 (GB10/Grace Hopper)
	// when the tensor offset within the GGUF file is not page-aligned.
	mmapBytes := ms.RawBytesGPU()
	rawBytes := make([]byte, len(mmapBytes))
	copy(rawBytes, mmapBytes)
	devPtr, err := e.pool.Alloc(e.deviceID, len(rawBytes))
	if err != nil {
		return nil, nil, err
	}
	cleanup := func() { e.pool.Free(e.deviceID, devPtr, len(rawBytes)) }
	if err := e.runtime.Memcpy(devPtr, unsafe.Pointer(&rawBytes[0]), len(rawBytes), gpuapi.MemcpyHostToDevice); err != nil {
		cleanup()
		return nil, nil, err
	}
	return devPtr, cleanup, nil
}

// Sync synchronizes the GPU stream, blocking until all enqueued operations complete.
// Use for benchmarking or when explicit synchronization is needed.
func (e *GPUEngine[T]) Sync() error {
	if e.stream != nil {
		return e.stream.Synchronize()
	}

	return nil
}

// Static type assertion: GPUEngine satisfies Engine.
var _ Engine[float32] = (*GPUEngine[float32])(nil)

package cublas

import (
	"fmt"
	"os"
	"sync"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// Available returns true if the cuBLAS library can be loaded at runtime.
// The result is cached after the first call.
func Available() bool {
	_, err := getCublasLib()
	return err == nil
}

// CudaDataType identifies the element data type for cublasGemmEx.
type CudaDataType int

const (
	CudaR32F     CudaDataType = 0  // CUDA_R_32F  (float32)
	CudaR16F     CudaDataType = 2  // CUDA_R_16F  (float16)
	CudaR16BF    CudaDataType = 14 // CUDA_R_16BF (bfloat16)
	CudaR8F_E4M3 CudaDataType = 28 // CUDA_R_8F_E4M3 (fp8 e4m3)
)

// CublasComputeType identifies the compute precision for cublasGemmEx.
type CublasComputeType int

const (
	CublasCompute32F CublasComputeType = 68 // CUBLAS_COMPUTE_32F
)

// cuBLAS math-mode constants (cublasMath_t). Only the modes this package
// uses are declared; see NVIDIA cuBLAS docs for the full enum.
const (
	// CublasDefaultMath is cuBLAS's default: on Ampere+ GPUs (including the
	// GB10) this permits TF32 tensor-core downcasting for float32 GEMMs.
	CublasDefaultMath = 0 // CUBLAS_DEFAULT_MATH
	// CublasPedanticMath disables TF32 downcasting and other
	// reduced-precision/algorithm-selection paths; NVIDIA's cuBLAS
	// documentation describes it as producing reproducible results for a
	// fixed problem size and GPU. Used by ZTENSOR_DETERMINISTIC=1
	// (internal/cuda.DeterministicEnabled).
	CublasPedanticMath = 2 // CUBLAS_PEDANTIC_MATH
)

// cuBLAS status codes.
const cublasStatusSuccess = 0

// cublasLib holds dlopen function pointers for cuBLAS.
type cublasLib struct {
	create              uintptr // cublasCreate_v2
	destroy             uintptr // cublasDestroy_v2
	setStream           uintptr // cublasSetStream_v2
	sgemm               uintptr // cublasSgemm_v2
	gemmEx              uintptr // cublasGemmEx
	sgemmStridedBatched uintptr // cublasSgemmStridedBatched
	setMathMode         uintptr // cublasSetMathMode (optional, best-effort)
}

var (
	cblasLib     *cublasLib
	cblasOnce    sync.Once
	cblasLoadErr error
)

// cuBLAS library paths to try.
var cublasLibPaths = []string{
	"libcublas.so.12",
	"libcublas.so",
}

func loadCublas() (*cublasLib, error) {
	var handle uintptr
	var lastErr string
	for _, path := range cublasLibPaths {
		var err error
		handle, err = cuda.DlopenPath(path)
		if err == nil {
			break
		}
		lastErr = err.Error()
	}
	if handle == 0 {
		return nil, fmt.Errorf("cublas: dlopen failed: %s", lastErr)
	}

	lib := &cublasLib{}
	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"cublasCreate_v2", &lib.create},
		{"cublasDestroy_v2", &lib.destroy},
		{"cublasSetStream_v2", &lib.setStream},
		{"cublasSgemm_v2", &lib.sgemm},
		{"cublasGemmEx", &lib.gemmEx},
		{"cublasSgemmStridedBatched", &lib.sgemmStridedBatched},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("cublas: %w", err)
		}
		*s.ptr = addr
	}

	// cublasSetMathMode is resolved best-effort: it backs the debug-only
	// ZTENSOR_DETERMINISTIC mode (T4.1), so a cuBLAS build stripped of it
	// must not break BLAS loading for everyone else. lib.setMathMode stays
	// zero if the symbol is unavailable; SetMathMode reports that as an
	// error at call time.
	if addr, err := cuda.Dlsym(handle, "cublasSetMathMode"); err == nil {
		lib.setMathMode = addr
	}

	return lib, nil
}

func getCublasLib() (*cublasLib, error) {
	cblasOnce.Do(func() {
		cblasLib, cblasLoadErr = loadCublas()
	})
	return cblasLib, cblasLoadErr
}

// Handle wraps a cuBLAS handle (opaque pointer).
type Handle struct {
	ptr uintptr // cublasHandle_t is a pointer
}

// Ptr returns the raw cuBLAS handle pointer for passing to C functions
// (e.g., the fused encoder kernel orchestrator). Same pre-existing
// uintptr<->unsafe.Pointer wrapping pattern used throughout the purego
// bindings (internal/cuda/runtime_purego.go and siblings); not specific to
// T4.1, just newly visible once package-scoped linting was fixed to load
// full packages instead of individual staged files.
func (h *Handle) Ptr() unsafe.Pointer { return unsafe.Pointer(h.ptr) } //nolint:govet

// CreateHandle creates a new cuBLAS context handle.
//
// Under ZTENSOR_DETERMINISTIC=1 (internal/cuda.DeterministicEnabled), this
// also sets CUBLAS_WORKSPACE_CONFIG (if the process has not already set one)
// before creating the handle, and CUBLAS_PEDANTIC_MATH on the handle after
// creation -- the two documented levers for bit-reproducible cuBLAS GEMMs.
// Both are best-effort: a workspace-config env var set too late (cuBLAS
// reads it once, at the FIRST handle creation in the process) or a cuBLAS
// build missing cublasSetMathMode only degrades determinism guarantees and
// is reported via a stderr warning, never a hard failure -- this handle is
// still needed for ordinary (non-deterministic-mode) training and inference.
func CreateHandle() (*Handle, error) {
	lib, err := getCublasLib()
	if err != nil {
		return nil, err
	}
	if cuda.DeterministicEnabled() {
		ensureDeterministicWorkspaceConfig()
	}
	var h uintptr
	status := cuda.Ccall(lib.create, uintptr(unsafe.Pointer(&h)))
	if status != cublasStatusSuccess {
		return nil, fmt.Errorf("cublasCreate failed with status %d", status)
	}
	handle := &Handle{ptr: h}
	if cuda.DeterministicEnabled() {
		if merr := SetMathMode(handle, CublasPedanticMath); merr != nil {
			determinismWarnFn("cublasSetMathMode(CUBLAS_PEDANTIC_MATH) unavailable: %v -- "+
				"this handle's GEMM determinism relies on CUBLAS_WORKSPACE_CONFIG alone", merr)
		}
	}
	return handle, nil
}

// determinismWarnFn sinks ZTENSOR_DETERMINISTIC setup warnings (workspace
// config set late, math mode unavailable). Tests swap it to capture output;
// mirrors internal/cuda's arenaPoisonWarnFn pattern.
var determinismWarnFn = func(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "ztensor cublas determinism: "+format+"\n", args...)
}

var workspaceConfigOnce sync.Once

// ensureDeterministicWorkspaceConfig sets CUBLAS_WORKSPACE_CONFIG to a fixed,
// bounded value if the process has not already set one. cuBLAS reads this
// variable once, the first time a cuBLAS context is created in the process,
// to restrict the workspace it may allocate for a GEMM -- which forecloses
// certain heuristically-chosen, workspace-dependent split-K/atomics
// reduction algorithms for large-K problems. This is the same variable
// PyTorch's determinism docs require for cuBLAS. Setting it here (at first
// CreateHandle, rather than at process start) is a best-effort convenience:
// it is NOT guaranteed to take effect if any other code path created a
// cuBLAS handle earlier in the process. Prefer setting
// CUBLAS_WORKSPACE_CONFIG in the environment before the process starts for
// a guaranteed effect.
func ensureDeterministicWorkspaceConfig() {
	workspaceConfigOnce.Do(func() {
		if os.Getenv("CUBLAS_WORKSPACE_CONFIG") == "" {
			_ = os.Setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
			determinismWarnFn("ZTENSOR_DETERMINISTIC=1: CUBLAS_WORKSPACE_CONFIG was unset; " +
				"set :4096:8 for this process. For a guaranteed effect, set it in the " +
				"environment before the process starts instead.")
		}
	})
}

// SetMathMode sets the cuBLAS math mode for handle h (cublasSetMathMode).
// ZTENSOR_DETERMINISTIC=1 uses this to request CUBLAS_PEDANTIC_MATH. Returns
// an error if the loaded cuBLAS build does not export cublasSetMathMode
// (very old builds) or the call itself fails; callers treat that as
// best-effort and warn rather than fail handle creation.
func SetMathMode(h *Handle, mode int) error {
	if h == nil {
		return fmt.Errorf("cublasSetMathMode: nil handle")
	}
	lib, err := getCublasLib()
	if err != nil {
		return err
	}
	if lib.setMathMode == 0 {
		return fmt.Errorf("cublasSetMathMode: symbol not available in loaded cuBLAS")
	}
	status := cuda.Ccall(lib.setMathMode, h.ptr, uintptr(mode))
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasSetMathMode failed with status %d", status)
	}
	return nil
}

// Destroy releases the cuBLAS handle resources.
func (h *Handle) Destroy() error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}
	status := cuda.Ccall(lib.destroy, h.ptr)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasDestroy failed with status %d", status)
	}
	return nil
}

// SetStream associates a CUDA stream with this cuBLAS handle.
func (h *Handle) SetStream(streamPtr unsafe.Pointer) error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}
	status := cuda.Ccall(lib.setStream, h.ptr, uintptr(streamPtr))
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasSetStream failed with status %d", status)
	}
	return nil
}

// cuBLAS operation constants.
const cublasOpN = 0 // CUBLAS_OP_N
const cublasOpT = 1 // CUBLAS_OP_T

// Sgemm performs single-precision general matrix multiplication.
// Row-major to column-major conversion: swap A/B and m/n.
func Sgemm(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, b unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	if h == nil {
		return fmt.Errorf("cublasSgemm: nil handle")
	}
	if a == nil {
		return fmt.Errorf("cublasSgemm: nil pointer for matrix A")
	}
	if b == nil {
		return fmt.Errorf("cublasSgemm: nil pointer for matrix B")
	}
	if c == nil {
		return fmt.Errorf("cublasSgemm: nil pointer for matrix C")
	}

	lib, err := getCublasLib()
	if err != nil {
		return err
	}

	// cuBLAS Sgemm takes pointers to alpha and beta.
	cAlpha := alpha
	cBeta := beta

	// Row-major to column-major: swap A<->B, swap m<->n.
	// cublasSgemm_v2(handle, transB, transA, n, m, k, &alpha, B, n, A, k, &beta, C, n)
	status := cuda.Ccall(lib.sgemm,
		h.ptr,
		uintptr(cublasOpN), // transa (for B)
		uintptr(cublasOpN), // transb (for A)
		uintptr(n),         // rows of op(B) = cols of C
		uintptr(m),         // cols of op(A) = rows of C
		uintptr(k),         // inner dimension
		uintptr(unsafe.Pointer(&cAlpha)),
		uintptr(b), // B first
		uintptr(n), // ldb
		uintptr(a), // A second
		uintptr(k), // lda
		uintptr(unsafe.Pointer(&cBeta)),
		uintptr(c),
		uintptr(n), // ldc
	)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasSgemm failed with status %d", status)
	}
	return nil
}

// SgemmNT performs single-precision C = A * B^T where A is [m, k] and
// B is [n, k] (row-major). Uses CUBLAS_OP_T on the first cuBLAS argument.
func SgemmNT(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, b unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}

	cAlpha := alpha
	cBeta := beta

	// Row-major to column-major: B comes first with CUBLAS_OP_T, A second with CUBLAS_OP_N.
	status := cuda.Ccall(lib.sgemm,
		h.ptr,
		uintptr(cublasOpT), // transpose B (cuBLAS first arg)
		uintptr(cublasOpN), // no-transpose A (cuBLAS second arg)
		uintptr(n),         // rows of op(B) = n
		uintptr(m),         // cols of op(A) = m
		uintptr(k),         // inner dimension
		uintptr(unsafe.Pointer(&cAlpha)),
		uintptr(b), // B first (cuBLAS convention)
		uintptr(k), // ldb = k (B_rm row width)
		uintptr(a), // A second
		uintptr(k), // lda = k (A_rm row width)
		uintptr(unsafe.Pointer(&cBeta)),
		uintptr(c),
		uintptr(n), // ldc = n (C_rm row width)
	)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasSgemm(NT) failed with status %d", status)
	}
	return nil
}

// SgemmStridedBatched performs batched single-precision GEMM with strided access.
// Row-major to column-major conversion: swap A/B and m/n (same trick as Sgemm).
//
// Parameters (in row-major terms):
//
//	m        - rows of A and C per batch
//	n        - columns of B and C per batch
//	k        - columns of A / rows of B
//	alpha    - scalar multiplier for A*B
//	a        - device pointer to A[0] (m x k, row-major)
//	strideA  - element stride between consecutive A matrices
//	b        - device pointer to B[0] (k x n, row-major)
//	strideB  - element stride between consecutive B matrices
//	beta     - scalar multiplier for C
//	c        - device pointer to C[0] (m x n, row-major), output
//	strideC  - element stride between consecutive C matrices
//	batch    - number of matrices in the batch
func SgemmStridedBatched(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, strideA int64,
	b unsafe.Pointer, strideB int64,
	beta float32,
	c unsafe.Pointer, strideC int64,
	batch int,
) error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}

	cAlpha := alpha
	cBeta := beta

	// Row-major to column-major: swap A<->B, swap m<->n, swap strides.
	// cublasSgemmStridedBatched(handle, transB, transA, n, m, k,
	//   &alpha, B, n, strideB, A, k, strideA, &beta, C, n, strideC, batchCount)
	status := cuda.Ccall(lib.sgemmStridedBatched,
		h.ptr,
		uintptr(cublasOpN), // transa (for B)
		uintptr(cublasOpN), // transb (for A)
		uintptr(n),         // rows of op(B) = cols of C
		uintptr(m),         // cols of op(A) = rows of C
		uintptr(k),         // inner dimension
		uintptr(unsafe.Pointer(&cAlpha)),
		uintptr(b),       // B first
		uintptr(n),       // ldb
		uintptr(strideB), // strideB
		uintptr(a),       // A second
		uintptr(k),       // lda
		uintptr(strideA), // strideA
		uintptr(unsafe.Pointer(&cBeta)),
		uintptr(c),
		uintptr(n),       // ldc
		uintptr(strideC), // strideC
		uintptr(batch),   // batchCount
	)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasSgemmStridedBatched failed with status %d", status)
	}
	return nil
}

// SgemmNTStridedBatched performs batched C = A * B^T using strided batched GEMM
// with CUBLAS_OP_T on the B operand.
func SgemmNTStridedBatched(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, strideA int64,
	b unsafe.Pointer, strideB int64,
	beta float32,
	c unsafe.Pointer, strideC int64,
	batch int,
) error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}

	cAlpha := alpha
	cBeta := beta

	// Row-major to column-major: B with OP_T first, A with OP_N second.
	status := cuda.Ccall(lib.sgemmStridedBatched,
		h.ptr,
		uintptr(cublasOpT), // transpose B (cuBLAS first arg)
		uintptr(cublasOpN), // no-transpose A (cuBLAS second arg)
		uintptr(n),         // rows of op(B) = n
		uintptr(m),         // cols of op(A) = m
		uintptr(k),         // inner dimension
		uintptr(unsafe.Pointer(&cAlpha)),
		uintptr(b),       // B first
		uintptr(k),       // ldb = k (B_rm row width)
		uintptr(strideB), // strideB
		uintptr(a),       // A second
		uintptr(k),       // lda = k (A_rm row width)
		uintptr(strideA), // strideA
		uintptr(unsafe.Pointer(&cBeta)),
		uintptr(c),
		uintptr(n),       // ldc = n (C_rm row width)
		uintptr(strideC), // strideC
		uintptr(batch),   // batchCount
	)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasSgemmNTStridedBatched failed with status %d", status)
	}
	return nil
}

// cublasGemmDefault is the CUBLAS_GEMM_DEFAULT algorithm selector.
// The C enum value is -1; as an unsigned 32-bit integer this is 0xFFFFFFFF.
const cublasGemmDefault uintptr = 0xFFFFFFFF

// GemmEx performs mixed-precision general matrix multiplication.
// Row-major to column-major conversion: swap A/B and m/n.
func GemmEx(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, aType CudaDataType,
	b unsafe.Pointer, bType CudaDataType,
	beta float32,
	c unsafe.Pointer, cType CudaDataType,
	computeType CublasComputeType,
) error {
	if h == nil {
		return fmt.Errorf("cublasGemmEx: nil handle")
	}
	if a == nil {
		return fmt.Errorf("cublasGemmEx: nil pointer for matrix A")
	}
	if b == nil {
		return fmt.Errorf("cublasGemmEx: nil pointer for matrix B")
	}
	if c == nil {
		return fmt.Errorf("cublasGemmEx: nil pointer for matrix C")
	}

	lib, err := getCublasLib()
	if err != nil {
		return err
	}

	cAlpha := alpha
	cBeta := beta

	// Row-major to column-major: swap A<->B, swap m<->n.
	// cublasGemmEx(handle, transa, transb, m, n, k,
	//   alpha, B, Btype, ldb, A, Atype, lda,
	//   beta, C, Ctype, ldc, computeType, algo)
	status := cuda.Ccall(lib.gemmEx,
		h.ptr,
		uintptr(cublasOpN), // transa (for B)
		uintptr(cublasOpN), // transb (for A)
		uintptr(n),         // rows of op(B) = cols of C
		uintptr(m),         // cols of op(A) = rows of C
		uintptr(k),         // inner dimension
		uintptr(unsafe.Pointer(&cAlpha)),
		uintptr(b),     // B first
		uintptr(bType), // Btype
		uintptr(n),     // ldb
		uintptr(a),     // A second
		uintptr(aType), // Atype
		uintptr(k),     // lda
		uintptr(unsafe.Pointer(&cBeta)),
		uintptr(c),
		uintptr(cType),       // Ctype
		uintptr(n),           // ldc
		uintptr(computeType), // computeType
		cublasGemmDefault,    // algo
	)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasGemmEx failed with status %d", status)
	}
	return nil
}

// bf16GemmEx is the shared core for the bf16 transpose-variant GEMMs. It issues
// a single cublasGemmEx with bf16 A/B/C operands and FP32 accumulation
// (CUBLAS_COMPUTE_32F), using the row-major->column-major swap convention of the
// other GEMMs in this file (B is the first cuBLAS operand, A the second).
//
// transFirst/transSecond are the cuBLAS ops applied to the first (B) and second
// (A) operands respectively; ldFirst/ldSecond/ldC are the leading dimensions of
// the stored buffers. dim1/dim2/dim3 are the cuBLAS (rows-of-Ccm, cols-of-Ccm,
// inner) triple, which equal (n, m, k) for every variant here.
func bf16GemmEx(h *Handle, transFirst, transSecond uintptr,
	dim1, dim2, dim3 int, alpha float32,
	first unsafe.Pointer, ldFirst int,
	second unsafe.Pointer, ldSecond int,
	beta float32, c unsafe.Pointer, ldC int,
	errLabel string,
) error {
	if h == nil {
		return fmt.Errorf("%s: nil handle", errLabel)
	}
	if first == nil || second == nil || c == nil {
		return fmt.Errorf("%s: nil matrix pointer", errLabel)
	}
	lib, err := getCublasLib()
	if err != nil {
		return err
	}
	cAlpha := alpha
	cBeta := beta
	status := cuda.Ccall(lib.gemmEx,
		h.ptr,
		transFirst,  // op on first operand (B)
		transSecond, // op on second operand (A)
		uintptr(dim1),
		uintptr(dim2),
		uintptr(dim3),
		uintptr(unsafe.Pointer(&cAlpha)),
		uintptr(first),
		uintptr(CudaR16BF),
		uintptr(ldFirst),
		uintptr(second),
		uintptr(CudaR16BF),
		uintptr(ldSecond),
		uintptr(unsafe.Pointer(&cBeta)),
		uintptr(c),
		uintptr(CudaR16BF),
		uintptr(ldC),
		uintptr(CublasCompute32F),
		cublasGemmDefault,
	)
	if status != cublasStatusSuccess {
		return fmt.Errorf("%s failed with status %d", errLabel, status)
	}
	return nil
}

// BFloat16GemmNT performs bf16 C = alpha * A * B^T + beta * C where A is [m, k]
// and B is [n, k] (row-major), accumulating in FP32. Mirrors SgemmNT: B is the
// first cuBLAS operand with CUBLAS_OP_T (ldb=k), A the second with CUBLAS_OP_N
// (lda=k), output dims (n, m, k), ldc=n. Avoids an explicit transpose of B.
func BFloat16GemmNT(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, b unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	// first=B OP_T ldb=k, second=A OP_N lda=k, dims (n, m, k), ldc=n.
	return bf16GemmEx(h, cublasOpT, cublasOpN, n, m, k, alpha,
		b, k, a, k, beta, c, n, "cublasGemmEx(bf16 NT)")
}

// BFloat16GemmTN performs bf16 C = alpha * A^T * B + beta * C where A is [k, m]
// and B is [k, n] (row-major), accumulating in FP32. This is the dW gradient
// shape (X^T * dY). B is the first cuBLAS operand with CUBLAS_OP_N (ldb=n), A
// the second with CUBLAS_OP_T (lda=m), output dims (n, m, k), ldc=n. Avoids an
// explicit transpose of A.
func BFloat16GemmTN(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, b unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	// first=B OP_N ldb=n, second=A OP_T lda=m, dims (n, m, k), ldc=n.
	return bf16GemmEx(h, cublasOpN, cublasOpT, n, m, k, alpha,
		b, n, a, m, beta, c, n, "cublasGemmEx(bf16 TN)")
}

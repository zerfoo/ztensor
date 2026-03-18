// fp8_gemm.cu -- FP8 E4M3 GEMM kernel using cublasLt.
// Requires sm_89+ (Ada Lovelace) for native FP8 matmul support.
// Input: A[M,K] FP8E4M3, B[K,N] FP8E4M3, scaleA, scaleB (floats).
// Output: C[M,N] FP16 (dequantized result).
// Compiled by nvcc into libkernels.so.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <string.h>

// ---------- bits_to_float ----------

static inline float bits_to_float(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------- Capability check ----------

extern "C" int fp8_gemm_check_sm89() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return 0;

    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

    // sm_89 = Ada Lovelace (RTX 4090, L4, L40). FP8 tensor cores require sm_89+.
    return (major > 8 || (major == 8 && minor >= 9)) ? 1 : 0;
}

// ---------- cublasLt FP8 GEMM launcher ----------
//
// Computes C = alpha * (A * B), where:
//   A is [M, K] in row-major FP8 E4M3
//   B is [K, N] in row-major FP8 E4M3
//   C is [M, N] in row-major FP16
//   alpha = scaleA * scaleB (combined dequantization scale)
//
// cublasLt expects column-major, so we compute C^T = B^T * A^T
// which gives us the correct row-major result.

extern "C" int launch_fp8_gemm(
    const void* A, const void* B, void* C,
    int M, int K, int N,
    unsigned int scale_a_bits, unsigned int scale_b_bits,
    void* stream_ptr)
{
    // Runtime sm_89+ check.
    if (!fp8_gemm_check_sm89()) {
        return -1; // Not supported.
    }

    cublasLtHandle_t ltHandle;
    cublasStatus_t stat = cublasLtCreate(&ltHandle);
    if (stat != CUBLAS_STATUS_SUCCESS) return (int)stat;

    // Combined scale: alpha = scaleA * scaleB.
    float scaleA = bits_to_float(scale_a_bits);
    float scaleB = bits_to_float(scale_b_bits);
    float alpha = scaleA * scaleB;
    float beta = 0.0f;

    // Matrix descriptors.
    // Row-major A[M,K] in FP8 -> column-major A^T[K,M].
    // Row-major B[K,N] in FP8 -> column-major B^T[N,K].
    // We compute C^T = B^T * A^T -> [N,K] x [K,M] = [N,M].
    // C^T[N,M] column-major = C[M,N] row-major.

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    // Create operation descriptor with FP32 compute and FP32 scale type.
    stat = cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (stat != CUBLAS_STATUS_SUCCESS) { cublasLtDestroy(ltHandle); return (int)stat; }

    // Set transpose operations: both A^T and B^T.
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // B^T layout: B is row-major [K,N], so column-major with ld=N, rows=K, cols=N.
    // After transpose: [N,K].
    stat = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, N, N);
    if (stat != CUBLAS_STATUS_SUCCESS) { cublasLtMatmulDescDestroy(operationDesc); cublasLtDestroy(ltHandle); return (int)stat; }

    // A^T layout: A is row-major [M,K], so column-major with ld=K, rows=M, cols=K.
    // After transpose: [K,M].
    stat = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, M, K, K);
    if (stat != CUBLAS_STATUS_SUCCESS) { cublasLtMatrixLayoutDestroy(Adesc); cublasLtMatmulDescDestroy(operationDesc); cublasLtDestroy(ltHandle); return (int)stat; }

    // C layout: column-major [N,M] with ld=N = row-major [M,N].
    stat = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, N, M, N);
    if (stat != CUBLAS_STATUS_SUCCESS) { cublasLtMatrixLayoutDestroy(Bdesc); cublasLtMatrixLayoutDestroy(Adesc); cublasLtMatmulDescDestroy(operationDesc); cublasLtDestroy(ltHandle); return (int)stat; }

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Execute matmul: C^T = alpha * B^T * A^T + beta * C^T.
    stat = cublasLtMatmul(
        ltHandle,
        operationDesc,
        &alpha, B, Adesc,    // B^T (transposed via desc)
                A, Bdesc,    // A^T (transposed via desc)
        &beta,  C, Cdesc,
                C, Cdesc,
        NULL,    // algo (NULL = heuristic)
        NULL, 0, // workspace
        stream);

    // Cleanup.
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);

    if (stat != CUBLAS_STATUS_SUCCESS) return (int)stat;

    return (int)cudaGetLastError();
}

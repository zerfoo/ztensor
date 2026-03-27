/* Ternary GEMV kernel interface.
 *
 * Ternary weights are packed 2 bits per value in row-major order:
 *   00 = -1, 01 = 0, 10 = +1
 *
 * Computes: y[m] = sum_k( decode(W_ternary[m,k]) * x[k] )
 * No floating-point multiply — only additions and subtractions.
 *
 * W_ternary: device pointer to packed 2-bit ternary data [M, K].
 * x: device pointer to [K] FP32 input vector.
 * y: device pointer to [M] FP32 output vector.
 * Batch=1 only (GEMV, not GEMM).
 */
#ifndef TERNARY_GEMV_H
#define TERNARY_GEMV_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ternary_gemv_f32 performs ternary GEMV:
 *   y[m] = sum of x[k] where W[m,k]=+1, minus sum of x[k] where W[m,k]=-1
 *
 * W_ternary: device pointer to packed 2-bit ternary weights [M, K].
 *            4 values per byte, row-major. Encoding: 00=-1, 01=0, 10=+1.
 * x:         device pointer to [K] float32 input vector.
 * y:         device pointer to [M] float32 output vector.
 * M, K:      matrix dimensions.
 * stream:    CUDA stream.
 */
cudaError_t ternary_gemv_f32(
    const void* W_ternary,
    const float* x, float* y,
    int M, int K,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_GEMV_H */

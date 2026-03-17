/* Q5_0 fused dequant-GEMV kernel interface.
 *
 * Q5_0 block format (22 bytes per 32 values):
 *   - 2 bytes: fp16 d (block scale)
 *   - 4 bytes: uint32 qh (32 high bits, one per element)
 *   - 16 bytes: qs (packed nibbles, two 4-bit values per byte)
 *
 * Computes: y[m] = sum_k( dequant(W_q5_0[m,k]) * x[k] )
 * W_q5_0 is raw Q5_0 blocks laid out row-major.
 * x is [K] FP32 input vector. y is [M] FP32 output vector.
 * Batch=1 only (GEMV, not GEMM).
 */
#ifndef GEMV_Q5_0_H
#define GEMV_Q5_0_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* gemv_q5_0_f32 performs Q5_0 fused dequant-GEMV:
 *   y[m] = sum_k( dequant(W_q5_0[m,k]) * x[k] )
 *
 * W_q5_0: device pointer to raw Q5_0 blocks for matrix W [M, K].
 *         M * ceil(K/32) blocks, each 22 bytes. Row-major layout.
 * x:      device pointer to [K] float input vector.
 * y:      device pointer to [M] float output vector.
 * M, K:   matrix dimensions. K must be a multiple of 32.
 * stream: CUDA stream.
 */
cudaError_t gemv_q5_0_f32(
    const void* W_q5_0, const float* x, float* y,
    int M, int K,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GEMV_Q5_0_H */

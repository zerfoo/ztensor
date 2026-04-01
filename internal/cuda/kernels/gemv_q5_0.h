/* Q5_0 fused dequant-GEMV kernel interface (separated GPU layout).
 *
 * GPU layout (from Q5_0Storage.RawBytesGPU):
 *   Region 1: [nBlocks * 2 bytes] fp16 scales, padded to 16-byte boundary
 *   Region 2: [nBlocks * 4 bytes] uint32 qh values, padded to 16-byte boundary
 *   Region 3: [nBlocks * 16 bytes] packed nibbles (qs)
 *
 * Computes: y[m] = sum_k( dequant(W_q5_0[m,k]) * x[k] )
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
 * W_q5_0:   device pointer to separated Q5_0 layout (scales | qh | qs).
 * x:        device pointer to [K] float input vector.
 * y:        device pointer to [M] float output vector.
 * M, K:     matrix dimensions. K must be a multiple of 32.
 * qhOffset: byte offset from W_q5_0 to the qh region.
 * qsOffset: byte offset from W_q5_0 to the qs region.
 * stream:   CUDA stream.
 */
cudaError_t gemv_q5_0_f32(
    const void* W_q5_0, const float* x, float* y,
    int M, int K,
    int qhOffset, int qsOffset,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GEMV_Q5_0_H */

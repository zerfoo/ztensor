/* NVFP4 E2M1 fused dequant-GEMV kernel interface.
 *
 * NVFP4 block format (block of 16 values):
 *   - Packed data: 8 bytes (2 FP4 per byte, little-endian nibble order)
 *   - Scale: 1 x float16 per block of 16
 *
 * FP4 E2M1 nibble (4 bits): bit3=sign, bits2:0 = magnitude code.
 * Magnitude LUT: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
 * Dequantized value = (-1)^sign * LUT[code] * scale
 *
 * Computes: y[m] = sum_k( dequant(W_fp4[m,k]) * x[k] )
 * W_fp4 is raw NVFP4 packed data laid out row-major.
 * scales is [M * blocks_per_row] float16 block scales.
 * x is [K] FP16 input vector. y is [M] FP32 output vector.
 * Batch=1 only (GEMV, not GEMM).
 *
 * Requires sm_100+ (Blackwell) for optimal performance.
 */
#ifndef FP4_GEMV_H
#define FP4_GEMV_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fp4_gemv_f16 performs NVFP4 fused dequant-GEMV with FP16 activations:
 *   y[m] = sum_k( dequant(W_fp4[m,k]) * x_fp16[k] )
 *
 * W_fp4:  device pointer to packed NVFP4 data for matrix W [M, K].
 *         M * ceil(K/16) blocks, each 8 bytes packed. Row-major layout.
 * scales: device pointer to [M * ceil(K/16)] float16 block scales.
 * x:      device pointer to [K] float16 input vector.
 * y:      device pointer to [M] float32 output vector.
 * M, K:   matrix dimensions. K must be a multiple of 16.
 * stream: CUDA stream.
 */
cudaError_t fp4_gemv_f16(
    const void* W_fp4, const void* scales,
    const void* x, float* y,
    int M, int K,
    cudaStream_t stream);

/* fp4_gemv_check_sm100 returns 1 if GPU supports sm_100+, 0 otherwise. */
int fp4_gemv_check_sm100(void);

#ifdef __cplusplus
}
#endif

#endif /* FP4_GEMV_H */

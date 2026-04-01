/* Q5_0 fused dequant-GEMV kernel for single-token decode (batch=1).
 *
 * GPU-optimized SEPARATED layout (from Q5_0Storage.RawBytesGPU):
 *   Region 1: [nBlocks * 2 bytes] fp16 scales, padded to 16-byte boundary
 *   Region 2: [nBlocks * 4 bytes] uint32 qh values, padded to 16-byte boundary
 *   Region 3: [nBlocks * 16 bytes] packed nibbles (qs)
 *
 * This layout ensures natural alignment: fp16 at 2-byte, uint32 at 4-byte.
 * Eliminates the byte-wise loads required for the interleaved 22-byte layout
 * on ARM64 Grace Hopper.
 *
 * Dequantization (matching llama.cpp dequantize_row_q5_0):
 *   For j in 0..15:
 *     packed = qs[j]
 *     low4  = packed & 0xF
 *     high4 = packed >> 4
 *     h0 = ((qh >> j)      & 1) << 4
 *     h1 = ((qh >> (j+16)) & 1) << 4
 *     val[j]    = d * float((low4  | h0) - 16)
 *     val[j+16] = d * float((high4 | h1) - 16)
 */

#include "gemv_q5_0.h"
#include <cuda_fp16.h>
#include <stdint.h>

#define Q5_0_BLOCK_SIZE  32
#define Q5_0_WARPS_PER_BLOCK 4
#define Q5_0_WARP_SIZE       32

/* ---------- Fused GEMV kernel (separated GPU layout) ----------
 *
 * y[row] = sum_k dequant(W_q5_0[row, k]) * x[k]
 *
 * W_q5_0 points to the separated layout base. qhOffset and qsOffset
 * are byte offsets to the qh and qs regions respectively.
 */
__global__ void gemv_q5_0_kernel(
    const uint8_t* __restrict__ W_q5_0,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int M, int K,
    int qhOffset, int qsOffset)
{
    extern __shared__ float sx[];

    /* Cooperatively load x into shared memory. */
    int threads_per_block = blockDim.x;
    for (int i = threadIdx.x; i < K; i += threads_per_block) {
        sx[i] = x[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / Q5_0_WARP_SIZE;
    int lane_id = threadIdx.x % Q5_0_WARP_SIZE;
    int row = blockIdx.x * Q5_0_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    int blocks_per_row = K / Q5_0_BLOCK_SIZE;

    /* Pointers to the three separated regions for this row. */
    const __half*    row_scales = (const __half*)(W_q5_0 + row * blocks_per_row * 2);
    const uint32_t*  row_qh    = (const uint32_t*)(W_q5_0 + qhOffset + row * blocks_per_row * 4);
    const uint8_t*   row_qs    = W_q5_0 + qsOffset + (size_t)row * blocks_per_row * 16;

    float acc = 0.0f;

    /* Each lane handles a strided subset of blocks. */
    for (int bi = lane_id; bi < blocks_per_row; bi += Q5_0_WARP_SIZE) {
        /* All loads are naturally aligned in the separated layout. */
        float d = __half2float(__ldg(&row_scales[bi]));
        uint32_t qh = __ldg(&row_qh[bi]);
        const uint8_t* qs = row_qs + bi * 16;
        int k_base = bi * Q5_0_BLOCK_SIZE;

        /* Process 16 packed bytes -> 32 dequantized values. */
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t packed = __ldg(&qs[j]);
            uint8_t low4  = packed & 0xF;
            uint8_t high4 = packed >> 4;

            uint8_t h0 = ((qh >> j) & 1) << 4;
            uint8_t h1 = ((qh >> (j + 16)) & 1) << 4;

            float dq0 = d * (float)((int)(low4  | h0) - 16);
            float dq1 = d * (float)((int)(high4 | h1) - 16);

            acc = __fmaf_rn(dq0, sx[k_base + j], acc);
            acc = __fmaf_rn(dq1, sx[k_base + j + 16], acc);
        }
    }

    /* Warp shuffle reduction. */
    #pragma unroll
    for (int offset = Q5_0_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

/* ---------- Dispatcher ---------- */

extern "C" cudaError_t gemv_q5_0_f32(
    const void* W_q5_0, const float* x, float* y,
    int M, int K,
    int qhOffset, int qsOffset,
    cudaStream_t stream)
{
    if (K % Q5_0_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    int threads = Q5_0_WARPS_PER_BLOCK * Q5_0_WARP_SIZE;  /* 128 */
    int grid = (M + Q5_0_WARPS_PER_BLOCK - 1) / Q5_0_WARPS_PER_BLOCK;
    int smem = K * sizeof(float);

    gemv_q5_0_kernel<<<grid, threads, smem, stream>>>(
        (const uint8_t*)W_q5_0, x, y, M, K, qhOffset, qsOffset);

    return cudaGetLastError();
}

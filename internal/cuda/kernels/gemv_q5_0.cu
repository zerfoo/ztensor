/* Q5_0 fused dequant-GEMV kernel for single-token decode (batch=1).
 *
 * Reads Q5_0 blocks directly, dequantizes in registers (no global
 * memory intermediary), multiplies by the activation vector, and accumulates
 * in FP32. This halves memory traffic compared to separate dequant + GEMV.
 *
 * Q5_0 block (22 bytes, 32 values):
 *   [0:2]   fp16 d      -- block scale
 *   [2:6]   uint32 qh   -- 32 high bits (one per element)
 *   [6:22]  16 bytes qs  -- packed nibbles (two 4-bit values per byte)
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
#define Q5_0_BLOCK_BYTES 22
#define Q5_0_WARPS_PER_BLOCK 4
#define Q5_0_WARP_SIZE       32

/* ---------- Fused GEMV kernel ----------
 *
 * y[row] = sum_k dequant(W_q5_0[row, k]) * x[k]
 *
 * Strategy:
 *   - Load input vector x into shared memory (all threads cooperate).
 *   - One warp per row for simplicity and good occupancy.
 *   - Each lane processes a strided subset of blocks.
 *   - Within each block, 16 packed bytes yield 32 dequantized values.
 *   - Warp shuffle reduction produces the final dot product.
 */
__global__ void gemv_q5_0_kernel(
    const uint8_t* __restrict__ W_q5_0,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int M, int K)
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
    const uint8_t* row_data = W_q5_0 + (size_t)row * blocks_per_row * Q5_0_BLOCK_BYTES;

    float acc = 0.0f;

    /* Each lane handles a strided subset of blocks. */
    for (int bi = lane_id; bi < blocks_per_row; bi += Q5_0_WARP_SIZE) {
        const uint8_t* blk = row_data + bi * Q5_0_BLOCK_BYTES;

        /* Read fp16 d using byte-wise load (ARM64 alignment safety).
         * Q5_0 blocks are 22 bytes — not a multiple of 4, so blk may
         * be misaligned for uint16/uint32 casts after the first block. */
        uint16_t d_bits = (uint16_t)__ldg(&blk[0]) | ((uint16_t)__ldg(&blk[1]) << 8);
        float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));

        /* Read qh (32 high bits) using byte-wise load. */
        uint32_t qh = (uint32_t)__ldg(&blk[2])
                     | ((uint32_t)__ldg(&blk[3]) << 8)
                     | ((uint32_t)__ldg(&blk[4]) << 16)
                     | ((uint32_t)__ldg(&blk[5]) << 24);

        const uint8_t* qs = blk + 6;
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
    cudaStream_t stream)
{
    if (K % Q5_0_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    int threads = Q5_0_WARPS_PER_BLOCK * Q5_0_WARP_SIZE;  /* 128 */
    int grid = (M + Q5_0_WARPS_PER_BLOCK - 1) / Q5_0_WARPS_PER_BLOCK;
    int smem = K * sizeof(float);

    gemv_q5_0_kernel<<<grid, threads, smem, stream>>>(
        (const uint8_t*)W_q5_0, x, y, M, K);

    return cudaGetLastError();
}

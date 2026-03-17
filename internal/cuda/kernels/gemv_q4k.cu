/* Q4_K fused dequant-GEMV kernel for single-token decode (batch=1).
 *
 * Reads Q4_K super-blocks directly, dequantizes in registers (no global
 * memory intermediary), multiplies by the activation vector, and accumulates
 * in FP32. This halves memory traffic compared to separate dequant + GEMV.
 *
 * Q4_K super-block (144 bytes, 256 values):
 *   [0:2]   fp16 d      -- super-block scale
 *   [2:4]   fp16 dmin   -- super-block min
 *   [4:16]  12 bytes    -- packed 6-bit scales/mins for 8 sub-blocks
 *   [16:144] 128 bytes  -- 256 x 4-bit quantized values
 *
 * Dequantization (matching llama.cpp dequantize_row_q4_K):
 *   For each group g (0..3), sub-blocks sb0=2g, sb1=2g+1:
 *     sc0 = d * scales[sb0],  mn0 = dmin * mins[sb0]
 *     sc1 = d * scales[sb1],  mn1 = dmin * mins[sb1]
 *     For each l in 0..31:
 *       val[g*64 + l]    = sc0 * (qdata[g*32+l] & 0xF) - mn0
 *       val[g*64 + l+32] = sc1 * (qdata[g*32+l] >> 4)  - mn1
 */

#include "gemv_q4k.h"
#include <cuda_fp16.h>
#include <stdint.h>

#define Q4K_SUPER_BLOCK_SIZE 256
#define Q4K_BLOCK_BYTES      144
#define Q4K_NUM_SUB_BLOCKS   8
#define Q4K_WARPS_PER_BLOCK  4
#define Q4K_WARP_SIZE        32

/* Decode 6-bit scales and mins from the 12-byte packed region.
 * Matches tensor/quantized_kquant.go decodeQ4KScalesMins. */
__device__ __forceinline__ void decode_scales_mins(
    const uint8_t* sc,
    float d, float dmin,
    float* __restrict__ sub_scales,
    float* __restrict__ sub_mins)
{
    /* Sub-blocks 0-3: 6 low bits. */
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sub_scales[i] = d * (float)(sc[i] & 63);
        sub_mins[i]   = dmin * (float)(sc[4+i] & 63);
    }
    /* Sub-blocks 4-7: 4 bits from bytes 8-11 + 2 high bits. */
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sub_scales[4+i] = d * (float)((sc[8+i] & 0xF) | ((sc[i] >> 6) << 4));
        sub_mins[4+i]   = dmin * (float)((sc[8+i] >> 4) | ((sc[4+i] >> 6) << 4));
    }
}

/* ---------- Fused GEMV kernel ----------
 *
 * y[row] = sum_k dequant(W_q4k[row, k]) * x[k]
 *
 * Strategy:
 *   - Load input vector x into shared memory (all threads cooperate).
 *   - One warp per row for simplicity and good occupancy.
 *   - Each lane processes a strided subset of super-blocks.
 *   - Within each super-block, 4 groups of 64 elements are processed.
 *   - Warp shuffle reduction produces the final dot product.
 */
__global__ void gemv_q4k_kernel(
    const uint8_t* __restrict__ W_q4k,
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

    int warp_id = threadIdx.x / Q4K_WARP_SIZE;
    int lane_id = threadIdx.x % Q4K_WARP_SIZE;
    int row = blockIdx.x * Q4K_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    int blocks_per_row = K / Q4K_SUPER_BLOCK_SIZE;
    const uint8_t* row_data = W_q4k + (size_t)row * blocks_per_row * Q4K_BLOCK_BYTES;

    float acc = 0.0f;

    /* Each lane handles a strided subset of super-blocks. */
    for (int bi = lane_id; bi < blocks_per_row; bi += Q4K_WARP_SIZE) {
        const uint8_t* blk = row_data + bi * Q4K_BLOCK_BYTES;

        /* Read fp16 d and dmin. */
        float d    = __half2float(__ldg((const __half*)(blk)));
        float dmin = __half2float(__ldg((const __half*)(blk + 2)));

        /* Decode sub-block scales and mins. */
        float sub_scales[Q4K_NUM_SUB_BLOCKS];
        float sub_mins[Q4K_NUM_SUB_BLOCKS];
        decode_scales_mins(blk + 4, d, dmin, sub_scales, sub_mins);

        const uint8_t* qdata = blk + 16;
        int k_base = bi * Q4K_SUPER_BLOCK_SIZE;

        /* Process 4 groups of 64 elements each. */
        #pragma unroll
        for (int group = 0; group < 4; group++) {
            int sb0 = group * 2;
            int sb1 = group * 2 + 1;
            float sc0 = sub_scales[sb0];
            float mn0 = sub_mins[sb0];
            float sc1 = sub_scales[sb1];
            float mn1 = sub_mins[sb1];

            int base_out = k_base + group * 64;
            int base_q = group * 32;

            #pragma unroll
            for (int l = 0; l < 32; l++) {
                uint8_t q = __ldg(&qdata[base_q + l]);

                float dq_lo = sc0 * (float)(q & 0xF) - mn0;
                float dq_hi = sc1 * (float)(q >> 4)   - mn1;

                acc = __fmaf_rn(dq_lo, sx[base_out + l], acc);
                acc = __fmaf_rn(dq_hi, sx[base_out + l + 32], acc);
            }
        }
    }

    /* Warp shuffle reduction. */
    #pragma unroll
    for (int offset = Q4K_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

/* ---------- Dispatcher ---------- */

extern "C" cudaError_t gemv_q4k_f32(
    const void* W_q4k, const float* x, float* y,
    int M, int K,
    cudaStream_t stream)
{
    if (K % Q4K_SUPER_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    int threads = Q4K_WARPS_PER_BLOCK * Q4K_WARP_SIZE;  /* 128 */
    int grid = (M + Q4K_WARPS_PER_BLOCK - 1) / Q4K_WARPS_PER_BLOCK;
    int smem = K * sizeof(float);

    gemv_q4k_kernel<<<grid, threads, smem, stream>>>(
        (const uint8_t*)W_q4k, x, y, M, K);

    return cudaGetLastError();
}

/* ========== dp4a INT8 dot-product variant ==========
 *
 * Uses __dp4a(a, b, c) = c + dot4(a_int8, b_int8) for 4 MACs/instruction.
 *
 * Reformulation for Q4_K + dp4a:
 *   dequant(q) * x = scale * nibble * x - min * x
 *   sum(dequant(q[i]) * x[i]) = scale * sum(nibble[i] * x[i]) - min * sum(x[i])
 *
 * The nibble*x dot product uses dp4a: quantize 4 consecutive x values to INT8
 * on the fly (per 4-element group), pack nibbles and q8_x into int32, dp4a.
 * The min*sum(x) correction uses FP32 accumulation of x from shared memory.
 *
 * Key: no large INT8 arrays — quantize x 4 elements at a time to minimize
 * register pressure. The x quantization scale is per 4-element group.
 */

/* Pack 4 byte values into one int32 for dp4a (little-endian). */
__device__ __forceinline__ int pack4_i8(int a, int b, int c, int d)
{
    return (int)((unsigned)(a & 0xFF) | ((unsigned)(b & 0xFF) << 8) |
                 ((unsigned)(c & 0xFF) << 16) | ((unsigned)(d & 0xFF) << 24));
}

__global__ void gemv_q4k_dp4a_kernel(
    const uint8_t* __restrict__ W_q4k,
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

    int warp_id = threadIdx.x / Q4K_WARP_SIZE;
    int lane_id = threadIdx.x % Q4K_WARP_SIZE;
    int row = blockIdx.x * Q4K_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    int blocks_per_row = K / Q4K_SUPER_BLOCK_SIZE;
    const uint8_t* row_data = W_q4k + (size_t)row * blocks_per_row * Q4K_BLOCK_BYTES;

    float acc = 0.0f;

    for (int bi = lane_id; bi < blocks_per_row; bi += Q4K_WARP_SIZE) {
        const uint8_t* blk = row_data + bi * Q4K_BLOCK_BYTES;

        float d    = __half2float(__ldg((const __half*)(blk)));
        float dmin = __half2float(__ldg((const __half*)(blk + 2)));

        float sub_scales[Q4K_NUM_SUB_BLOCKS];
        float sub_mins[Q4K_NUM_SUB_BLOCKS];
        decode_scales_mins(blk + 4, d, dmin, sub_scales, sub_mins);

        const uint8_t* qdata = blk + 16;
        int k_base = bi * Q4K_SUPER_BLOCK_SIZE;

        #pragma unroll
        for (int group = 0; group < 4; group++) {
            float sc0 = sub_scales[group * 2];
            float mn0 = sub_mins[group * 2];
            float sc1 = sub_scales[group * 2 + 1];
            float mn1 = sub_mins[group * 2 + 1];

            int base_out = k_base + group * 64;
            int base_q = group * 32;

            float sum_x_lo = 0.0f, sum_x_hi = 0.0f;

            #pragma unroll
            for (int l = 0; l < 32; l += 4) {
                uint8_t q0 = __ldg(&qdata[base_q + l]);
                uint8_t q1 = __ldg(&qdata[base_q + l + 1]);
                uint8_t q2 = __ldg(&qdata[base_q + l + 2]);
                uint8_t q3 = __ldg(&qdata[base_q + l + 3]);

                /* Load 4 x values for lo nibble (sub-block 0). */
                float v0 = sx[base_out + l];
                float v1 = sx[base_out + l + 1];
                float v2 = sx[base_out + l + 2];
                float v3 = sx[base_out + l + 3];
                sum_x_lo += v0 + v1 + v2 + v3;

                /* Quantize x[4] to INT8 inline (per-group-of-4 scale). */
                float amax_lo = fmaxf(fmaxf(fabsf(v0), fabsf(v1)),
                                      fmaxf(fabsf(v2), fabsf(v3)));
                float qscale_lo = (amax_lo > 0.0f) ? 127.0f / amax_lo : 0.0f;
                int x_lo = pack4_i8(__float2int_rn(v0 * qscale_lo),
                                    __float2int_rn(v1 * qscale_lo),
                                    __float2int_rn(v2 * qscale_lo),
                                    __float2int_rn(v3 * qscale_lo));
                float x_inv_lo = (amax_lo > 0.0f) ? amax_lo / 127.0f : 0.0f;

                int w_lo = pack4_i8(q0 & 0xF, q1 & 0xF, q2 & 0xF, q3 & 0xF);
                int dp_lo = __dp4a(w_lo, x_lo, 0);
                acc += sc0 * x_inv_lo * (float)dp_lo;

                /* Load 4 x values for hi nibble (sub-block 1). */
                float v4 = sx[base_out + 32 + l];
                float v5 = sx[base_out + 32 + l + 1];
                float v6 = sx[base_out + 32 + l + 2];
                float v7 = sx[base_out + 32 + l + 3];
                sum_x_hi += v4 + v5 + v6 + v7;

                float amax_hi = fmaxf(fmaxf(fabsf(v4), fabsf(v5)),
                                      fmaxf(fabsf(v6), fabsf(v7)));
                float qscale_hi = (amax_hi > 0.0f) ? 127.0f / amax_hi : 0.0f;
                int x_hi = pack4_i8(__float2int_rn(v4 * qscale_hi),
                                    __float2int_rn(v5 * qscale_hi),
                                    __float2int_rn(v6 * qscale_hi),
                                    __float2int_rn(v7 * qscale_hi));
                float x_inv_hi = (amax_hi > 0.0f) ? amax_hi / 127.0f : 0.0f;

                int w_hi = pack4_i8(q0 >> 4, q1 >> 4, q2 >> 4, q3 >> 4);
                int dp_hi = __dp4a(w_hi, x_hi, 0);
                acc += sc1 * x_inv_hi * (float)dp_hi;
            }

            acc -= mn0 * sum_x_lo + mn1 * sum_x_hi;
        }
    }

    /* Warp shuffle reduction. */
    #pragma unroll
    for (int offset = Q4K_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

/* ---------- dp4a Dispatcher ---------- */

extern "C" cudaError_t gemv_q4k_dp4a_f32(
    const void* W_q4k, const float* x, float* y,
    int M, int K,
    cudaStream_t stream)
{
    if (K % Q4K_SUPER_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    int threads = Q4K_WARPS_PER_BLOCK * Q4K_WARP_SIZE;  /* 128 */
    int grid = (M + Q4K_WARPS_PER_BLOCK - 1) / Q4K_WARPS_PER_BLOCK;
    int smem = K * sizeof(float);

    gemv_q4k_dp4a_kernel<<<grid, threads, smem, stream>>>(
        (const uint8_t*)W_q4k, x, y, M, K);

    return cudaGetLastError();
}

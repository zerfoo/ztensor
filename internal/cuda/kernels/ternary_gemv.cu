/* Ternary GEMV CUDA kernel.
 *
 * Packed 2-bit ternary weights: 00=-1, 01=0, 10=+1.
 * Each thread processes one output row, decoding packed weights and
 * accumulating +x[j] or -x[j] (no multiply needed).
 * Warp shuffle reduction produces the final dot product.
 *
 * Strategy:
 *   - Load input vector x into shared memory.
 *   - One warp per row. Each lane processes a strided subset of bytes.
 *   - Within each byte, 4 ternary values are decoded via bit ops.
 *   - Warp shuffle reduction across lanes.
 */

#include "ternary_gemv.h"
#include <stdint.h>

#define TERNARY_WARPS_PER_BLOCK 4
#define TERNARY_WARP_SIZE       32

__global__ void ternary_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int M, int K)
{
    extern __shared__ float sx[];

    /* Cooperatively load x into shared memory. */
    int threads_per_block = blockDim.x;
    for (int i = threadIdx.x; i < K; i += threads_per_block) {
        sx[i] = __ldg(&x[i]);
    }
    __syncthreads();

    int warp_id = threadIdx.x / TERNARY_WARP_SIZE;
    int lane_id = threadIdx.x % TERNARY_WARP_SIZE;
    int row = blockIdx.x * TERNARY_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    /* Number of full bytes per row (4 values per byte). */
    int bytes_per_row = (K + 3) / 4;
    int row_byte_start = row * bytes_per_row;

    float acc = 0.0f;

    /* Each lane handles a strided subset of bytes. */
    for (int bi = lane_id; bi < bytes_per_row; bi += TERNARY_WARP_SIZE) {
        uint8_t packed = __ldg(&weights[row_byte_start + bi]);
        int k_base = bi * 4;

        /* Process 4 values from one byte. */
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            int k = k_base + e;
            if (k >= K) break;

            uint8_t bits = (packed >> (e * 2)) & 0x03;
            /* 00 = -1: subtract x[k]
             * 01 =  0: skip
             * 10 = +1: add x[k]
             */
            if (bits == 0) {
                acc -= sx[k];
            } else if (bits == 2) {
                acc += sx[k];
            }
        }
    }

    /* Warp shuffle reduction. */
    #pragma unroll
    for (int offset = TERNARY_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

/* ---------- Dispatcher ---------- */

extern "C" cudaError_t ternary_gemv_f32(
    const void* W_ternary,
    const float* x, float* y,
    int M, int K,
    cudaStream_t stream)
{
    int threads = TERNARY_WARPS_PER_BLOCK * TERNARY_WARP_SIZE;  /* 128 */
    int grid = (M + TERNARY_WARPS_PER_BLOCK - 1) / TERNARY_WARPS_PER_BLOCK;
    int smem = K * sizeof(float);

    ternary_gemv_kernel<<<grid, threads, smem, stream>>>(
        (const uint8_t*)W_ternary, x, y, M, K);

    return cudaGetLastError();
}

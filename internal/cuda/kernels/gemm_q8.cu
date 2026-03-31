/* Q8_0 dequant-GEMM kernel for Zerfoo's Q8Storage format.
 *
 * Each Q8_0 block (36 bytes per 32 values):
 *   bytes[0:4]  = float32 scale (little-endian IEEE 754)
 *   bytes[4:36] = 32 x int8 quantized values
 *   Dequant: val = int8_val * scale
 *
 * Two kernels:
 *   gemv_q8_kernel -- optimized for N=1 (single-token generation).
 *     Uses vectorized int4 loads (16 int8 values at once) and float4
 *     loads from shared memory. Warp-per-row with warp shuffle reduction.
 *   gemm_q8_kernel -- general GEMM fallback for N>1 (prompt processing).
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define Q8_BLOCK_SIZE 32
#define Q8_BLOCK_BYTES 36

/* ---------- Optimized GEMV kernel (N=1) ---------- */
#define Q8_WARPS_PER_BLOCK 8
#define Q8_WARP_SIZE 32

__global__ void gemv_q8_kernel(
    const uint8_t* __restrict__ W_q8,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int M, int K)
{
    extern __shared__ float sx[];

    /* Cooperatively load x[0..K-1] into shared memory.
     * Use per-element loads instead of float4 to avoid misaligned access
     * when the activation pointer x is not 16-byte aligned (common on
     * ARM64/Grace Hopper when x comes from pool allocations with
     * non-aligned offsets). Shared memory loads later in the kernel are
     * always aligned since shared memory base is 16-byte aligned. */
    int threads_per_block = blockDim.x;
    for (int i = threadIdx.x; i < K; i += threads_per_block) {
        sx[i] = __ldg(&x[i]);
    }
    __syncthreads();

    int warp_id = threadIdx.x / Q8_WARP_SIZE;
    int lane_id = threadIdx.x % Q8_WARP_SIZE;
    int row = blockIdx.x * Q8_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    int blocks_per_row = K / Q8_BLOCK_SIZE;
    const uint8_t* row_data = W_q8 + (size_t)row * blocks_per_row * Q8_BLOCK_BYTES;

    float acc = 0.0f;

    /* Each lane handles a strided subset of Q8 blocks. */
    for (int bi = lane_id; bi < blocks_per_row; bi += Q8_WARP_SIZE) {
        const uint8_t* blk = row_data + bi * Q8_BLOCK_BYTES;

        /* Read float32 scale (4 bytes, little-endian) using __ldg. */
        float scale = __ldg((const float*)blk);

        int k_base = bi * Q8_BLOCK_SIZE;
        const int8_t* qvals = (const int8_t*)(blk + 4);

        /* Read 32 int8 quantized values using per-byte loads.
         * Avoid int4 (16-byte) vectorized loads because the Q8 block
         * layout (4-byte scale + 32-byte data = 36 bytes) means qvals
         * is only 4-byte aligned, not 16-byte aligned. On ARM64/Grace
         * Hopper, misaligned int4 loads cause fatal errors. */

        /* Process first 16 values using float4 loads from shared memory
         * (shared memory is always 16-byte aligned). */
        float4 sx0 = ((float4*)&sx[k_base])[0];
        float4 sx1 = ((float4*)&sx[k_base])[1];
        float4 sx2 = ((float4*)&sx[k_base])[2];
        float4 sx3 = ((float4*)&sx[k_base])[3];

        acc += scale * (
            (float)qvals[0]  * sx0.x + (float)qvals[1]  * sx0.y +
            (float)qvals[2]  * sx0.z + (float)qvals[3]  * sx0.w +
            (float)qvals[4]  * sx1.x + (float)qvals[5]  * sx1.y +
            (float)qvals[6]  * sx1.z + (float)qvals[7]  * sx1.w +
            (float)qvals[8]  * sx2.x + (float)qvals[9]  * sx2.y +
            (float)qvals[10] * sx2.z + (float)qvals[11] * sx2.w +
            (float)qvals[12] * sx3.x + (float)qvals[13] * sx3.y +
            (float)qvals[14] * sx3.z + (float)qvals[15] * sx3.w);

        /* Process second 16 values. */
        float4 sx4 = ((float4*)&sx[k_base + 16])[0];
        float4 sx5 = ((float4*)&sx[k_base + 16])[1];
        float4 sx6 = ((float4*)&sx[k_base + 16])[2];
        float4 sx7 = ((float4*)&sx[k_base + 16])[3];

        acc += scale * (
            (float)qvals[16] * sx4.x + (float)qvals[17] * sx4.y +
            (float)qvals[18] * sx4.z + (float)qvals[19] * sx4.w +
            (float)qvals[20] * sx5.x + (float)qvals[21] * sx5.y +
            (float)qvals[22] * sx5.z + (float)qvals[23] * sx5.w +
            (float)qvals[24] * sx6.x + (float)qvals[25] * sx6.y +
            (float)qvals[26] * sx6.z + (float)qvals[27] * sx6.w +
            (float)qvals[28] * sx7.x + (float)qvals[29] * sx7.y +
            (float)qvals[30] * sx7.z + (float)qvals[31] * sx7.w);
    }

    /* Warp shuffle reduction. */
    #pragma unroll
    for (int offset = Q8_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

/* ---------- General GEMM kernel (N>1) ---------- */
#define Q8_TILE_M 16
#define Q8_TILE_N 16

__global__ void gemm_q8_kernel(
    const uint8_t* __restrict__ A_q8,
    const float*   __restrict__ B,
    float*         __restrict__ C,
    int M, int K, int N)
{
    int row = blockIdx.y * Q8_TILE_M + threadIdx.y;
    int col = blockIdx.x * Q8_TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    int blocks_per_row = K / Q8_BLOCK_SIZE;
    float acc = 0.0f;

    const uint8_t* row_blocks = A_q8 + (size_t)row * blocks_per_row * Q8_BLOCK_BYTES;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const uint8_t* blk = row_blocks + bi * Q8_BLOCK_BYTES;

        float scale = __ldg((const float*)blk);

        int k_base = bi * Q8_BLOCK_SIZE;
        const int8_t* qvals = (const int8_t*)(blk + 4);

        #pragma unroll
        for (int j = 0; j < Q8_BLOCK_SIZE; j++) {
            acc += (float)qvals[j] * scale * __ldg(&B[(k_base + j) * N + col]);
        }
    }

    C[row * N + col] = acc;
}

/* ---------- Dispatcher ---------- */

extern "C" cudaError_t gemm_q8_f32(
    const void* A_q8, const float* B, float* C,
    int M, int K, int N,
    cudaStream_t stream)
{
    if (K % Q8_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    if (N == 1) {
        /* GEMV fast path: y = W * x. */
        int threads = Q8_WARPS_PER_BLOCK * Q8_WARP_SIZE;  /* 256 */
        int grid = (M + Q8_WARPS_PER_BLOCK - 1) / Q8_WARPS_PER_BLOCK;
        int smem = K * sizeof(float);

        gemv_q8_kernel<<<grid, threads, smem, stream>>>(
            (const uint8_t*)A_q8, B, C, M, K);
    } else {
        dim3 block(Q8_TILE_N, Q8_TILE_M);
        dim3 grid_dim((N + Q8_TILE_N - 1) / Q8_TILE_N, (M + Q8_TILE_M - 1) / Q8_TILE_M);

        gemm_q8_kernel<<<grid_dim, block, 0, stream>>>(
            (const uint8_t*)A_q8, B, C, M, K, N);
    }

    return cudaGetLastError();
}

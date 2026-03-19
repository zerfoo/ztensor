/* Warp-specialized GEMV kernel for decode-phase LLM inference.
 *
 * Computes y[M] = A[M x N] * x[N] where A is row-major.
 * Supports both FP32 and FP16 data types.
 *
 * Strategy:
 *   - Each warp owns a tile of output rows (warp-specialized).
 *   - Vectorized loads (float4 / half2) for memory bandwidth.
 *   - Warp-shuffle reduction (__shfl_down_sync) — no shared memory for reduction.
 *   - x loaded into shared memory cooperatively (reused across all rows in block).
 *   - Multiple warps per block for higher occupancy on tall-skinny matrices.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* ======================== FP32 Kernel ======================== */

#define BLOCK_SIZE_F32       256
#define WARP_SIZE            32
#define WARPS_PER_BLOCK_F32  (BLOCK_SIZE_F32 / WARP_SIZE)  /* 8 */
#define ROWS_PER_BLOCK_F32   WARPS_PER_BLOCK_F32

__global__ void gemv_warp_f32_kernel(
    float*       __restrict__ y,
    const float* __restrict__ A,
    const float* __restrict__ x,
    int M, int N)
{
    extern __shared__ float sx[];

    /* Cooperatively load x into shared memory. */
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE_F32) {
        sx[i] = x[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * ROWS_PER_BLOCK_F32 + warp_id;

    if (row >= M) return;

    const float* row_ptr = A + (size_t)row * N;
    float acc = 0.0f;

    /* Vectorized float4 path for the bulk of the row. */
    int n4 = N / 4;
    const float4* row4 = (const float4*)row_ptr;
    const float4* sx4  = (const float4*)sx;

    for (int i = lane_id; i < n4; i += WARP_SIZE) {
        float4 a4 = __ldg(&row4[i]);
        float4 x4 = sx4[i];
        acc = __fmaf_rn(a4.x, x4.x, acc);
        acc = __fmaf_rn(a4.y, x4.y, acc);
        acc = __fmaf_rn(a4.z, x4.z, acc);
        acc = __fmaf_rn(a4.w, x4.w, acc);
    }

    /* Handle remainder elements (N not divisible by 4). */
    int rem_start = n4 * 4;
    for (int i = rem_start + lane_id; i < N; i += WARP_SIZE) {
        acc = __fmaf_rn(row_ptr[i], sx[i], acc);
    }

    /* Warp shuffle reduction. */
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

extern "C" cudaError_t launch_gemv_warp_f32(
    float* y, const float* A, const float* x,
    int M, int N, cudaStream_t stream)
{
    int grid = (M + ROWS_PER_BLOCK_F32 - 1) / ROWS_PER_BLOCK_F32;
    int smem = N * sizeof(float);

    gemv_warp_f32_kernel<<<grid, BLOCK_SIZE_F32, smem, stream>>>(y, A, x, M, N);

    return cudaGetLastError();
}

/* ======================== FP16 Kernel ======================== */

#define BLOCK_SIZE_F16       256
#define WARPS_PER_BLOCK_F16  (BLOCK_SIZE_F16 / WARP_SIZE)  /* 8 */
#define ROWS_PER_BLOCK_F16   WARPS_PER_BLOCK_F16

__global__ void gemv_warp_f16_kernel(
    __half*       __restrict__ y,
    const __half* __restrict__ A,
    const __half* __restrict__ x,
    int M, int N)
{
    extern __shared__ __half sx_fp16[];

    /* Cooperatively load x into shared memory. */
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE_F16) {
        sx_fp16[i] = x[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * ROWS_PER_BLOCK_F16 + warp_id;

    if (row >= M) return;

    const __half* row_ptr = A + (size_t)row * N;

    /* Accumulate in FP32 for precision. */
    float acc = 0.0f;

    /* Vectorized half2 path for the bulk of the row. */
    int n2 = N / 2;
    const __half2* row2 = (const __half2*)row_ptr;
    const __half2* sx2  = (const __half2*)sx_fp16;

    for (int i = lane_id; i < n2; i += WARP_SIZE) {
        __half2 a2 = row2[i];
        __half2 x2 = sx2[i];
        /* Accumulate in FP32 using __half2float. */
        acc += __half2float(a2.x) * __half2float(x2.x);
        acc += __half2float(a2.y) * __half2float(x2.y);
    }

    /* Handle remainder (N odd). */
    int rem_start = n2 * 2;
    for (int i = rem_start + lane_id; i < N; i += WARP_SIZE) {
        acc += __half2float(row_ptr[i]) * __half2float(sx_fp16[i]);
    }

    /* Warp shuffle reduction. */
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = __float2half(acc);
    }
}

extern "C" cudaError_t launch_gemv_warp_f16(
    void* y, const void* A, const void* x,
    int M, int N, cudaStream_t stream)
{
    int grid = (M + ROWS_PER_BLOCK_F16 - 1) / ROWS_PER_BLOCK_F16;
    int smem = N * sizeof(__half);

    gemv_warp_f16_kernel<<<grid, BLOCK_SIZE_F16, smem, stream>>>(
        (__half*)y, (const __half*)A, (const __half*)x, M, N);

    return cudaGetLastError();
}

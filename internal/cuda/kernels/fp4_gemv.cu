/* NVFP4 E2M1 fused dequant-GEMV kernel for single-token decode (batch=1).
 *
 * Reads NVFP4 blocks directly, dequantizes in registers (no global
 * memory intermediary), multiplies by the FP16 activation vector, and
 * accumulates in FP32. This halves memory traffic compared to separate
 * dequant + GEMV.
 *
 * NVFP4 block (16 values):
 *   - 8 bytes packed data: 2 x 4-bit E2M1 per byte (low nibble = even idx)
 *   - 1 x float16 scale per block
 *
 * FP4 E2M1 nibble: bit3 = sign, bits 2:0 = magnitude code
 * Magnitude LUT: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
 *
 * Dequantization:
 *   val = (-1)^sign * LUT[code] * scale
 *
 * Requires sm_100+ (Blackwell) for native FP4 tensor core support.
 * Falls back gracefully: the Go layer routes to FP8 GEMV on older GPUs.
 */

#include "fp4_gemv.h"
#include <cuda_fp16.h>
#include <stdint.h>

#define FP4_BLOCK_SIZE     16
#define FP4_BLOCK_BYTES    8    /* 16 values * 4 bits / 8 = 8 bytes packed data */
#define FP4_WARPS_PER_BLOCK 4
#define FP4_WARP_SIZE      32

/* E2M1 magnitude lookup table (device constant memory). */
__device__ __constant__ float fp4_lut[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

/* ---------- Capability check ---------- */

extern "C" int fp4_gemv_check_sm100() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return 0;

    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

    /* sm_100 = Blackwell (B200, GB10/DGX Spark). */
    return (major >= 10) ? 1 : 0;
}

/* ---------- Fused GEMV kernel ----------
 *
 * y[row] = sum_k dequant(W_fp4[row, k]) * x[k]
 *
 * Strategy:
 *   - Load input vector x (FP16) into shared memory, convert to FP32.
 *   - One warp per row for simplicity and good occupancy.
 *   - Each lane processes a strided subset of NVFP4 blocks.
 *   - Within each block, 16 elements are dequantized and dot-producted.
 *   - Warp shuffle reduction produces the final dot product.
 */
__global__ void fp4_gemv_kernel(
    const uint8_t* __restrict__ W_fp4,
    const __half*  __restrict__ scales,
    const __half*  __restrict__ x,
    float*         __restrict__ y,
    int M, int K)
{
    extern __shared__ float sx[];

    /* Cooperatively load x (FP16) into shared memory as FP32. */
    int threads_per_block = blockDim.x;
    for (int i = threadIdx.x; i < K; i += threads_per_block) {
        sx[i] = __half2float(__ldg(&x[i]));
    }
    __syncthreads();

    int warp_id = threadIdx.x / FP4_WARP_SIZE;
    int lane_id = threadIdx.x % FP4_WARP_SIZE;
    int row = blockIdx.x * FP4_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    int blocks_per_row = K / FP4_BLOCK_SIZE;
    const uint8_t* row_data = W_fp4 + (size_t)row * blocks_per_row * FP4_BLOCK_BYTES;
    const __half* row_scales = scales + (size_t)row * blocks_per_row;

    float acc = 0.0f;

    /* Each lane handles a strided subset of blocks. */
    for (int bi = lane_id; bi < blocks_per_row; bi += FP4_WARP_SIZE) {
        const uint8_t* blk = row_data + bi * FP4_BLOCK_BYTES;
        float scale = __half2float(__ldg(&row_scales[bi]));
        int k_base = bi * FP4_BLOCK_SIZE;

        /* Process 16 values from 8 packed bytes. */
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            uint8_t packed = __ldg(&blk[b]);

            /* Low nibble (even index). */
            uint8_t lo = packed & 0x0F;
            float sign_lo = (lo & 0x08) ? -1.0f : 1.0f;
            float mag_lo = fp4_lut[lo & 0x07];
            float dq_lo = sign_lo * mag_lo * scale;

            /* High nibble (odd index). */
            uint8_t hi = packed >> 4;
            float sign_hi = (hi & 0x08) ? -1.0f : 1.0f;
            float mag_hi = fp4_lut[hi & 0x07];
            float dq_hi = sign_hi * mag_hi * scale;

            acc = __fmaf_rn(dq_lo, sx[k_base + b * 2], acc);
            acc = __fmaf_rn(dq_hi, sx[k_base + b * 2 + 1], acc);
        }
    }

    /* Warp shuffle reduction. */
    #pragma unroll
    for (int offset = FP4_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

/* ---------- Dispatcher ---------- */

extern "C" cudaError_t fp4_gemv_f16(
    const void* W_fp4, const void* scales,
    const void* x, float* y,
    int M, int K,
    cudaStream_t stream)
{
    if (K % FP4_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    int threads = FP4_WARPS_PER_BLOCK * FP4_WARP_SIZE;  /* 128 */
    int grid = (M + FP4_WARPS_PER_BLOCK - 1) / FP4_WARPS_PER_BLOCK;
    int smem = K * sizeof(float);

    fp4_gemv_kernel<<<grid, threads, smem, stream>>>(
        (const uint8_t*)W_fp4, (const __half*)scales,
        (const __half*)x, y, M, K);

    return cudaGetLastError();
}

/* Q5_0 dequantization kernel: converts Q5_0 blocks to FP32.
 *
 * Q5_0 block (22 bytes, 32 values):
 *   [0:2]   fp16 d (scale)
 *   [2:6]   uint32 qh (32 high bits, one per element)
 *   [6:22]  16 bytes qs (packed 4-bit low nibbles)
 *
 * Each thread block processes one block (32 threads, one per value).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define Q5_0_BLOCK_SIZE  32
#define Q5_0_BLOCK_BYTES 22

__global__ void dequant_q5_0_kernel(
    const uint8_t* __restrict__ src,
    float*         __restrict__ dst,
    int total_blocks)
{
    int block_idx = blockIdx.x;
    if (block_idx >= total_blocks) return;

    int tid = threadIdx.x; /* 0..31 */
    const uint8_t* blk = src + (size_t)block_idx * Q5_0_BLOCK_BYTES;

    /* Read fp16 d using byte-wise load for ARM64 alignment. */
    uint16_t d_bits = (uint16_t)__ldg(&blk[0]) | ((uint16_t)__ldg(&blk[1]) << 8);
    float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));

    /* Read qh using byte-wise load (blk+2 may not be 4-byte aligned). */
    uint32_t qh = (uint32_t)__ldg(&blk[2]) | ((uint32_t)__ldg(&blk[3]) << 8)
                 | ((uint32_t)__ldg(&blk[4]) << 16) | ((uint32_t)__ldg(&blk[5]) << 24);

    const uint8_t* qs = blk + 6;

    int q;
    if (tid < 16) {
        /* First half: low nibble + high bit at position tid. */
        uint8_t packed = __ldg(&qs[tid]);
        uint8_t low4  = packed & 0x0F;
        uint8_t h     = ((qh >> tid) & 1) << 4;
        q = (int)(low4 | h) - 16;
    } else {
        /* Second half: high nibble + high bit at position tid. */
        uint8_t packed = __ldg(&qs[tid - 16]);
        uint8_t high4  = packed >> 4;
        uint8_t h      = ((qh >> tid) & 1) << 4;
        q = (int)(high4 | h) - 16;
    }

    dst[block_idx * Q5_0_BLOCK_SIZE + tid] = d * (float)q;
}

extern "C" cudaError_t dequant_q5_0_f32(
    const void* src, float* dst,
    int rows, int K,
    cudaStream_t stream)
{
    if (K % Q5_0_BLOCK_SIZE != 0) return cudaErrorInvalidValue;
    int total_blocks = rows * (K / Q5_0_BLOCK_SIZE);
    dequant_q5_0_kernel<<<total_blocks, Q5_0_BLOCK_SIZE, 0, stream>>>(
        (const uint8_t*)src, dst, total_blocks);
    return cudaGetLastError();
}

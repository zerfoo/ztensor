/* Q6_K dequantization kernel: converts Q6_K super-blocks to FP32.
 *
 * Q6_K super-block (210 bytes, 256 values):
 *   [0:128]   ql -- low 4 bits of each 6-bit value
 *   [128:192] qh -- high 2 bits of each 6-bit value
 *   [192:208] sc -- int8 scales for 16 sub-blocks of 16 values
 *   [208:210] d  -- fp16 super-block scale
 *
 * Each thread block processes one super-block (256 threads, one per value).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define Q6K_SUPER_BLOCK_SIZE 256
#define Q6K_BLOCK_BYTES      210

__global__ void dequant_q6k_kernel(
    const uint8_t* __restrict__ src,
    float*         __restrict__ dst,
    int total_blocks)
{
    int block_idx = blockIdx.x;
    if (block_idx >= total_blocks) return;

    int tid = threadIdx.x; /* 0..255 */
    const uint8_t* blk = src + (size_t)block_idx * Q6K_BLOCK_BYTES;

    const uint8_t* ql = blk;        /* 128 bytes */
    const uint8_t* qh = blk + 128;  /* 64 bytes */
    const int8_t*  sc = (const int8_t*)(blk + 192); /* 16 bytes */

    /* Read fp16 d using byte-wise load for ARM64 alignment. */
    uint16_t d_bits = (uint16_t)__ldg(&blk[208]) | ((uint16_t)__ldg(&blk[209]) << 8);
    float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));

    /* Map tid to half (0 or 1), position within half (0..127).
     * Within each half of 128 values, there are 4 groups of 32. */
    int half = tid / 128;
    int pos  = tid % 128;
    int group = pos / 32;   /* 0..3 */
    int l     = pos % 32;   /* 0..31 */

    int qlOff = half * 64;
    int qhOff = half * 32;
    int scOff = half * 8;

    /* sub-block index for the int8 scale */
    int is = l / 16;

    /* Reconstruct 6-bit value from ql (4 low bits) + qh (2 high bits). */
    uint8_t ql_byte, qh_byte;
    int8_t  scale;
    int     q;

    switch (group) {
    case 0:
        ql_byte = __ldg(&ql[qlOff + l]);
        qh_byte = __ldg(&qh[qhOff + l]);
        scale   = (int8_t)__ldg((const uint8_t*)&sc[scOff + is + 0]);
        q = (int)((ql_byte & 0xF) | ((qh_byte & 3) << 4)) - 32;
        break;
    case 1:
        ql_byte = __ldg(&ql[qlOff + 32 + l]);
        qh_byte = __ldg(&qh[qhOff + l]);
        scale   = (int8_t)__ldg((const uint8_t*)&sc[scOff + is + 2]);
        q = (int)((ql_byte & 0xF) | (((qh_byte >> 2) & 3) << 4)) - 32;
        break;
    case 2:
        ql_byte = __ldg(&ql[qlOff + l]);
        qh_byte = __ldg(&qh[qhOff + l]);
        scale   = (int8_t)__ldg((const uint8_t*)&sc[scOff + is + 4]);
        q = (int)((ql_byte >> 4) | (((qh_byte >> 4) & 3) << 4)) - 32;
        break;
    case 3:
        ql_byte = __ldg(&ql[qlOff + 32 + l]);
        qh_byte = __ldg(&qh[qhOff + l]);
        scale   = (int8_t)__ldg((const uint8_t*)&sc[scOff + is + 6]);
        q = (int)((ql_byte >> 4) | (((qh_byte >> 6) & 3) << 4)) - 32;
        break;
    }

    float val = d * (float)scale * (float)q;
    dst[block_idx * Q6K_SUPER_BLOCK_SIZE + tid] = val;
}

extern "C" cudaError_t dequant_q6k_f32(
    const void* src, float* dst,
    int rows, int K,
    cudaStream_t stream)
{
    if (K % Q6K_SUPER_BLOCK_SIZE != 0) return cudaErrorInvalidValue;
    int total_blocks = rows * (K / Q6K_SUPER_BLOCK_SIZE);
    dequant_q6k_kernel<<<total_blocks, Q6K_SUPER_BLOCK_SIZE, 0, stream>>>(
        (const uint8_t*)src, dst, total_blocks);
    return cudaGetLastError();
}

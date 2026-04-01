/* Q5_K dequantization kernel: converts Q5_K super-blocks to FP32.
 *
 * Q5_K super-block (176 bytes, 256 values):
 *   [0:2]    fp16 d (super-block scale)
 *   [2:4]    fp16 dmin (super-block min)
 *   [4:16]   12 bytes packed 6-bit scales/mins (same as Q4_K)
 *   [16:144] 128 bytes ql (low 4 bits per value)
 *   [144:176] 32 bytes qh (high 1 bit per value, 256 bits)
 *
 * Same scale/min encoding as Q4_K, but each value has 5 bits (4 from ql + 1 from qh).
 * Each thread block processes one super-block (256 threads).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define Q5K_SUPER_BLOCK_SIZE 256
#define Q5K_BLOCK_BYTES      176

/* Same decode as Q4_K (shared scale/min format). */
__device__ __forceinline__ void dq5k_decode_scales_mins(
    const uint8_t* sc,
    float d, float dmin,
    float* __restrict__ sub_scales,
    float* __restrict__ sub_mins)
{
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sub_scales[i] = d * (float)(sc[i] & 63);
        sub_mins[i]   = dmin * (float)(sc[4+i] & 63);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sub_scales[4+i] = d * (float)((sc[8+i] & 0xF) | ((sc[i] >> 6) << 4));
        sub_mins[4+i]   = dmin * (float)((sc[8+i] >> 4) | ((sc[4+i] >> 6) << 4));
    }
}

__global__ void dequant_q5k_kernel(
    const uint8_t* __restrict__ src,
    float*         __restrict__ dst,
    int total_blocks)
{
    int block_idx = blockIdx.x;
    if (block_idx >= total_blocks) return;

    int tid = threadIdx.x; /* 0..255 */
    const uint8_t* blk = src + (size_t)block_idx * Q5K_BLOCK_BYTES;

    float d    = __half2float(__ldg((const __half*)(blk)));
    float dmin = __half2float(__ldg((const __half*)(blk + 2)));

    __shared__ float s_scales[8];
    __shared__ float s_mins[8];
    if (tid < 8) {
        float sub_scales[8], sub_mins[8];
        dq5k_decode_scales_mins(blk + 4, d, dmin, sub_scales, sub_mins);
        s_scales[tid] = sub_scales[tid];
        s_mins[tid]   = sub_mins[tid];
    }
    __syncthreads();

    const uint8_t* ql = blk + 16;   /* 128 bytes */
    const uint8_t* qh = blk + 144;  /* 32 bytes */

    int group = tid / 64;
    int pos   = tid % 64;

    int sb, q_val;
    uint8_t q_byte = __ldg(&ql[group * 32 + (pos < 32 ? pos : pos - 32)]);
    uint8_t h_byte = __ldg(&qh[pos < 32 ? pos : pos - 32]);

    /* Compute which bit pair in qh to use.
     * group 0: u1=1,u2=2; group 1: u1=4,u2=8; group 2: u1=16,u2=32; group 3: u1=64,u2=128 */
    uint8_t u1 = 1 << (group * 2);
    uint8_t u2 = 1 << (group * 2 + 1);

    if (pos < 32) {
        sb = group * 2;
        uint8_t h = (h_byte & u1) ? 16 : 0;
        q_val = (int)((q_byte & 0xF) | h) - 16;    /* minus dmin handled below */
    } else {
        sb = group * 2 + 1;
        uint8_t h = (h_byte & u2) ? 16 : 0;
        q_val = (int)((q_byte >> 4) | h) - 16;
    }

    /* Q5_K dequant: scale * nibble_with_highbit - min */
    float val = s_scales[sb] * (float)q_val - s_mins[sb];
    dst[block_idx * Q5K_SUPER_BLOCK_SIZE + tid] = val;
}

extern "C" cudaError_t dequant_q5k_f32(
    const void* src, float* dst,
    int rows, int K,
    cudaStream_t stream)
{
    if (K % Q5K_SUPER_BLOCK_SIZE != 0) return cudaErrorInvalidValue;
    int total_blocks = rows * (K / Q5K_SUPER_BLOCK_SIZE);
    dequant_q5k_kernel<<<total_blocks, Q5K_SUPER_BLOCK_SIZE, 0, stream>>>(
        (const uint8_t*)src, dst, total_blocks);
    return cudaGetLastError();
}

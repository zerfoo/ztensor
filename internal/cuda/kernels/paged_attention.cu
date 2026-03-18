/* Paged attention forward kernel (float32).
 *
 * Computes scaled dot-product attention where K/V are stored in
 * non-contiguous blocks accessed via a block table. This is the core
 * kernel for paged KV cache inference (vLLM-style).
 *
 * Each thread block handles one (batch, query_head) pair. It iterates
 * over logical blocks, loads K/V via block pointer indirection, and
 * accumulates the softmax numerator and denominator online (log-sum-exp
 * trick) so no O(seq_len) buffer is needed.
 *
 * Block layout: each K/V block holds [block_size, num_kv_heads, head_dim]
 * floats. The block_indices array maps logical block index to physical
 * block index in the block_ptrs arrays.
 */

#include "paged_attention.h"
#include <float.h>
#include <math.h>

#ifndef PAGED_BLOCK_THREADS
#define PAGED_BLOCK_THREADS 32
#endif

/* Maximum head dimension (shared-memory based, cooperative reduction). */
#define MAX_HEAD_DIM_PAGED 256

/* Kernel: one CUDA thread block per (batch, query_head).
 *
 * Each thread handles a stripe of the head dimension for dot products
 * and accumulator updates. Thread 0 does the online softmax bookkeeping
 * and broadcasts rescaling factors via shared memory.
 *
 * Shared memory layout:
 *   sQ[head_dim]                  -- scaled query
 *   sAcc[head_dim]                -- softmax-weighted V accumulator
 *   sPartial[PAGED_BLOCK_THREADS] -- dot-product partial sums + broadcast
 */
__global__ void paged_attention_kernel(
    const float* __restrict__ Q,
    float* __restrict__ O,
    const float** __restrict__ block_ptrs_k,
    const float** __restrict__ block_ptrs_v,
    const int* __restrict__ block_indices,
    int seq_len, int block_size, int head_dim,
    int num_q_heads, int num_kv_heads)
{
    int bh = blockIdx.x;           /* (batch, query_head) index */
    int tid = threadIdx.x;         /* thread within block [0, PAGED_BLOCK_THREADS) */

    /* GQA: map query head to KV head. */
    int head_ratio = num_q_heads / num_kv_heads;
    int batch_idx = bh / num_q_heads;
    int q_head = bh % num_q_heads;
    int kv_head = q_head / head_ratio;

    /* Q/O base: single row at [bh, :]. */
    const float* Q_bh = Q + bh * head_dim;
    float* O_bh = O + bh * head_dim;

    float scale = rsqrtf((float)head_dim);

    /* Shared memory. */
    extern __shared__ float smem[];
    float* sQ = smem;
    float* sAcc = smem + head_dim;
    float* sPartial = smem + 2 * head_dim;

    /* Load and scale Q into shared memory cooperatively. */
    for (int d = tid; d < head_dim; d += PAGED_BLOCK_THREADS) {
        sQ[d] = Q_bh[d] * scale;
    }
    /* Zero the accumulator. */
    for (int d = tid; d < head_dim; d += PAGED_BLOCK_THREADS) {
        sAcc[d] = 0.0f;
    }
    __syncthreads();

    /* Thread 0 tracks the online softmax state. */
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    /* Number of logical blocks covering seq_len tokens. */
    int num_logical_blocks = (seq_len + block_size - 1) / block_size;

    /* KV head stride within a block: blocks are [block_size, num_kv_heads, head_dim]. */
    int kv_stride = num_kv_heads * head_dim;

    /* Iterate over logical blocks. */
    for (int lb = 0; lb < num_logical_blocks; lb++) {
        int phys_block = block_indices[batch_idx * num_logical_blocks + lb];
        const float* K_block = block_ptrs_k[phys_block];
        const float* V_block = block_ptrs_v[phys_block];

        /* Number of valid positions in this block. */
        int pos_start = lb * block_size;
        int pos_count = block_size;
        if (pos_start + pos_count > seq_len) {
            pos_count = seq_len - pos_start;
        }

        /* Iterate over positions within the block. */
        for (int p = 0; p < pos_count; p++) {
            /* K position offset: [p, kv_head, :] = p * kv_stride + kv_head * head_dim. */
            const float* K_pos = K_block + p * kv_stride + kv_head * head_dim;
            const float* V_pos = V_block + p * kv_stride + kv_head * head_dim;

            /* Step 1: parallel dot product Q * K[pos]. */
            float partial = 0.0f;
            for (int d = tid; d < head_dim; d += PAGED_BLOCK_THREADS) {
                partial += sQ[d] * K_pos[d];
            }
            sPartial[tid] = partial;
            __syncthreads();

            /* Tree reduction to get the full dot product in sPartial[0]. */
            for (int stride = PAGED_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    sPartial[tid] += sPartial[tid + stride];
                }
                __syncthreads();
            }

            /* Step 2: thread 0 does online softmax update. */
            if (tid == 0) {
                float s = sPartial[0];
                float prev_max = row_max;
                if (s > row_max) row_max = s;

                float exp_diff = expf(prev_max - row_max);
                float exp_s = expf(s - row_max);
                row_sum = row_sum * exp_diff + exp_s;

                /* Broadcast rescaling factors. */
                sPartial[0] = exp_diff;
                sPartial[1] = exp_s;
            }
            __syncthreads();

            /* Step 3: all threads update accumulator cooperatively. */
            {
                float exp_diff = sPartial[0];
                float exp_s = sPartial[1];
                for (int d = tid; d < head_dim; d += PAGED_BLOCK_THREADS) {
                    sAcc[d] = sAcc[d] * exp_diff + exp_s * V_pos[d];
                }
            }
            __syncthreads();
        }
    }

    /* Write output: O = sAcc / row_sum. */
    float inv_sum;
    if (tid == 0) {
        inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        sPartial[0] = inv_sum;
    }
    __syncthreads();
    inv_sum = sPartial[0];
    for (int d = tid; d < head_dim; d += PAGED_BLOCK_THREADS) {
        O_bh[d] = sAcc[d] * inv_sum;
    }
}

extern "C" cudaError_t paged_attention_forward_f32(
    const float* Q, float* O,
    const float** block_ptrs_k,
    const float** block_ptrs_v,
    const int* block_indices,
    int seq_len, int block_size, int head_dim,
    int num_q_heads, int num_kv_heads,
    int batch,
    cudaStream_t stream)
{
    if (head_dim > MAX_HEAD_DIM_PAGED) {
        return cudaErrorInvalidValue;
    }

    int num_bh = batch * num_q_heads;

    dim3 grid(num_bh);
    dim3 block(PAGED_BLOCK_THREADS);

    /* Shared memory: sQ[head_dim] + sAcc[head_dim] + sPartial[PAGED_BLOCK_THREADS]. */
    size_t smem = (2 * head_dim + PAGED_BLOCK_THREADS) * sizeof(float);

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(paged_attention_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
    }

    paged_attention_kernel<<<grid, block, smem, stream>>>(
        Q, O, block_ptrs_k, block_ptrs_v, block_indices,
        seq_len, block_size, head_dim,
        num_q_heads, num_kv_heads);

    return cudaGetLastError();
}

/* Ragged batching attention forward kernel (float32).
 *
 * Computes scaled dot-product attention for variable-length sequences packed
 * into a single batch. Each sequence only attends to positions within its own
 * slice (block-diagonal mask), preventing cross-sequence contamination.
 *
 * Grid: one thread block per (sequence, query_head) pair.
 * Each thread block handles a stripe of the head dimension for dot products
 * and accumulator updates, using the online softmax (log-sum-exp) trick.
 *
 * Layout: Q/K/V are packed along the token axis.
 *   Q: [total_tokens, num_q_heads, head_dim]
 *   K: [total_tokens, num_kv_heads, head_dim]
 *   V: [total_tokens, num_kv_heads, head_dim]
 *   O: [total_tokens, num_q_heads, head_dim]
 *
 * seq_lens[i] gives the number of tokens for sequence i.
 * cum_seq_lens[i] gives the starting token offset for sequence i.
 */

#include "ragged_attention.h"
#include <float.h>
#include <math.h>

#ifndef RAGGED_BLOCK_THREADS
#define RAGGED_BLOCK_THREADS 32
#endif

#define MAX_HEAD_DIM_RAGGED 256

/* Kernel: one CUDA thread block per (seq_idx, q_head_within_seq, token_within_seq).
 *
 * Actually we process one query token at a time per block. The grid is:
 *   blockIdx.x = linear index over (seq_idx, token_within_seq, q_head).
 * We flatten as: for each sequence i, for each token t in [0, seq_lens[i]),
 * for each q_head h in [0, num_q_heads), => one block.
 *
 * But since sequences have variable lengths, we use a simpler scheme:
 *   Grid = total_tokens * num_q_heads
 *   blockIdx.x = token_global * num_q_heads + q_head
 * And we use cum_seq_lens to find which sequence a token belongs to.
 */
__global__ void ragged_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int* __restrict__ seq_lens,
    const int* __restrict__ cum_seq_lens,
    int batch, int num_q_heads, int num_kv_heads, int head_dim)
{
    int idx = blockIdx.x;
    int q_head = idx % num_q_heads;
    int token_global = idx / num_q_heads;
    int tid = threadIdx.x;

    /* GQA: map query head to KV head. */
    int head_ratio = num_q_heads / num_kv_heads;
    int kv_head = q_head / head_ratio;

    /* Find which sequence this token belongs to using cum_seq_lens.
     * Linear scan is fine since batch is typically small (2-32). */
    int seq_idx = 0;
    for (int i = 0; i < batch; i++) {
        if (token_global >= cum_seq_lens[i]) {
            seq_idx = i;
        }
    }

    int seq_start = cum_seq_lens[seq_idx];
    int seq_length = seq_lens[seq_idx];

    float scale = rsqrtf((float)head_dim);

    /* Shared memory: sQ[head_dim] + sAcc[head_dim] + sPartial[RAGGED_BLOCK_THREADS]. */
    extern __shared__ float smem[];
    float* sQ = smem;
    float* sAcc = smem + head_dim;
    float* sPartial = smem + 2 * head_dim;

    /* Q layout: [total_tokens, num_q_heads, head_dim].
     * Q for this block's token+head: Q[(token_global * num_q_heads + q_head) * head_dim + d]. */
    int q_offset = (token_global * num_q_heads + q_head) * head_dim;

    /* Load and scale Q into shared memory. */
    for (int d = tid; d < head_dim; d += RAGGED_BLOCK_THREADS) {
        sQ[d] = Q[q_offset + d] * scale;
    }
    /* Zero accumulator. */
    for (int d = tid; d < head_dim; d += RAGGED_BLOCK_THREADS) {
        sAcc[d] = 0.0f;
    }
    __syncthreads();

    /* Online softmax state (thread 0). */
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    /* Iterate over all KV positions in this sequence only (block-diagonal mask). */
    for (int j = 0; j < seq_length; j++) {
        int kv_token = seq_start + j;

        /* K layout: [total_tokens, num_kv_heads, head_dim].
         * K for token kv_token, head kv_head: K[(kv_token * num_kv_heads + kv_head) * head_dim + d]. */
        const float* K_pos = K + (kv_token * num_kv_heads + kv_head) * head_dim;
        const float* V_pos = V + (kv_token * num_kv_heads + kv_head) * head_dim;

        /* Parallel dot product Q * K[j]. */
        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += RAGGED_BLOCK_THREADS) {
            partial += sQ[d] * K_pos[d];
        }
        sPartial[tid] = partial;
        __syncthreads();

        /* Tree reduction. */
        for (int stride = RAGGED_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sPartial[tid] += sPartial[tid + stride];
            }
            __syncthreads();
        }

        /* Thread 0: online softmax update. */
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

        /* All threads update accumulator. */
        {
            float exp_diff = sPartial[0];
            float exp_s = sPartial[1];
            for (int d = tid; d < head_dim; d += RAGGED_BLOCK_THREADS) {
                sAcc[d] = sAcc[d] * exp_diff + exp_s * V_pos[d];
            }
        }
        __syncthreads();
    }

    /* Write output: O = sAcc / row_sum. */
    float inv_sum;
    if (tid == 0) {
        inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        sPartial[0] = inv_sum;
    }
    __syncthreads();
    inv_sum = sPartial[0];

    /* O layout: same as Q: [total_tokens, num_q_heads, head_dim]. */
    int o_offset = (token_global * num_q_heads + q_head) * head_dim;
    for (int d = tid; d < head_dim; d += RAGGED_BLOCK_THREADS) {
        O[o_offset + d] = sAcc[d] * inv_sum;
    }
}

extern "C" cudaError_t ragged_attention_forward_f32(
    const float* Q, const float* K, const float* V, float* O,
    const int* seq_lens, const int* cum_seq_lens,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream)
{
    if (head_dim > MAX_HEAD_DIM_RAGGED) {
        return cudaErrorInvalidValue;
    }

    /* Compute total tokens from cum_seq_lens and seq_lens.
     * We need to launch from host, so we read cum_seq_lens + seq_lens on host.
     * The caller must provide total_tokens = cum_seq_lens[batch-1] + seq_lens[batch-1].
     * But we can't read device memory here, so we accept total_tokens as a derived param.
     *
     * Actually, we compute total_tokens by summing seq_lens on host. But seq_lens
     * is a device pointer. We need to copy it.
     *
     * Simpler approach: we pass total_tokens derived from the last cum_seq_lens + seq_lens.
     * Since both are device pointers, we copy the needed values.
     */
    int last_cum = 0, last_len = 0;
    cudaError_t err;

    err = cudaMemcpyAsync(&last_cum, cum_seq_lens + (batch - 1), sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(&last_len, seq_lens + (batch - 1), sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    int total_tokens = last_cum + last_len;
    int num_blocks = total_tokens * num_q_heads;

    dim3 grid(num_blocks);
    dim3 block(RAGGED_BLOCK_THREADS);

    /* Shared memory: sQ[head_dim] + sAcc[head_dim] + sPartial[RAGGED_BLOCK_THREADS]. */
    size_t smem = (2 * head_dim + RAGGED_BLOCK_THREADS) * sizeof(float);

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(ragged_attention_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
    }

    ragged_attention_kernel<<<grid, block, smem, stream>>>(
        Q, K, V, O,
        seq_lens, cum_seq_lens,
        batch, num_q_heads, num_kv_heads, head_dim);

    return cudaGetLastError();
}

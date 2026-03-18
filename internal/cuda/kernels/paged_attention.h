/* Paged attention forward kernel interface (float32).
 * Computes: O = softmax(Q * K^T / sqrt(head_dim)) * V
 * where K/V are stored in non-contiguous blocks accessed via a block table.
 *
 * This kernel is designed for autoregressive decode: Q has a single row
 * per (batch, head), and K/V are fetched from paged blocks.
 */
#ifndef PAGED_ATTENTION_H
#define PAGED_ATTENTION_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* paged_attention_forward_f32 computes scaled dot-product attention with
 * block-table indirection for paged KV caches.
 *
 * Q:             [batch * num_q_heads, head_dim] -- single query per head.
 * O:             [batch * num_q_heads, head_dim] -- output, same shape as Q.
 * block_ptrs_k:  [num_blocks_in_table] -- device pointers to K block data.
 *                Each block holds [block_size, num_kv_heads, head_dim] floats.
 * block_ptrs_v:  [num_blocks_in_table] -- device pointers to V block data.
 *                Same layout as K blocks.
 * block_indices: [max_num_blocks] -- maps logical block index to physical
 *                block index in block_ptrs_k/v arrays.
 * seq_len:       actual number of valid K/V token positions.
 * block_size:    number of token positions per block.
 * head_dim:      dimension per head.
 * num_q_heads:   number of query heads per batch element.
 * num_kv_heads:  number of KV heads per batch element.
 * batch:         number of sequences in the batch.
 * stream:        CUDA stream for async execution.
 */
cudaError_t paged_attention_forward_f32(
    const float* Q, float* O,
    const float** block_ptrs_k,
    const float** block_ptrs_v,
    const int* block_indices,
    int seq_len, int block_size, int head_dim,
    int num_q_heads, int num_kv_heads,
    int batch,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PAGED_ATTENTION_H */

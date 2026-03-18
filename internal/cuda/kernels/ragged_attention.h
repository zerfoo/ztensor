/* Ragged batching attention forward kernel interface (float32).
 *
 * Computes scaled dot-product attention for variable-length sequences packed
 * into a single batch (ragged batching). A block-diagonal attention mask
 * prevents cross-sequence attention contamination.
 *
 * Q, K, V are packed contiguously: sequence i occupies positions
 * [cum_seq_lens[i], cum_seq_lens[i] + seq_lens[i]) along the sequence axis.
 * Total tokens = sum(seq_lens).
 */
#ifndef RAGGED_ATTENTION_H
#define RAGGED_ATTENTION_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ragged_attention_forward_f32 computes scaled dot-product attention with
 * block-diagonal masking for ragged (variable-length) batches.
 *
 * Q:           [total_tokens * num_q_heads, head_dim] -- packed queries.
 * K:           [total_tokens * num_kv_heads, head_dim] -- packed keys.
 * V:           [total_tokens * num_kv_heads, head_dim] -- packed values.
 * O:           [total_tokens * num_q_heads, head_dim] -- output.
 * seq_lens:    [batch] -- actual sequence length for each sequence.
 * cum_seq_lens:[batch] -- cumulative sequence lengths (prefix sums).
 *              cum_seq_lens[i] = sum(seq_lens[0..i-1]), cum_seq_lens[0] = 0.
 * batch:       number of sequences.
 * num_q_heads: number of query heads per sequence.
 * num_kv_heads:number of KV heads per sequence.
 * head_dim:    dimension per head.
 * stream:      CUDA stream for async execution.
 */
cudaError_t ragged_attention_forward_f32(
    const float* Q, const float* K, const float* V, float* O,
    const int* seq_lens, const int* cum_seq_lens,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* RAGGED_ATTENTION_H */

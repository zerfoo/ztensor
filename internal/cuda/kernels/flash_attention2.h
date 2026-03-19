/* FlashAttention-2 fused kernel interface.
 *
 * Implements the FlashAttention-2 algorithm with:
 * - Tiled online softmax (O(N) memory, no materialized attention matrix)
 * - GQA support (num_q_heads may differ from num_kv_heads)
 * - Multi-warp parallel KV tile processing for decode
 * - Warp-level reduction for efficient dot products
 *
 * Forward: Q, K, V all have seq_len rows. Output O = softmax(QK^T/sqrt(d)) * V.
 * Decode:  Q has 1 row per query head. K/V have kv_len rows (KV cache).
 */
#ifndef FLASH_ATTENTION2_H
#define FLASH_ATTENTION2_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* flash_attention2_forward_f32 computes full-sequence scaled dot-product
 * attention using the FlashAttention-2 tiled algorithm.
 *
 * Q, K, V: [batch * heads * seq_len * head_dim] float32 (row-major).
 * O:       [batch * heads * seq_len * head_dim] output.
 * causal:  if nonzero, apply causal (upper-triangular) mask.
 */
cudaError_t flash_attention2_forward_f32(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int heads, int seq_len, int head_dim,
    int causal, cudaStream_t stream);

/* flash_attention2_decode_f32 computes single-query attention for
 * autoregressive decode with GQA support and multi-warp KV parallelism.
 *
 * Q:  [batch * num_q_heads, 1, head_dim]
 * K:  [batch, max_kv_len, num_kv_heads * head_dim]  (heads packed in dim)
 * V:  [batch, max_kv_len, num_kv_heads * head_dim]
 * O:  [batch * num_q_heads, 1, head_dim]
 *
 * kv_len:      actual KV sequence length (used if kv_len_ptr is null).
 * kv_len_ptr:  GPU-resident int32; if non-null, read at runtime for CUDA
 *              graph compatibility.
 * num_q_heads: query heads per batch element.
 * num_kv_heads: KV heads per batch element.
 */
cudaError_t flash_attention2_decode_f32(
    const float* Q, const float* K, const float* V, float* O,
    int num_bh, int max_kv_len, int head_dim,
    int kv_len, const int* kv_len_ptr,
    int num_q_heads, int num_kv_heads,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FLASH_ATTENTION2_H */

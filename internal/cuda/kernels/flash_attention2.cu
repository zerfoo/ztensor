/* FlashAttention-2 fused CUDA kernels (float32).
 *
 * Two kernels:
 * 1. flash_attention2_fwd_kernel — prefill: outer loop over Q tiles, inner
 *    loop over KV tiles, online softmax with logsumexp rescaling. Each thread
 *    block handles one (batch, head, q_tile). O(N) memory.
 *
 * 2. flash_attention2_decode_kernel — decode: one query row per (batch, head).
 *    Multiple warps process KV tiles in parallel (split-K). Each warp
 *    maintains its own partial softmax accumulator. A final cross-warp
 *    reduction merges the partial results using the logsumexp correction.
 *    GQA: num_q_heads may differ from num_kv_heads.
 *
 * Key FlashAttention-2 features:
 * - Reversed loop order (outer Q, inner KV) for better write locality
 * - Online softmax rescaling (no materialized attention matrix)
 * - O(N) memory — only tile-sized shared memory, no O(N^2) buffers
 * - Multi-warp parallel KV processing for decode path
 * - Warp shuffle reduction for dot products (no shared memory bottleneck)
 */

#include "flash_attention2.h"
#include <float.h>
#include <math.h>

/* ---- Compile-time constants ---- */

/* Tile size for the forward (prefill) kernel. */
#ifndef FA2_FWD_TILE
#define FA2_FWD_TILE 32
#endif

/* Maximum head dimension for forward kernel (register-limited). */
#define FA2_MAX_HEAD_DIM 128

/* Maximum head dimension for decode kernel (shared-memory based). */
#define FA2_MAX_HEAD_DIM_DECODE 256

/* Number of warps for the decode kernel. More warps = more KV parallelism. */
#ifndef FA2_DECODE_WARPS
#define FA2_DECODE_WARPS 4
#endif

#define FA2_WARP_SIZE 32
#define FA2_DECODE_THREADS (FA2_DECODE_WARPS * FA2_WARP_SIZE)

/* ================================================================
 * Forward (prefill) kernel — FlashAttention-2 tiled algorithm.
 * One block per (batch, head, q_tile). Inner loop over KV tiles.
 * ================================================================ */
__global__ void flash_attention2_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len, int head_dim, int causal)
{
    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid = threadIdx.x;

    int q_start = q_tile * FA2_FWD_TILE;
    if (q_start >= seq_len) return;

    int q_end = min(q_start + FA2_FWD_TILE, seq_len);
    int q_count = q_end - q_start;

    int bh_offset = bh * seq_len * head_dim;
    const float* Q_bh = Q + bh_offset;
    const float* K_bh = K + bh_offset;
    const float* V_bh = V + bh_offset;
    float* O_bh = O + bh_offset;

    float scale = rsqrtf((float)head_dim);

    /* Dynamic shared memory: sK[tile * head_dim] + sV[tile * head_dim]. */
    extern __shared__ float smem[];
    float* sK = smem;
    float* sV = smem + FA2_FWD_TILE * head_dim;

    /* Per-thread accumulators (one query row per thread). */
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[FA2_MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    /* Load query row into registers (pre-scaled). */
    float q_row[FA2_MAX_HEAD_DIM];
    if (tid < q_count) {
        int q_idx = q_start + tid;
        for (int d = 0; d < head_dim; d++) {
            q_row[d] = Q_bh[q_idx * head_dim + d] * scale;
        }
    }

    int num_kv_tiles = (seq_len + FA2_FWD_TILE - 1) / FA2_FWD_TILE;
    int max_kv_tile = causal
        ? min(num_kv_tiles, (q_start + FA2_FWD_TILE - 1) / FA2_FWD_TILE + 1)
        : num_kv_tiles;

    for (int kv_tile = 0; kv_tile < max_kv_tile; kv_tile++) {
        int k_start = kv_tile * FA2_FWD_TILE;
        int k_end = min(k_start + FA2_FWD_TILE, seq_len);
        int k_count = k_end - k_start;

        /* Cooperative load of K and V tile into shared memory. */
        for (int row = tid; row < k_count; row += FA2_FWD_TILE) {
            int k_idx = k_start + row;
            for (int d = 0; d < head_dim; d++) {
                sK[row * head_dim + d] = K_bh[k_idx * head_dim + d];
                sV[row * head_dim + d] = V_bh[k_idx * head_dim + d];
            }
        }
        __syncthreads();

        if (tid < q_count) {
            int q_idx = q_start + tid;

            for (int j = 0; j < k_count; j++) {
                int k_idx = k_start + j;
                if (causal && k_idx > q_idx) continue;

                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    s += q_row[d] * sK[j * head_dim + d];
                }

                /* Online softmax update. */
                float prev_max = row_max;
                if (s > row_max) row_max = s;

                float exp_diff = expf(prev_max - row_max);
                float exp_s = expf(s - row_max);
                row_sum = row_sum * exp_diff + exp_s;

                for (int d = 0; d < head_dim; d++) {
                    acc[d] = acc[d] * exp_diff + exp_s * sV[j * head_dim + d];
                }
            }
        }

        __syncthreads();
    }

    /* Write normalized output. */
    if (tid < q_count) {
        int q_idx = q_start + tid;
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int d = 0; d < head_dim; d++) {
            O_bh[q_idx * head_dim + d] = acc[d] * inv_sum;
        }
    }
}

/* ================================================================
 * Decode kernel — FlashAttention-2 with multi-warp KV parallelism.
 *
 * Each thread block handles one (batch, query_head). Multiple warps
 * tile the KV dimension: warp w processes KV positions with stride
 * FA2_DECODE_WARPS. Each warp maintains its own partial softmax state
 * (max, sum, acc[d]) in shared memory. After all KV positions are
 * processed, a cross-warp reduction merges partial results.
 *
 * Warp-level shuffle handles dot-product reduction (no smem needed
 * for the inner loop hot path).
 *
 * Shared memory layout:
 *   sQ[head_dim]                              — scaled query (read-only)
 *   warp_max[FA2_DECODE_WARPS]                — per-warp softmax max
 *   warp_sum[FA2_DECODE_WARPS]                — per-warp softmax sum
 *   warp_acc[FA2_DECODE_WARPS * head_dim]     — per-warp V accumulator
 * ================================================================ */
__global__ void flash_attention2_decode_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int max_kv_len, int head_dim, int kv_dim,
    int kv_len_param,
    const int* __restrict__ kv_len_ptr,
    int num_q_heads, int num_kv_heads)
{
    int bh = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / FA2_WARP_SIZE;
    int lane_id = tid % FA2_WARP_SIZE;

    int kv_len = kv_len_ptr ? *kv_len_ptr : kv_len_param;

    /* GQA head mapping. */
    int head_ratio = num_q_heads / num_kv_heads;
    int batch_idx = bh / num_q_heads;
    int q_head = bh % num_q_heads;
    int kv_head = q_head / head_ratio;

    const float* Q_bh = Q + bh * head_dim;
    float* O_bh = O + bh * head_dim;
    const float* K_base = K + batch_idx * max_kv_len * kv_dim + kv_head * head_dim;
    const float* V_base = V + batch_idx * max_kv_len * kv_dim + kv_head * head_dim;

    float scale = rsqrtf((float)head_dim);

    extern __shared__ float smem[];
    float* sQ = smem;
    float* warp_max_arr = smem + head_dim;
    float* warp_sum_arr = warp_max_arr + FA2_DECODE_WARPS;
    float* warp_acc = warp_sum_arr + FA2_DECODE_WARPS;

    /* Load and scale Q cooperatively. */
    for (int d = tid; d < head_dim; d += FA2_DECODE_THREADS) {
        sQ[d] = Q_bh[d] * scale;
    }

    /* Zero per-warp accumulators. */
    for (int d = tid; d < FA2_DECODE_WARPS * head_dim; d += FA2_DECODE_THREADS) {
        warp_acc[d] = 0.0f;
    }
    __syncthreads();

    /* Per-warp softmax state (in registers, lane 0 is authoritative). */
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    /* My warp's accumulator base pointer. */
    float* my_acc = warp_acc + warp_id * head_dim;

    /* Each warp iterates over KV positions with interleaved assignment.
     * Warp w handles positions w, w + num_warps, w + 2*num_warps, ... */
    for (int j = warp_id; j < kv_len; j += FA2_DECODE_WARPS) {
        /* Dot product Q * K[j] with warp shuffle reduction. */
        float partial = 0.0f;
        for (int d = lane_id; d < head_dim; d += FA2_WARP_SIZE) {
            partial += sQ[d] * K_base[j * kv_dim + d];
        }
        /* Warp-level tree reduction. */
        for (int offset = FA2_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        }
        /* Broadcast the score from lane 0 to all lanes. */
        float s = __shfl_sync(0xFFFFFFFF, partial, 0);

        /* Online softmax update (all lanes compute the same values). */
        float prev_max = local_max;
        if (s > local_max) local_max = s;

        float exp_diff = expf(prev_max - local_max);
        float exp_s = expf(s - local_max);
        local_sum = local_sum * exp_diff + exp_s;

        /* Rescale accumulator and add weighted V[j]. */
        for (int d = lane_id; d < head_dim; d += FA2_WARP_SIZE) {
            my_acc[d] = my_acc[d] * exp_diff + exp_s * V_base[j * kv_dim + d];
        }
    }

    /* Write per-warp softmax state to shared memory for cross-warp merge. */
    if (lane_id == 0) {
        warp_max_arr[warp_id] = local_max;
        warp_sum_arr[warp_id] = local_sum;
    }
    __syncthreads();

    /* ---- Cross-warp reduction ----
     * Find global max, then rescale each warp's accumulator and sum. */

    /* Step 1: compute global max (all threads read the small array). */
    float global_max = warp_max_arr[0];
    for (int w = 1; w < FA2_DECODE_WARPS; w++) {
        if (warp_max_arr[w] > global_max) global_max = warp_max_arr[w];
    }

    /* Step 2: each warp rescales its own accumulator by
     * exp(warp_max - global_max). */
    float my_warp_max = warp_max_arr[warp_id];
    float my_warp_sum = warp_sum_arr[warp_id];
    float rescale = expf(my_warp_max - global_max);
    float scaled_sum = my_warp_sum * rescale;

    for (int d = lane_id; d < head_dim; d += FA2_WARP_SIZE) {
        my_acc[d] *= rescale;
    }

    /* Write rescaled sum back. */
    if (lane_id == 0) {
        warp_sum_arr[warp_id] = scaled_sum;
    }
    __syncthreads();

    /* Step 3: compute global sum and write output. All threads participate
     * in the final accumulator reduction over warps. */
    float global_sum = 0.0f;
    for (int w = 0; w < FA2_DECODE_WARPS; w++) {
        global_sum += warp_sum_arr[w];
    }
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    /* Sum accumulators across all warps and write normalized output. */
    for (int d = tid; d < head_dim; d += FA2_DECODE_THREADS) {
        float total = 0.0f;
        for (int w = 0; w < FA2_DECODE_WARPS; w++) {
            total += warp_acc[w * head_dim + d];
        }
        O_bh[d] = total * inv_sum;
    }
}

/* ---- C entry points ---- */

extern "C" cudaError_t flash_attention2_forward_f32(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int heads, int seq_len, int head_dim,
    int causal, cudaStream_t stream)
{
    if (head_dim > FA2_MAX_HEAD_DIM) {
        return cudaErrorInvalidValue;
    }

    int num_bh = batch * heads;
    int num_q_tiles = (seq_len + FA2_FWD_TILE - 1) / FA2_FWD_TILE;

    dim3 grid(num_bh, num_q_tiles);
    dim3 block(FA2_FWD_TILE);

    size_t smem = 2 * FA2_FWD_TILE * head_dim * sizeof(float);

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(flash_attention2_fwd_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
    }

    flash_attention2_fwd_kernel<<<grid, block, smem, stream>>>(
        Q, K, V, O, seq_len, head_dim, causal);

    return cudaGetLastError();
}

extern "C" cudaError_t flash_attention2_decode_f32(
    const float* Q, const float* K, const float* V, float* O,
    int num_bh, int max_kv_len, int head_dim,
    int kv_len, const int* kv_len_ptr,
    int num_q_heads, int num_kv_heads,
    cudaStream_t stream)
{
    if (head_dim > FA2_MAX_HEAD_DIM_DECODE) {
        return cudaErrorInvalidValue;
    }

    int kv_dim = num_kv_heads * head_dim;

    dim3 grid(num_bh);
    dim3 block(FA2_DECODE_THREADS);

    /* sQ[head_dim] + warp_max[warps] + warp_sum[warps] + warp_acc[warps * head_dim] */
    size_t smem = (head_dim + 2 * FA2_DECODE_WARPS + FA2_DECODE_WARPS * head_dim) * sizeof(float);

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(flash_attention2_decode_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
    }

    flash_attention2_decode_kernel<<<grid, block, smem, stream>>>(
        Q, K, V, O, max_kv_len, head_dim, kv_dim,
        kv_len, kv_len_ptr,
        num_q_heads, num_kv_heads);

    return cudaGetLastError();
}

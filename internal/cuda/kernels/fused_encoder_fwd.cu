/* fused_encoder_fwd.cu -- Fused PatchTST encoder layer forward pass.
 *
 * Host-side orchestrator that replaces ~78 discrete Engine[T] operations
 * per encoder layer with a single C function call. Internally launches
 * cuBLAS GEMMs for matrix multiplications and custom CUDA sub-kernels
 * for LayerNorm, head transpose, GELU, softmax, and residual operations.
 *
 * Sub-kernel inventory:
 *   kernel_layernorm_fwd      Standard LayerNorm (mean + variance)
 *   kernel_bias_add           Broadcast bias addition along rows
 *   kernel_head_split         [B*S, H*D] -> [B*H, S, D] transpose
 *   kernel_head_merge         [B*H, S, D] -> [B*S, H*D] transpose
 *   kernel_scaled_softmax_fwd Row-wise scaled softmax
 *   kernel_bias_gelu_fwd      Fused bias add + GELU activation
 *   kernel_bias_residual_add  Fused bias add + residual connection
 *
 * cuBLAS calls (7 total per layer):
 *   3x Sgemm for Q/K/V projections
 *   1x SgemmStridedBatched for attention scores (Q @ K^T)
 *   1x SgemmStridedBatched for attention output  (scores @ V)
 *   1x Sgemm for output projection
 *   1x Sgemm for FFN1 projection
 *   1x Sgemm for FFN2 projection
 *
 * Compile: nvcc -O3 --use_fast_math -arch=sm_121 -lcublas -c fused_encoder_fwd.cu
 */

#include "fused_encoder_fwd.h"
#include <cublas_v2.h>
#include <math.h>
#include <string.h>  /* memcpy for bits_to_float */

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/* Minimum block size for reduction kernels. */
static inline int next_pow2(int v) {
    int b = 1;
    while (b < v && b < 256) b <<= 1;
    return b;
}

/* Row-major C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N] via cuBLAS.
 * cuBLAS is column-major; the standard trick swaps A/B and m/n. */
static inline cublasStatus_t sgemm_nn(
    cublasHandle_t h, int M, int N, int K, float alpha,
    const float* A, const float* B, float beta, float* C)
{
    return cublasSgemm(h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,   /* cuBLAS "A" = B_rm, lda = N */
        A, K,   /* cuBLAS "B" = A_rm, ldb = K */
        &beta,
        C, N);  /* cuBLAS "C", ldc = N */
}

/* Row-major batched C[b,M,N] = alpha * A[b,M,K] * B[b,K,N]^T + beta * C[b,M,N].
 * B is [b,N,K] (each batch element is [N,K], transposed to give [K,N]).
 * Used for attention: scores = Q @ K^T. */
static inline cublasStatus_t sgemm_nt_batched(
    cublasHandle_t h, int M, int N, int K, float alpha,
    const float* A, long long sA,
    const float* B, long long sB,
    float beta,
    float* C, long long sC,
    int batch)
{
    /* Row-major C = A * B^T.  Column-major: C^T = B * A^T.
     * cublas(transa=T, transb=N, m=N, n=M, k=K,
     *        A_cublas=A_rm[M,K]->cm[K,M] with transa=T -> [M,K],  NO wrong.
     *
     * Easier derivation:
     * Row-major X[p,q] in memory = col-major X'[q,p].
     * C_rm[M,N] = A_rm[M,K] * B_rm[N,K]^T
     * In col-major: C'[N,M] = (A * B^T)' = B * A'
     * B_rm[N,K] -> col-major B'[K,N].  We need "B" in cuBLAS sense -> B'[K,N] with op=N -> [K,N].
     *   BUT we need first dim = m=N.  So op=T -> [N,K].  Hmm.
     *
     * Let's just do it step by step:
     * cublasSgemm(h, transa, transb, m, n, k, alpha, Acub, lda, Bcub, ldb, beta, Ccub, ldc)
     * Ccub[m,n] = op(Acub)[m,k] * op(Bcub)[k,n]
     *
     * We want C'[N,M] = B_rm_as_cm * A_rm_as_cm
     * = B'[K,N]^T * A'[K,M]  with B' transposed = [N,K]
     * No, C'[N,M] = ?[N,k] * ?[k,M].  k=K.
     * First factor [N,K]: take B'[K,N] with CUBLAS_OP_T -> [N,K].  Acub=B_rm, lda=K. transa=T. m=N.
     * Second factor [K,M]: take A'[K,M] with CUBLAS_OP_N -> [K,M].  Bcub=A_rm, ldb=K. transb=N. n=M.
     * Ccub[N,M] at C_rm, ldc=N.
     */
    return cublasSgemmStridedBatched(h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, K, sB,    /* Acub = B_rm, lda=K, transa=T -> [N,K] */
        A, K, sA,    /* Bcub = A_rm, ldb=K, transb=N -> [K,M] */
        &beta,
        C, N, sC,
        batch);
}

/* Row-major batched C[b,M,N] = alpha * A[b,M,K] * B[b,K,N] + beta * C[b,M,N].
 * Standard NN batched multiply. */
static inline cublasStatus_t sgemm_nn_batched(
    cublasHandle_t h, int M, int N, int K, float alpha,
    const float* A, long long sA,
    const float* B, long long sB,
    float beta,
    float* C, long long sC,
    int batch)
{
    return cublasSgemmStridedBatched(h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N, sB,
        A, K, sA,
        &beta,
        C, N, sC,
        batch);
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: Standard LayerNorm forward                             */
/*  Each block processes one row of length D.                          */
/*  out[i] = (x[i] - mean) / sqrt(var + eps) * scale[i] + bias[i]    */
/*  Also writes invstd_out for backward use.                           */
/* ------------------------------------------------------------------ */

__global__ void kernel_layernorm_fwd(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ out,
    float* __restrict__ invstd_out,
    int D)
{
    int row = blockIdx.x;
    const float* xr = x + row * D;
    float* outr = out + row * D;

    extern __shared__ float smem[];

    /* Phase 1: compute mean. */
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        local_sum += xr[i];
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smem[0] / (float)D;

    /* Phase 2: compute variance. */
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float c = xr[i] - mean;
        local_var += c * c;
    }
    smem[threadIdx.x] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float invstd = rsqrtf(smem[0] / (float)D + 1e-5f);

    /* Store invstd for backward. */
    if (threadIdx.x == 0 && invstd_out != NULL) {
        invstd_out[row] = invstd;
    }

    /* Phase 3: normalize, scale, and bias. */
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        outr[i] = (xr[i] - mean) * invstd * scale[i] + bias[i];
    }
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: Broadcast bias addition.                               */
/*  out[i*cols + j] = x[i*cols + j] + bias[j]                         */
/* ------------------------------------------------------------------ */

__global__ void kernel_bias_add(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int n, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + bias[idx % cols];
    }
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: Head split transpose.                                  */
/*  [bsC*numPatches, nHeads*headDim] -> [bsC*nHeads, numPatches, headDim]  */
/*                                                                     */
/*  Input layout:  in[b*S + s][h*D + d]  (b=batch, s=seq, h=head, d=dim) */
/*  Output layout: out[(b*H + h)*S + s][d]                              */
/* ------------------------------------------------------------------ */

__global__ void kernel_head_split(
    const float* __restrict__ in,
    float* __restrict__ out,
    int bsC, int numPatches, int nHeads, int headDim)
{
    int total = bsC * numPatches * nHeads * headDim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int dModel = nHeads * headDim;
    /* Decode flat index -> (b, s, h, d) in input layout. */
    int d  = idx % headDim;
    int h  = (idx / headDim) % nHeads;
    int s  = (idx / dModel) % numPatches;
    int b  = idx / (numPatches * dModel);

    /* Input: in[(b * numPatches + s) * dModel + h * headDim + d] */
    int in_idx  = (b * numPatches + s) * dModel + h * headDim + d;
    /* Output: out[((b * nHeads + h) * numPatches + s) * headDim + d] */
    int out_idx = ((b * nHeads + h) * numPatches + s) * headDim + d;

    out[out_idx] = in[in_idx];
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: Head merge transpose (reverse of head_split).          */
/*  [bsC*nHeads, numPatches, headDim] -> [bsC*numPatches, nHeads*headDim]  */
/* ------------------------------------------------------------------ */

__global__ void kernel_head_merge(
    const float* __restrict__ in,
    float* __restrict__ out,
    int bsC, int numPatches, int nHeads, int headDim)
{
    int total = bsC * numPatches * nHeads * headDim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int dModel = nHeads * headDim;
    /* Decode flat index -> (b, h, s, d) in input layout. */
    int d  = idx % headDim;
    int s  = (idx / headDim) % numPatches;
    int h  = (idx / (numPatches * headDim)) % nHeads;
    int b  = idx / (nHeads * numPatches * headDim);

    /* Input:  in[((b * nHeads + h) * numPatches + s) * headDim + d] */
    int in_idx  = ((b * nHeads + h) * numPatches + s) * headDim + d;
    /* Output: out[(b * numPatches + s) * dModel + h * headDim + d] */
    int out_idx = (b * numPatches + s) * dModel + h * headDim + d;

    out[out_idx] = in[in_idx];
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: Row-wise softmax (in-place).                           */
/*  Each block handles one row of length cols.                         */
/*  out[i] = exp(in[i] - max) / sum(exp(in[j] - max))                 */
/* ------------------------------------------------------------------ */

__global__ void kernel_softmax_fwd(
    float* __restrict__ data,
    int cols)
{
    int row = blockIdx.x;
    float* r = data + row * cols;

    extern __shared__ float smem[];

    /* Find row max. */
    float local_max = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = r[i];
        if (v > local_max) local_max = v;
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = smem[0];

    /* Compute exp and sum. */
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(r[i] - row_max);
        r[i] = e;
        local_sum += e;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];

    /* Normalize. */
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        r[i] *= inv_sum;
    }
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: Fused bias add + GELU activation.                      */
/*  pre_act[i] = x[i] + bias[i % cols]                                */
/*  out[i] = 0.5 * pre_act * (1 + tanh(sqrt(2/pi) * (pre_act + 0.044715 * pre_act^3))) */
/* ------------------------------------------------------------------ */

__global__ void kernel_bias_gelu_fwd(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ pre_act_out,
    float* __restrict__ out,
    int n, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float v = x[idx] + bias[idx % cols];
    if (pre_act_out != NULL) pre_act_out[idx] = v;

    /* GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    float v3 = v * v * v;
    float inner = 0.7978845608f * (v + 0.044715f * v3);  /* sqrt(2/pi) ~ 0.7978845608 */
    out[idx] = 0.5f * v * (1.0f + tanhf(inner));
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: Fused bias add + residual connection.                  */
/*  out[i] = proj[i] + bias[i % cols] + residual[i]                   */
/* ------------------------------------------------------------------ */

__global__ void kernel_bias_residual_add(
    const float* __restrict__ proj,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int n, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    out[idx] = proj[idx] + bias[idx % cols] + residual[idx];
}

/* ------------------------------------------------------------------ */
/*  Orchestrator: fused encoder forward                                */
/* ------------------------------------------------------------------ */

extern "C" {

long long fused_encoder_fwd_scratch_bytes(
    int totalRows, int dModel, int nHeads, int headDim, int ffnDim,
    int bsC, int numPatches)
{
    long long bnh = (long long)bsC * nHeads;
    long long bytes = 0;
    /* FEB_NORMED1 */     bytes += (long long)totalRows * dModel * 4;
    /* FEB_LN1_INVSTD */  bytes += (long long)totalRows * 4;
    /* FEB_Q */           bytes += (long long)totalRows * dModel * 4;
    /* FEB_K */           bytes += (long long)totalRows * dModel * 4;
    /* FEB_V */           bytes += (long long)totalRows * dModel * 4;
    /* FEB_QH */          bytes += bnh * numPatches * headDim * 4;
    /* FEB_KH */          bytes += bnh * numPatches * headDim * 4;
    /* FEB_VH */          bytes += bnh * numPatches * headDim * 4;
    /* FEB_ATTN_SCORES */ bytes += bnh * numPatches * numPatches * 4;
    /* FEB_ATTN_OUT_H */  bytes += bnh * numPatches * headDim * 4;
    /* FEB_ATTN_OUT */    bytes += (long long)totalRows * dModel * 4;
    /* FEB_X_RES1 */      bytes += (long long)totalRows * dModel * 4;
    /* FEB_NORMED2 */     bytes += (long long)totalRows * dModel * 4;
    /* FEB_LN2_INVSTD */  bytes += (long long)totalRows * 4;
    /* FEB_FFN1_PRE */    bytes += (long long)totalRows * ffnDim * 4;
    /* FEB_FFN1_OUT */    bytes += (long long)totalRows * ffnDim * 4;
    return bytes;
}

cudaError_t fused_encoder_fwd_f32(
    void*        cublas_handle,
    const void** weights,
    void**       bufs,
    const float* input,
    float*       output,
    int totalRows, int dModel, int nHeads, int headDim, int ffnDim,
    int bsC, int numPatches,
    cudaStream_t stream)
{
    cublasHandle_t h = (cublasHandle_t)cublas_handle;
    cublasStatus_t blas_stat;

    /* Extract weight pointers. */
    const float* qW     = (const float*)weights[FEW_QW];
    const float* qB     = (const float*)weights[FEW_QB];
    const float* kW     = (const float*)weights[FEW_KW];
    const float* kB     = (const float*)weights[FEW_KB];
    const float* vW     = (const float*)weights[FEW_VW];
    const float* vB     = (const float*)weights[FEW_VB];
    const float* oW     = (const float*)weights[FEW_OW];
    const float* oB     = (const float*)weights[FEW_OB];
    const float* ffn1W  = (const float*)weights[FEW_FFN1W];
    const float* ffn1B  = (const float*)weights[FEW_FFN1B];
    const float* ffn2W  = (const float*)weights[FEW_FFN2W];
    const float* ffn2B  = (const float*)weights[FEW_FFN2B];
    const float* norm1W = (const float*)weights[FEW_NORM1W];
    const float* norm1B = (const float*)weights[FEW_NORM1B];
    const float* norm2W = (const float*)weights[FEW_NORM2W];
    const float* norm2B = (const float*)weights[FEW_NORM2B];

    /* Extract buffer pointers. */
    float* normed1    = (float*)bufs[FEB_NORMED1];
    float* ln1Invstd  = (float*)bufs[FEB_LN1_INVSTD];
    float* Q          = (float*)bufs[FEB_Q];
    float* K          = (float*)bufs[FEB_K];
    float* V          = (float*)bufs[FEB_V];
    float* Qh         = (float*)bufs[FEB_QH];
    float* Kh         = (float*)bufs[FEB_KH];
    float* Vh         = (float*)bufs[FEB_VH];
    float* attnScores = (float*)bufs[FEB_ATTN_SCORES];
    float* attnOutH   = (float*)bufs[FEB_ATTN_OUT_H];
    float* attnOut    = (float*)bufs[FEB_ATTN_OUT];
    float* xRes1      = (float*)bufs[FEB_X_RES1];
    float* normed2    = (float*)bufs[FEB_NORMED2];
    float* ln2Invstd  = (float*)bufs[FEB_LN2_INVSTD];
    float* ffn1Pre    = (float*)bufs[FEB_FFN1_PRE];
    float* ffn1Out    = (float*)bufs[FEB_FFN1_OUT];

    int bnh = bsC * nHeads;
    int trDm = totalRows * dModel;  /* total elements [totalRows, dModel] */
    int trFf = totalRows * ffnDim;  /* total elements [totalRows, ffnDim] */

    /* Common kernel launch params. */
    int block256 = 256;
    int elemGridTrDm = (trDm + block256 - 1) / block256;
    int elemGridTrFf = (trFf + block256 - 1) / block256;
    int totalElems = bsC * numPatches * nHeads * headDim;
    int elemGridTotal = (totalElems + block256 - 1) / block256;

    /* LayerNorm block size: next power of 2 up to min(dModel, 256). */
    int lnBlock = next_pow2(dModel);
    size_t lnSmem = lnBlock * sizeof(float);

    /* ------------------------------------------------------------ */
    /* Step 1: LayerNorm1                                            */
    /* ------------------------------------------------------------ */
    kernel_layernorm_fwd<<<totalRows, lnBlock, lnSmem, stream>>>(
        input, norm1W, norm1B, normed1, ln1Invstd, dModel);

    /* ------------------------------------------------------------ */
    /* Step 2: Q/K/V projections via cuBLAS + bias add               */
    /*   Q = normed1 @ qW + qB                                      */
    /*   K = normed1 @ kW + kB                                      */
    /*   V = normed1 @ vW + vB                                      */
    /* ------------------------------------------------------------ */

    /* Q = normed1 @ qW  (beta=0 to zero-initialize output) */
    blas_stat = sgemm_nn(h, totalRows, dModel, dModel, 1.0f, normed1, qW, 0.0f, Q);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    kernel_bias_add<<<elemGridTrDm, block256, 0, stream>>>(Q, qB, Q, trDm, dModel);

    /* K = normed1 @ kW + kB */
    blas_stat = sgemm_nn(h, totalRows, dModel, dModel, 1.0f, normed1, kW, 0.0f, K);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    kernel_bias_add<<<elemGridTrDm, block256, 0, stream>>>(K, kB, K, trDm, dModel);

    /* V = normed1 @ vW + vB */
    blas_stat = sgemm_nn(h, totalRows, dModel, dModel, 1.0f, normed1, vW, 0.0f, V);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    kernel_bias_add<<<elemGridTrDm, block256, 0, stream>>>(V, vB, V, trDm, dModel);

    /* ------------------------------------------------------------ */
    /* Step 3: Head split transpose for Q, K, V                      */
    /*   [bsC*numPatches, nHeads*headDim] -> [bsC*nHeads, numPatches, headDim] */
    /* ------------------------------------------------------------ */
    kernel_head_split<<<elemGridTotal, block256, 0, stream>>>(Q, Qh, bsC, numPatches, nHeads, headDim);
    kernel_head_split<<<elemGridTotal, block256, 0, stream>>>(K, Kh, bsC, numPatches, nHeads, headDim);
    kernel_head_split<<<elemGridTotal, block256, 0, stream>>>(V, Vh, bsC, numPatches, nHeads, headDim);

    /* ------------------------------------------------------------ */
    /* Step 4: Attention scores = (Qh @ Kh^T) / sqrt(headDim)       */
    /*   [bnh, numPatches, headDim] @ [bnh, headDim, numPatches]     */
    /*   -> [bnh, numPatches, numPatches]                            */
    /* ------------------------------------------------------------ */
    float attnScale = 1.0f / sqrtf((float)headDim);
    long long strideQK = (long long)numPatches * headDim;
    long long strideScores = (long long)numPatches * numPatches;

    blas_stat = sgemm_nt_batched(h,
        numPatches, numPatches, headDim,
        attnScale,
        Qh, strideQK,
        Kh, strideQK,
        0.0f,
        attnScores, strideScores,
        bnh);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* ------------------------------------------------------------ */
    /* Step 5: Softmax on attention scores (in-place)                */
    /*   Each row of [bnh * numPatches, numPatches]                  */
    /* ------------------------------------------------------------ */
    int sfBlock = next_pow2(numPatches);
    size_t sfSmem = sfBlock * sizeof(float);
    kernel_softmax_fwd<<<bnh * numPatches, sfBlock, sfSmem, stream>>>(
        attnScores, numPatches);

    /* ------------------------------------------------------------ */
    /* Step 6: Attention output = scores @ Vh                        */
    /*   [bnh, numPatches, numPatches] @ [bnh, numPatches, headDim]  */
    /*   -> [bnh, numPatches, headDim]                               */
    /* ------------------------------------------------------------ */
    blas_stat = sgemm_nn_batched(h,
        numPatches, headDim, numPatches,
        1.0f,
        attnScores, strideScores,
        Vh, strideQK,
        0.0f,
        attnOutH, strideQK,
        bnh);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* ------------------------------------------------------------ */
    /* Step 7: Head merge transpose                                  */
    /*   [bsC*nHeads, numPatches, headDim] -> [bsC*numPatches, nHeads*headDim] */
    /* ------------------------------------------------------------ */
    kernel_head_merge<<<elemGridTotal, block256, 0, stream>>>(attnOutH, attnOut, bsC, numPatches, nHeads, headDim);

    /* ------------------------------------------------------------ */
    /* Step 8: Output projection + bias + residual 1                 */
    /*   proj = attnOut @ oW                                         */
    /*   xRes1 = proj + oB + input   (fused bias + residual)         */
    /* ------------------------------------------------------------ */
    /* Use output buffer as temporary for projection result. */
    blas_stat = sgemm_nn(h, totalRows, dModel, dModel, 1.0f, attnOut, oW, 0.0f, output);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    kernel_bias_residual_add<<<elemGridTrDm, block256, 0, stream>>>(
        output, oB, input, xRes1, trDm, dModel);

    /* ------------------------------------------------------------ */
    /* Step 9: LayerNorm2                                            */
    /* ------------------------------------------------------------ */
    kernel_layernorm_fwd<<<totalRows, lnBlock, lnSmem, stream>>>(
        xRes1, norm2W, norm2B, normed2, ln2Invstd, dModel);

    /* ------------------------------------------------------------ */
    /* Step 10: FFN1 + bias + GELU                                   */
    /*   ffn1Pre = normed2 @ ffn1W  (linear projection)              */
    /*   ffn1Out = gelu(ffn1Pre + ffn1B)                             */
    /* ------------------------------------------------------------ */
    blas_stat = sgemm_nn(h, totalRows, ffnDim, dModel, 1.0f, normed2, ffn1W, 0.0f, ffn1Pre);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    kernel_bias_gelu_fwd<<<elemGridTrFf, block256, 0, stream>>>(
        ffn1Pre, ffn1B, ffn1Pre, ffn1Out, trFf, ffnDim);

    /* ------------------------------------------------------------ */
    /* Step 11: FFN2 + bias + residual 2                             */
    /*   proj = ffn1Out @ ffn2W                                      */
    /*   output = proj + ffn2B + xRes1                               */
    /* ------------------------------------------------------------ */
    blas_stat = sgemm_nn(h, totalRows, dModel, ffnDim, 1.0f, ffn1Out, ffn2W, 0.0f, output);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    kernel_bias_residual_add<<<elemGridTrDm, block256, 0, stream>>>(
        output, ffn2B, xRes1, output, trDm, dModel);

    return cudaGetLastError();
}

} /* extern "C" */

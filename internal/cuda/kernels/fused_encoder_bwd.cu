/* fused_encoder_bwd.cu -- Fused PatchTST encoder layer backward pass.
 *
 * Host-side orchestrator that computes all gradients for one encoder layer
 * in a single C function call. Uses the forward cache (FEB_* buffers)
 * to avoid recomputation.
 *
 * Sub-kernel inventory (backward-specific):
 *   kernel_layernorm_bwd       LayerNorm backward (dScale, dBias, dInput)
 *   kernel_gelu_bwd            GELU derivative * upstream gradient
 *   kernel_softmax_bwd         Softmax backward (Jacobian-vector product)
 *   kernel_bias_grad_reduce    Sum rows to compute bias gradients
 *   kernel_enc_bwd_add_elementwise     Element-wise addition for residual gradients
 *   kernel_matmul_grad_accum   Accumulate weight gradient: dW += A^T @ B
 *
 * cuBLAS calls (~14 total per layer):
 *   FFN2 backward:  dW, dInput via Sgemm (2 calls)
 *   FFN1 backward:  dW, dInput via Sgemm (2 calls)
 *   Output proj bwd: dW, dInput via Sgemm (2 calls)
 *   Attention bwd:  dV, dScores, dQ, dK via batched Sgemm (4 calls)
 *   Q/K/V bwd:     dW*3, dInput*3 via Sgemm (6 calls)
 *
 * Compile: nvcc -O3 --use_fast_math -arch=sm_121 -lcublas -c fused_encoder_bwd.cu
 */

#include "fused_encoder_bwd.h"
#include <cublas_v2.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Helpers (same as fwd, repeated for compilation unit independence)   */
/* ------------------------------------------------------------------ */

static inline int next_pow2_bwd(int v) {
    int b = 1;
    while (b < v && b < 256) b <<= 1;
    return b;
}

/* Row-major C[M,N] = alpha * A[M,K] * B[K,N] + beta * C */
static inline cublasStatus_t sgemm_nn(
    cublasHandle_t h, int M, int N, int K, float alpha,
    const float* A, const float* B, float beta, float* C)
{
    return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

/* Row-major C[M,N] = alpha * A^T[M,K] * B[K,N] + beta * C
 * A is [K,M] (transposed to give [M,K]). */
static inline cublasStatus_t sgemm_tn(
    cublasHandle_t h, int M, int N, int K, float alpha,
    const float* A, const float* B, float beta, float* C)
{
    /* Row-major A^T * B: column-major C^T = B^T * A
     * cublas(CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, alpha, B, N, A, M, beta, C, N)
     * Wait, A is [K,M] row-major = [M,K] col-major.
     * We want A^T[M,K] * B[K,N] in row-major.
     * Col-major: C^T[N,M] = B_cm * (A^T)_cm
     * B[K,N] row-major = B_cm[N,K] col-major.
     * A^T: A is [K,M] row-major = A_cm[M,K] col-major. A^T in col-major = [K,M].
     * So (A^T)_cm[K,M] with op=N -> [K,M].
     * We need: C_cm[N,M] = ?[N,k] * ?[k,M]
     * First: B_cm[N,K] with op=T -> [K,N]? No, we need [N,K].
     *   B_cm[N,K] with op=N -> [N,K]. m=N, k... doesn't match.
     * Let me use: cublas(transa, transb, m, n, k):
     *   C_cm[m,n] = op(A_cub)[m,k] * op(B_cub)[k,n]
     *   m=N, n=M, k=K
     *   op(A_cub)[N,K]: B row-major[K,N] -> col-major [N,K]. op=N. A_cub=B, lda=N. OK.
     *   op(B_cub)[K,M]: A row-major[K,M] -> col-major [M,K]. op=T -> [K,M]. B_cub=A, ldb=M. OK.
     */
    return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K, &alpha, B, N, A, M, &beta, C, N);
}

/* Row-major batched C[b,M,N] = alpha * A[b,M,K] * B[b,K,N]^T + beta * C */
static inline cublasStatus_t sgemm_nt_batched(
    cublasHandle_t h, int M, int N, int K, float alpha,
    const float* A, long long sA,
    const float* B, long long sB,
    float beta,
    float* C, long long sC,
    int batch)
{
    return cublasSgemmStridedBatched(h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K, &alpha,
        B, K, sB,
        A, K, sA,
        &beta,
        C, N, sC,
        batch);
}

/* Row-major batched C[b,M,N] = alpha * A[b,M,K] * B[b,K,N] + beta * C */
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
        N, M, K, &alpha,
        B, N, sB,
        A, K, sA,
        &beta,
        C, N, sC,
        batch);
}

/* Row-major batched C[b,M,N] = alpha * A^T[b,M,K] * B[b,K,N] + beta * C
 * A is [b,K,M], transposed per batch to [b,M,K]. */
static inline cublasStatus_t sgemm_tn_batched(
    cublasHandle_t h, int M, int N, int K, float alpha,
    const float* A, long long sA,
    const float* B, long long sB,
    float beta,
    float* C, long long sC,
    int batch)
{
    /* Same derivation as sgemm_tn but batched. */
    return cublasSgemmStridedBatched(h,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K, &alpha,
        B, N, sB,
        A, M, sA,
        &beta,
        C, N, sC,
        batch);
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: LayerNorm backward                                     */
/*  Given forward: y = (x - mean) * invstd * scale + bias             */
/*  Computes: dScale, dBias (accumulated), dX                         */
/*                                                                     */
/*  Each block handles one row.                                        */
/*  dScale[j] += sum_i (x[i,j] - mean[i]) * invstd[i] * dY[i,j]     */
/*  dBias[j]  += sum_i dY[i,j]                                        */
/*  dX[i,j]   = invstd * (scale * dY - mean(scale*dY) - centered *    */
/*              mean(scale*dY*centered) * invstd^2)                    */
/*                                                                     */
/*  Simplified: uses the standard LayerNorm backward formula.          */
/* ------------------------------------------------------------------ */

__global__ void kernel_layernorm_bwd(
    const float* __restrict__ dY,
    const float* __restrict__ x,
    const float* __restrict__ invstd,
    const float* __restrict__ scale,
    float* __restrict__ dX,
    float* __restrict__ dScale,  /* [D], atomicAdd */
    float* __restrict__ dBias,   /* [D], atomicAdd */
    int D)
{
    int row = blockIdx.x;
    const float* dy_r = dY + row * D;
    const float* x_r = x + row * D;
    float* dx_r = dX + row * D;
    float inv = invstd[row];

    extern __shared__ float smem[];

    /* Compute mean of x for this row. */
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        local_sum += x_r[i];
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smem[0] / (float)D;

    /* Compute dScale and dBias contributions (atomicAdd across rows). */
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float centered = (x_r[i] - mean) * inv;
        atomicAdd(&dScale[i], dy_r[i] * centered);
        atomicAdd(&dBias[i], dy_r[i]);
    }

    /* Compute dX using the standard LayerNorm backward formula:
     * ds = sum(dy * scale * centered)
     * dm = sum(dy * scale)
     * dX = inv * (scale * dY - dm/D - centered * ds/D) */
    float ds_local = 0.0f;
    float dm_local = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float centered = (x_r[i] - mean) * inv;
        float scaled_dy = dy_r[i] * scale[i];
        ds_local += scaled_dy * centered;
        dm_local += scaled_dy;
    }
    smem[threadIdx.x] = ds_local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float ds = smem[0];

    smem[threadIdx.x] = dm_local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float dm = smem[0];

    float inv_D = 1.0f / (float)D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float centered = (x_r[i] - mean) * inv;
        dx_r[i] = inv * (dy_r[i] * scale[i] - dm * inv_D - centered * ds * inv_D);
    }
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: GELU backward                                          */
/*  dX[i] = dY[i] * gelu'(pre_act[i])                                */
/*  where gelu'(x) = 0.5*(1+tanh(u)) + 0.5*x*sech^2(u)*du/dx        */
/*  u = sqrt(2/pi)*(x + 0.044715*x^3), du/dx = sqrt(2/pi)*(1+3*0.044715*x^2) */
/* ------------------------------------------------------------------ */

__global__ void kernel_gelu_bwd(
    const float* __restrict__ dY,
    const float* __restrict__ pre_act,
    float* __restrict__ dX,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = pre_act[idx];
    float x2 = x * x;
    float x3 = x2 * x;
    float sqrt2pi = 0.7978845608f;
    float c = 0.044715f;
    float u = sqrt2pi * (x + c * x3);
    float tanh_u = tanhf(u);
    float sech2 = 1.0f - tanh_u * tanh_u;
    float du_dx = sqrt2pi * (1.0f + 3.0f * c * x2);

    float gelu_deriv = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2 * du_dx;
    dX[idx] = dY[idx] * gelu_deriv;
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: Softmax backward                                       */
/*  Given forward: s = softmax(logits)                                 */
/*  dLogits[i] = s[i] * (dS[i] - sum_j(s[j] * dS[j]))               */
/*  Each block handles one row.                                        */
/* ------------------------------------------------------------------ */

__global__ void kernel_softmax_bwd(
    const float* __restrict__ scores,   /* softmax output from forward */
    const float* __restrict__ dScores,  /* upstream gradient */
    float* __restrict__ dLogits,
    int cols)
{
    int row = blockIdx.x;
    const float* s_r = scores + row * cols;
    const float* ds_r = dScores + row * cols;
    float* dl_r = dLogits + row * cols;

    extern __shared__ float smem[];

    /* Compute dot = sum(s * dS). */
    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_dot += s_r[i] * ds_r[i];
    }
    smem[threadIdx.x] = local_dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float dot = smem[0];

    /* dLogits = s * (dS - dot) */
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        dl_r[i] = s_r[i] * (ds_r[i] - dot);
    }
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: Bias gradient reduction                                */
/*  dBias[j] += sum_i dY[i*cols + j]                                  */
/*  Each block handles one column j.                                   */
/* ------------------------------------------------------------------ */

__global__ void kernel_bias_grad_reduce(
    const float* __restrict__ dY,
    float* __restrict__ dBias,
    int rows, int cols)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;

    float sum = 0.0f;
    for (int i = 0; i < rows; i++) {
        sum += dY[i * cols + j];
    }
    atomicAdd(&dBias[j], sum);
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: Element-wise addition.                                 */
/*  out[i] = a[i] + b[i]                                              */
/* ------------------------------------------------------------------ */

__global__ void kernel_enc_bwd_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

/* ------------------------------------------------------------------ */
/*  Sub-kernel: Three-way addition.                                    */
/*  out[i] = a[i] + b[i] + c[i]                                       */
/* ------------------------------------------------------------------ */

__global__ void kernel_enc_bwd_add3(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx] + c[idx];
    }
}

/* Forward-declare head transpose kernels from fused_encoder_fwd.cu.
 * Since both .cu files are compiled into the same libkernels.so,
 * these symbols are available at link time.
 * We redeclare the device-visible __global__ functions to launch them.
 * NOTE: We cannot forward-declare __global__ functions across compilation
 * units, so we duplicate the simple kernels here. */

__global__ void kernel_head_split_bwd(
    const float* __restrict__ in,
    float* __restrict__ out,
    int bsC, int numPatches, int nHeads, int headDim)
{
    int total = bsC * numPatches * nHeads * headDim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int dModel = nHeads * headDim;
    int d  = idx % headDim;
    int h  = (idx / headDim) % nHeads;
    int s  = (idx / dModel) % numPatches;
    int b  = idx / (numPatches * dModel);
    int in_idx  = (b * numPatches + s) * dModel + h * headDim + d;
    int out_idx = ((b * nHeads + h) * numPatches + s) * headDim + d;
    out[out_idx] = in[in_idx];
}

__global__ void kernel_head_merge_bwd(
    const float* __restrict__ in,
    float* __restrict__ out,
    int bsC, int numPatches, int nHeads, int headDim)
{
    int total = bsC * numPatches * nHeads * headDim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int dModel = nHeads * headDim;
    int d  = idx % headDim;
    int s  = (idx / headDim) % numPatches;
    int h  = (idx / (numPatches * headDim)) % nHeads;
    int b  = idx / (nHeads * numPatches * headDim);
    int in_idx  = ((b * nHeads + h) * numPatches + s) * headDim + d;
    int out_idx = (b * numPatches + s) * dModel + h * headDim + d;
    out[out_idx] = in[in_idx];
}

/* ------------------------------------------------------------------ */
/*  Orchestrator: fused encoder backward                               */
/* ------------------------------------------------------------------ */

extern "C" {

cudaError_t fused_encoder_bwd_f32(
    void*        cublas_handle,
    const void** weights,
    const void** weight_t,
    const void** fwd_bufs,
    void**       bwd_bufs,
    void**       grads,
    const float* dOutput,
    float*       dInput,
    const float* input,
    int totalRows, int dModel, int nHeads, int headDim, int ffnDim,
    int bsC, int numPatches,
    cudaStream_t stream)
{
    cublasHandle_t h = (cublasHandle_t)cublas_handle;
    cublasStatus_t blas_stat;

    /* Forward cache (read-only). */
    const float* normed1    = (const float*)fwd_bufs[FEB_NORMED1];
    const float* ln1Invstd  = (const float*)fwd_bufs[FEB_LN1_INVSTD];
    const float* fwd_Qh     = (const float*)fwd_bufs[FEB_QH];
    const float* fwd_Kh     = (const float*)fwd_bufs[FEB_KH];
    const float* fwd_Vh     = (const float*)fwd_bufs[FEB_VH];
    const float* attnScores = (const float*)fwd_bufs[FEB_ATTN_SCORES];
    const float* attnOut    = (const float*)fwd_bufs[FEB_ATTN_OUT];
    const float* xRes1      = (const float*)fwd_bufs[FEB_X_RES1];
    const float* normed2    = (const float*)fwd_bufs[FEB_NORMED2];
    const float* ln2Invstd  = (const float*)fwd_bufs[FEB_LN2_INVSTD];
    const float* ffn1Pre    = (const float*)fwd_bufs[FEB_FFN1_PRE];
    const float* ffn1Out    = (const float*)fwd_bufs[FEB_FFN1_OUT];

    /* Weight transposes (read-only). */
    const float* qWT    = (const float*)weight_t[FEWT_QWT];
    const float* kWT    = (const float*)weight_t[FEWT_KWT];
    const float* vWT    = (const float*)weight_t[FEWT_VWT];
    const float* oWT    = (const float*)weight_t[FEWT_OWT];
    const float* ffn1WT = (const float*)weight_t[FEWT_FFN1WT];
    const float* ffn2WT = (const float*)weight_t[FEWT_FFN2WT];

    /* Weights (for norm scale). */
    const float* norm1W = (const float*)weights[FEW_NORM1W];
    const float* norm2W = (const float*)weights[FEW_NORM2W];

    /* Backward scratch buffers. */
    float* dFfn1Out   = (float*)bwd_bufs[FEBB_DFFN1_OUT];
    float* dFfn1Pre   = (float*)bwd_bufs[FEBB_DFFN1_PRE];
    float* dNormed2   = (float*)bwd_bufs[FEBB_DNORMED2];
    float* dXRes1     = (float*)bwd_bufs[FEBB_DX_RES1];
    float* dAttnOut   = (float*)bwd_bufs[FEBB_DATTN_OUT];
    float* dAttnOutH  = (float*)bwd_bufs[FEBB_DATTN_OUT_H];
    float* dQh        = (float*)bwd_bufs[FEBB_DQH];
    float* dKh        = (float*)bwd_bufs[FEBB_DKH];
    float* dVh        = (float*)bwd_bufs[FEBB_DVH];
    float* dScoresBuf = (float*)bwd_bufs[FEBB_DSCORES];
    float* dQ         = (float*)bwd_bufs[FEBB_DQ];
    float* dK         = (float*)bwd_bufs[FEBB_DK];
    float* dV         = (float*)bwd_bufs[FEBB_DV];
    float* dNormed1   = (float*)bwd_bufs[FEBB_DNORMED1];
    float* temp       = (float*)bwd_bufs[FEBB_TEMP];

    /* Gradient accumulators (accumulated, not zeroed). */
    float* dg_qW     = (float*)grads[FEG_DQW];
    float* dg_qB     = (float*)grads[FEG_DQB];
    float* dg_kW     = (float*)grads[FEG_DKW];
    float* dg_kB     = (float*)grads[FEG_DKB];
    float* dg_vW     = (float*)grads[FEG_DVW];
    float* dg_vB     = (float*)grads[FEG_DVB];
    float* dg_oW     = (float*)grads[FEG_DOW];
    float* dg_oB     = (float*)grads[FEG_DOB];
    float* dg_ffn1W  = (float*)grads[FEG_DFFN1W];
    float* dg_ffn1B  = (float*)grads[FEG_DFFN1B];
    float* dg_ffn2W  = (float*)grads[FEG_DFFN2W];
    float* dg_ffn2B  = (float*)grads[FEG_DFFN2B];
    float* dg_norm1W = (float*)grads[FEG_DNORM1W];
    float* dg_norm1B = (float*)grads[FEG_DNORM1B];
    float* dg_norm2W = (float*)grads[FEG_DNORM2W];
    float* dg_norm2B = (float*)grads[FEG_DNORM2B];

    int bnh = bsC * nHeads;
    int trDm = totalRows * dModel;
    int trFf = totalRows * ffnDim;
    int block256 = 256;
    int elemGridTrDm = (trDm + block256 - 1) / block256;
    int elemGridTrFf = (trFf + block256 - 1) / block256;
    int totalElems = bsC * numPatches * nHeads * headDim;
    int elemGridTotal = (totalElems + block256 - 1) / block256;

    int lnBlock = next_pow2_bwd(dModel);
    size_t lnSmem = lnBlock * sizeof(float);

    long long strideQK = (long long)numPatches * headDim;
    long long strideScores = (long long)numPatches * numPatches;
    float attnScale = 1.0f / sqrtf((float)headDim);

    /* ============================================================ */
    /* The backward proceeds in reverse layer order.                 */
    /* dOutput is the gradient from the next layer (or loss).        */
    /* ============================================================ */

    /* ------------------------------------------------------------ */
    /* Step 1: Residual 2 backward                                   */
    /*   output = proj + ffn2B + xRes1                               */
    /*   dOutput flows to both FFN2 proj path AND xRes1 (skip)       */
    /* ------------------------------------------------------------ */
    /* dOutput is used directly for FFN2 backward and also added to  */
    /* the residual path later (Step 5).                             */

    /* ------------------------------------------------------------ */
    /* Step 2: FFN2 backward                                         */
    /*   proj = ffn1Out @ ffn2W                                      */
    /*   dW_ffn2 += ffn1Out^T @ dOutput     [ffnDim, dModel]        */
    /*   dB_ffn2 += sum(dOutput, axis=0)    [dModel]                 */
    /*   dFfn1Out = dOutput @ ffn2W^T       [totalRows, ffnDim]      */
    /* ------------------------------------------------------------ */

    /* dW_ffn2 += ffn1Out^T @ dOutput */
    blas_stat = sgemm_tn(h, ffnDim, dModel, totalRows, 1.0f,
        ffn1Out, dOutput, 1.0f, dg_ffn2W);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* dB_ffn2 += sum(dOutput, axis=0) */
    int biasGrid = (dModel + block256 - 1) / block256;
    kernel_bias_grad_reduce<<<biasGrid, block256, 0, stream>>>(
        dOutput, dg_ffn2B, totalRows, dModel);

    /* dFfn1Out = dOutput @ ffn2W^T */
    blas_stat = sgemm_nn(h, totalRows, ffnDim, dModel, 1.0f,
        dOutput, ffn2WT, 0.0f, dFfn1Out);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* ------------------------------------------------------------ */
    /* Step 3: GELU backward                                         */
    /*   dFfn1Pre = dFfn1Out * gelu'(ffn1Pre)                       */
    /* ------------------------------------------------------------ */
    kernel_gelu_bwd<<<elemGridTrFf, block256, 0, stream>>>(
        dFfn1Out, ffn1Pre, dFfn1Pre, trFf);

    /* ------------------------------------------------------------ */
    /* Step 4: FFN1 backward                                         */
    /*   proj = normed2 @ ffn1W                                      */
    /*   dW_ffn1 += normed2^T @ dFfn1Pre   [dModel, ffnDim]         */
    /*   dB_ffn1 += sum(dFfn1Pre, axis=0)  [ffnDim]                 */
    /*   dNormed2 = dFfn1Pre @ ffn1W^T     [totalRows, dModel]      */
    /* ------------------------------------------------------------ */
    blas_stat = sgemm_tn(h, dModel, ffnDim, totalRows, 1.0f,
        normed2, dFfn1Pre, 1.0f, dg_ffn1W);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    int ffnBiasGrid = (ffnDim + block256 - 1) / block256;
    kernel_bias_grad_reduce<<<ffnBiasGrid, block256, 0, stream>>>(
        dFfn1Pre, dg_ffn1B, totalRows, ffnDim);

    blas_stat = sgemm_nn(h, totalRows, dModel, ffnDim, 1.0f,
        dFfn1Pre, ffn1WT, 0.0f, dNormed2);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* ------------------------------------------------------------ */
    /* Step 5: LayerNorm2 backward + residual skip                   */
    /*   dXRes1_ln = layernorm_bwd(dNormed2, xRes1, ln2Invstd, norm2W) */
    /*   dXRes1 = dXRes1_ln + dOutput (residual skip from step 1)   */
    /* ------------------------------------------------------------ */
    kernel_layernorm_bwd<<<totalRows, lnBlock, lnSmem, stream>>>(
        dNormed2, xRes1, ln2Invstd, norm2W,
        dXRes1, dg_norm2W, dg_norm2B, dModel);

    /* Add residual skip: dXRes1 += dOutput */
    kernel_enc_bwd_add<<<elemGridTrDm, block256, 0, stream>>>(
        dXRes1, dOutput, dXRes1, trDm);

    /* ------------------------------------------------------------ */
    /* Step 6: Output projection backward                            */
    /*   proj = attnOut @ oW                                         */
    /*   dW_o += attnOut^T @ dXRes1       [dModel, dModel]           */
    /*   dB_o += sum(dXRes1, axis=0)      [dModel]                   */
    /*   dAttnOut = dXRes1 @ oW^T         [totalRows, dModel]        */
    /* ------------------------------------------------------------ */
    blas_stat = sgemm_tn(h, dModel, dModel, totalRows, 1.0f,
        attnOut, dXRes1, 1.0f, dg_oW);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    kernel_bias_grad_reduce<<<biasGrid, block256, 0, stream>>>(
        dXRes1, dg_oB, totalRows, dModel);

    blas_stat = sgemm_nn(h, totalRows, dModel, dModel, 1.0f,
        dXRes1, oWT, 0.0f, dAttnOut);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* ------------------------------------------------------------ */
    /* Step 7: Head split dAttnOut for multi-head backward           */
    /*   [totalRows, dModel] -> [bnh, numPatches, headDim]           */
    /* ------------------------------------------------------------ */
    kernel_head_split_bwd<<<elemGridTotal, block256, 0, stream>>>(
        dAttnOut, dAttnOutH, bsC, numPatches, nHeads, headDim);

    /* ------------------------------------------------------------ */
    /* Step 8: Attention backward                                    */
    /*   Forward: scores = softmax(Qh @ Kh^T / sqrt(d))             */
    /*           attnOutH = scores @ Vh                              */
    /*                                                               */
    /* 8a: dVh = scores^T @ dAttnOutH   [bnh, numPatches, headDim]  */
    /* 8b: dScoresRaw = dAttnOutH @ Vh^T [bnh, numPatches, numPatches] */
    /* 8c: dLogits = softmax_bwd(scores, dScoresRaw) * scale        */
    /* 8d: dQh = dLogits @ Kh           [bnh, numPatches, headDim]  */
    /* 8e: dKh = dLogits^T @ Qh         [bnh, numPatches, headDim]  */
    /* ------------------------------------------------------------ */

    /* 8a: dVh = scores^T @ dAttnOutH */
    blas_stat = sgemm_tn_batched(h,
        numPatches, headDim, numPatches, 1.0f,
        attnScores, strideScores,
        dAttnOutH, strideQK,
        0.0f,
        dVh, strideQK,
        bnh);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* 8b: dScoresRaw = dAttnOutH @ Vh^T */
    blas_stat = sgemm_nt_batched(h,
        numPatches, numPatches, headDim, 1.0f,
        dAttnOutH, strideQK,
        fwd_Vh, strideQK,
        0.0f,
        dScoresBuf, strideScores,
        bnh);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* 8c: Softmax backward + scale */
    int sfBlock = next_pow2_bwd(numPatches);
    size_t sfSmem = sfBlock * sizeof(float);
    kernel_softmax_bwd<<<bnh * numPatches, sfBlock, sfSmem, stream>>>(
        attnScores, dScoresBuf, dScoresBuf, numPatches);

    /* Apply attention scale to dLogits. The forward scaled Q@K^T by
     * 1/sqrt(d) before softmax, so the backward gradient through the
     * scale is just multiply by 1/sqrt(d). */
    {
        int scoreElems = bnh * numPatches * numPatches;
        int scoreGrid = (scoreElems + block256 - 1) / block256;
        /* Inline scale multiply kernel (reuse temp if needed). */
        /* We scale dScoresBuf in-place using a simple kernel. */
        /* For simplicity, fold scale into the cuBLAS calls below by
         * using alpha=attnScale. Actually, the scale was already applied
         * in the forward (scores = softmax(Q@K^T * scale)). The softmax
         * backward gives dLogits = softmax_bwd(dScoresRaw). The chain
         * rule through the scale multiply gives dScaled = dLogits * scale.
         * We apply this by setting alpha=attnScale in the dQ/dK GEMMs. */
        (void)scoreGrid;
    }

    /* 8d: dQh = attnScale * dLogits @ Kh */
    blas_stat = sgemm_nn_batched(h,
        numPatches, headDim, numPatches, attnScale,
        dScoresBuf, strideScores,
        fwd_Kh, strideQK,
        0.0f,
        dQh, strideQK,
        bnh);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* 8e: dKh = attnScale * dLogits^T @ Qh */
    blas_stat = sgemm_tn_batched(h,
        numPatches, headDim, numPatches, attnScale,
        dScoresBuf, strideScores,
        fwd_Qh, strideQK,
        0.0f,
        dKh, strideQK,
        bnh);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* ------------------------------------------------------------ */
    /* Step 9: Head merge dQh/dKh/dVh back to flat                   */
    /*   [bnh, numPatches, headDim] -> [totalRows, dModel]           */
    /* ------------------------------------------------------------ */
    kernel_head_merge_bwd<<<elemGridTotal, block256, 0, stream>>>(dQh, dQ, bsC, numPatches, nHeads, headDim);
    kernel_head_merge_bwd<<<elemGridTotal, block256, 0, stream>>>(dKh, dK, bsC, numPatches, nHeads, headDim);
    kernel_head_merge_bwd<<<elemGridTotal, block256, 0, stream>>>(dVh, dV, bsC, numPatches, nHeads, headDim);

    /* ------------------------------------------------------------ */
    /* Step 10: Q/K/V projection backward                            */
    /*   Q = normed1 @ qW + qB                                      */
    /*   dW_q += normed1^T @ dQ          [dModel, dModel]            */
    /*   dB_q += sum(dQ, axis=0)         [dModel]                    */
    /*   dNormed1_q = dQ @ qW^T          [totalRows, dModel]         */
    /*   (same for K, V; sum the three dNormed1 contributions)       */
    /* ------------------------------------------------------------ */

    /* Q backward */
    blas_stat = sgemm_tn(h, dModel, dModel, totalRows, 1.0f,
        normed1, dQ, 1.0f, dg_qW);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    kernel_bias_grad_reduce<<<biasGrid, block256, 0, stream>>>(
        dQ, dg_qB, totalRows, dModel);
    /* dNormed1 = dQ @ qW^T (first contribution) */
    blas_stat = sgemm_nn(h, totalRows, dModel, dModel, 1.0f,
        dQ, qWT, 0.0f, dNormed1);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* K backward */
    blas_stat = sgemm_tn(h, dModel, dModel, totalRows, 1.0f,
        normed1, dK, 1.0f, dg_kW);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    kernel_bias_grad_reduce<<<biasGrid, block256, 0, stream>>>(
        dK, dg_kB, totalRows, dModel);
    /* dNormed1 += dK @ kW^T (accumulate) */
    blas_stat = sgemm_nn(h, totalRows, dModel, dModel, 1.0f,
        dK, kWT, 1.0f, dNormed1);  /* beta=1 to accumulate */
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* V backward */
    blas_stat = sgemm_tn(h, dModel, dModel, totalRows, 1.0f,
        normed1, dV, 1.0f, dg_vW);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    kernel_bias_grad_reduce<<<biasGrid, block256, 0, stream>>>(
        dV, dg_vB, totalRows, dModel);
    /* dNormed1 += dV @ vW^T (accumulate) */
    blas_stat = sgemm_nn(h, totalRows, dModel, dModel, 1.0f,
        dV, vWT, 1.0f, dNormed1);  /* beta=1 to accumulate */
    if (blas_stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    /* ------------------------------------------------------------ */
    /* Step 11: LayerNorm1 backward + residual skip                  */
    /*   dLN1Input = layernorm_bwd(dNormed1, input, ln1Invstd, norm1W) */
    /*   dInput = dLN1Input + dXRes1  (residual skip from step 5)    */
    /* ------------------------------------------------------------ */
    /* Use temp for LN1 backward output, then add residual. */
    kernel_layernorm_bwd<<<totalRows, lnBlock, lnSmem, stream>>>(
        dNormed1, input, ln1Invstd, norm1W,
        temp, dg_norm1W, dg_norm1B, dModel);

    /* dInput = temp + dXRes1 */
    kernel_enc_bwd_add<<<elemGridTrDm, block256, 0, stream>>>(
        temp, dXRes1, dInput, trDm);

    return cudaGetLastError();
}

} /* extern "C" */

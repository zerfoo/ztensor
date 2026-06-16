// fused_norm_bf16.cu -- bfloat16 variants of the three forward-only fused
// normalization kernels (the bf16 analogues of fused_add_rmsnorm.cu,
// fused_norm_add.cu, and fused_qk_norm_rope.cu; ADR 075 lever L4).
//
// Parameters, weights and outputs are bf16 (2 bytes). ALL reductions
// (sum-of-squares for RMSNorm) and the normalization arithmetic accumulate in
// FP32 inside the kernel, then the result is rounded to bf16 on store. This
// matches the no-fast-math torch-numerics convention used by the f32 kernels
// and by elementwise_bf16.cu: the bf16 result equals round_to_bf16 of the
// FP32-accurate computation, which is the parity oracle the bf16-vs-f32 gate
// checks. bf16 shares f32's 8-bit exponent (same dynamic range), so these
// kernels do not reopen the ADR-072 forward-conditioning cliff; the 7-bit
// mantissa is the only precision difference vs f32.
//
// These are FORWARD-ONLY kernels: the f32 originals have no backward kernel, so
// there is no bf16 backward to mirror. GPU verification on GB10 is pending (the
// CUDA-gated parity tests in compute/ exercise these on a real device).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <string.h>

// ---------- Fused Add + RMSNorm (bf16) ----------
// sum_out[i]    = input[i] + residual[i]           (FP32 add, stored bf16)
// normed_out[i] = sum_out[i] / rms * weight[i]     (RMSNorm, FP32 accumulate)
//   where rms = sqrt(mean(sum^2) + eps)
// Each block handles one row of length D.

__global__ void kernel_fused_add_rmsnorm_bf16(const __nv_bfloat16* __restrict__ input,
                                              const __nv_bfloat16* __restrict__ residual,
                                              const __nv_bfloat16* __restrict__ weight,
                                              __nv_bfloat16* __restrict__ normed_out,
                                              __nv_bfloat16* __restrict__ sum_out,
                                              float eps, int D) {
    int row = blockIdx.x;
    const __nv_bfloat16* inp = input + row * D;
    const __nv_bfloat16* res = residual + row * D;
    __nv_bfloat16* nout = normed_out + row * D;
    __nv_bfloat16* sout = sum_out + row * D;

    extern __shared__ float sdata[];

    // Phase 1: sum = input + residual (FP32), store bf16, accumulate sum of
    // squares in FP32.
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __bfloat162float(inp[i]) + __bfloat162float(res[i]);
        sout[i] = __float2bfloat16(v);
        local_sq += v * v;
    }
    sdata[threadIdx.x] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float scale = rsqrtf(sdata[0] / (float)D + eps);

    // Phase 2: normalize the FP32 sum and scale by weight; round to bf16.
    // Re-derive the FP32 sum from the bf16 store so the normalization uses the
    // same rounded value the consumer sees (matches the f32 kernel, which reads
    // back sum_out[i]).
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __bfloat162float(sout[i]);
        nout[i] = __float2bfloat16(v * scale * __bfloat162float(weight[i]));
    }
}

// ---------- Fused RMSNorm + Add (bf16) ----------
// output[i] = rmsnorm(input, weight, eps)[i] + residual[i]
// Each block handles one row of length D.

__global__ void kernel_fused_norm_add_bf16(const __nv_bfloat16* __restrict__ input,
                                           const __nv_bfloat16* __restrict__ weight,
                                           const __nv_bfloat16* __restrict__ residual,
                                           __nv_bfloat16* __restrict__ output,
                                           float eps, int D) {
    int row = blockIdx.x;
    const __nv_bfloat16* x = input + row * D;
    const __nv_bfloat16* r = residual + row * D;
    __nv_bfloat16* o = output + row * D;

    extern __shared__ float smem[];

    // Phase 1: sum of squares (FP32 accumulate).
    float local_sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(x[d]);
        local_sq += v * v;
    }
    smem[threadIdx.x] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    float scale = rsqrtf(smem[0] / (float)D + eps);

    // Phase 2: normalize, multiply by weight, add residual; round to bf16.
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(x[d]) * scale * __bfloat162float(weight[d])
                  + __bfloat162float(r[d]);
        o[d] = __float2bfloat16(v);
    }
}

// ---------- Fused per-head QK RMSNorm + RoPE (bf16) ----------
// Per-head: RMSNorm(x, weight, eps) then RoPE(x, cos, sin).
// input/output/cos/sin: bf16 (cos/sin match the engine's generic tensor type T,
// exactly as the f32 kernel takes f32 angles for T=float32). The angles are
// upconverted to FP32 before the rotation so the RoPE arithmetic is full
// precision and only the final store rounds to bf16. Heads 0..numQHeads-1 use
// weightQ, the rest use weightK. Each block processes one head.

__global__ void kernel_fused_qk_norm_rope_bf16(const __nv_bfloat16* __restrict__ input,
                                               const __nv_bfloat16* __restrict__ weightQ,
                                               const __nv_bfloat16* __restrict__ weightK,
                                               const __nv_bfloat16* __restrict__ cosAngles,
                                               const __nv_bfloat16* __restrict__ sinAngles,
                                               __nv_bfloat16* __restrict__ output,
                                               float eps, int headDim, int numQHeads,
                                               int halfRotary) {
    int head = blockIdx.x;
    const __nv_bfloat16* x = input + head * headDim;
    __nv_bfloat16* o = output + head * headDim;
    const __nv_bfloat16* w = (head < numQHeads) ? weightQ : weightK;

    extern __shared__ float smem[];
    // smem layout: [headDim floats for normalized x] [blockDim.x floats for reduction]
    float* x_norm = smem;
    float* reduce = smem + headDim;

    // Phase 1: sum of squares (FP32 accumulate).
    float local_sq = 0.0f;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float v = __bfloat162float(x[d]);
        local_sq += v * v;
    }
    reduce[threadIdx.x] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            reduce[threadIdx.x] += reduce[threadIdx.x + s];
        }
        __syncthreads();
    }
    float scale = rsqrtf(reduce[0] / (float)headDim + eps);

    // Phase 2: normalize, hold in shared memory as FP32 (so RoPE mixes
    // full-precision normalized values before the single bf16 rounding on store).
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        x_norm[d] = __bfloat162float(x[d]) * scale * __bfloat162float(w[d]);
    }
    __syncthreads();

    // Phase 3: RoPE. First halfRotary dims rotate with the paired dim at
    // d+halfRotary; dims beyond 2*halfRotary are passthrough.
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        if (d < halfRotary) {
            float c = __bfloat162float(cosAngles[d]);
            float s_val = __bfloat162float(sinAngles[d]);
            float x0 = x_norm[d];
            float x1 = x_norm[d + halfRotary];
            o[d] = __float2bfloat16(x0 * c - x1 * s_val);
            o[d + halfRotary] = __float2bfloat16(x0 * s_val + x1 * c);
        } else if (d >= 2 * halfRotary) {
            o[d] = __float2bfloat16(x_norm[d]);
        }
        // d in [halfRotary, 2*halfRotary) is written by the d < halfRotary branch.
    }
}

// ---------- Launcher helpers ----------

static inline float bits_to_float_norm_bf16(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------- Launcher functions (extern "C" for CGO / purego) ----------
//
// ABI NOTE: matches the f32 fused norm kernels -- the purego dispatch marshals
// every argument through integer registers, so the f32 eps scalar crosses the
// boundary as its IEEE-754 bit pattern in an `unsigned int`.

extern "C" {

cudaError_t fused_add_rmsnorm_bf16(const void* input, const void* residual,
                                   const void* weight, void* normed_out,
                                   void* sum_out,
                                   unsigned int eps_bits,
                                   int rows, int D, cudaStream_t stream) {
    float eps = bits_to_float_norm_bf16(eps_bits);
    int block = 1;
    while (block < D && block < 256) block <<= 1;
    size_t smem = block * sizeof(float);
    kernel_fused_add_rmsnorm_bf16<<<rows, block, smem, stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)residual,
        (const __nv_bfloat16*)weight, (__nv_bfloat16*)normed_out,
        (__nv_bfloat16*)sum_out, eps, D);
    return cudaGetLastError();
}

cudaError_t fused_norm_add_bf16(const void* input, const void* weight,
                                const void* residual, void* output,
                                unsigned int eps_bits, int rows, int D,
                                cudaStream_t stream) {
    float eps = bits_to_float_norm_bf16(eps_bits);
    int block = 1;
    while (block < D && block < 256) block <<= 1;
    size_t smem = block * sizeof(float);
    kernel_fused_norm_add_bf16<<<rows, block, smem, stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)residual, (__nv_bfloat16*)output, eps, D);
    return cudaGetLastError();
}

cudaError_t fused_qk_norm_rope_bf16(const void* input, const void* weightQ,
                                    const void* weightK, const void* cosAngles,
                                    const void* sinAngles, void* output,
                                    unsigned int eps_bits,
                                    int totalHeads, int headDim, int numQHeads,
                                    int halfRotary, cudaStream_t stream) {
    float eps = bits_to_float_norm_bf16(eps_bits);
    int block = 1;
    while (block < headDim && block < 256) block <<= 1;
    // Shared memory: headDim (normalized x, FP32) + block (reduction buffer).
    size_t smem = (headDim + block) * sizeof(float);
    kernel_fused_qk_norm_rope_bf16<<<totalHeads, block, smem, stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)weightQ,
        (const __nv_bfloat16*)weightK, (const __nv_bfloat16*)cosAngles,
        (const __nv_bfloat16*)sinAngles, (__nv_bfloat16*)output,
        eps, headDim, numQHeads, halfRotary);
    return cudaGetLastError();
}

} // extern "C"

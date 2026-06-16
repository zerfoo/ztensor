// fused_adamw_bf16.cu -- CUDA kernel for the on-device AdamW step on bf16
// parameters (the bf16 analogue of fused_adamw.cu / ADR 070 end state, ADR 075
// lever L1).
//
// The parameter and gradient are bf16 (2 bytes), but the optimizer state and
// all arithmetic are kept in higher precision to preserve training stability:
//
//   param: bf16[n]   -- read and updated in place (round-to-bf16 on store)
//   grad:  bf16[n]   -- read, then zeroed in place
//   m:     f32[n]    -- first moment (f32 sidecar), updated in place
//   v:     f64[n]    -- second moment (f64 sidecar), updated in place
//
// Keeping m in f32 and v in f64 mirrors the f32 fused path: sqrt(v)+eps never
// collapses to eps under bf16/f32 underflow on near-zero gradients, and the
// master accumulation does not suffer bf16's 7-bit-mantissa rounding on every
// micro-step. Only the published parameter is bf16 -- the update itself is a
// full-precision AdamW step, then rounded to bf16 on write-back. This is the
// standard "bf16 weights + f32/f64 optimizer state" mixed-precision recipe.
//
// Bias-correction is folded into `alpha` and `lrWd` on the host exactly as in
// the f32 path, so the trajectories match modulo the bf16 weight rounding.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void kernel_fused_adamw_bf16(__nv_bfloat16* __restrict__ param,
                                        float* __restrict__ m,
                                        double* __restrict__ v,
                                        __nv_bfloat16* __restrict__ grad,
                                        double beta1, double beta2,
                                        double oneMinusBeta1, double oneMinusBeta2,
                                        double eps, double alpha, double lrWd,
                                        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double g = (double)__bfloat162float(grad[idx]);

    // First moment (kept in f32, matching the f32 path's mData[i] = f32(mNew)).
    double mOld = (double)__ldg(&m[idx]);
    double mNew = beta1 * mOld + oneMinusBeta1 * g;
    m[idx] = (float)mNew;

    // Second moment in f64 (the precision the stability fix needs).
    double vNew = beta2 * v[idx] + oneMinusBeta2 * g * g;
    v[idx] = vNew;

    // Full-precision update using the f64 first moment in the numerator, exactly
    // as stepMixedV / fused_adamw_f32 do.
    double denomI = sqrt(vNew) + eps;
    double update = alpha * mNew / denomI;

    double pv = (double)__bfloat162float(param[idx]);
    pv = pv - update - lrWd * pv;
    param[idx] = __float2bfloat16((float)pv);

    // Zero the gradient in place (kept device-resident, same buffer).
    grad[idx] = __float2bfloat16(0.0f);
}

// ---------- Launcher function (extern "C" for CGO / purego) ----------
//
// ABI NOTE: identical to fused_adamw.cu -- the purego dispatch marshals every
// argument through integer registers, so f64 scalars cross the boundary as
// their raw IEEE-754 bit patterns in `unsigned long long` params.

static inline double bits_to_double_bf16(unsigned long long b) {
    double d;
    __builtin_memcpy(&d, &b, sizeof(d));
    return d;
}

extern "C" {

cudaError_t fused_adamw_bf16(void* param, float* m, double* v, void* grad,
                             unsigned long long beta1Bits, unsigned long long beta2Bits,
                             unsigned long long oneMinusBeta1Bits, unsigned long long oneMinusBeta2Bits,
                             unsigned long long epsBits, unsigned long long alphaBits,
                             unsigned long long lrWdBits,
                             int n, cudaStream_t stream) {
    double beta1 = bits_to_double_bf16(beta1Bits);
    double beta2 = bits_to_double_bf16(beta2Bits);
    double oneMinusBeta1 = bits_to_double_bf16(oneMinusBeta1Bits);
    double oneMinusBeta2 = bits_to_double_bf16(oneMinusBeta2Bits);
    double eps = bits_to_double_bf16(epsBits);
    double alpha = bits_to_double_bf16(alphaBits);
    double lrWd = bits_to_double_bf16(lrWdBits);

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_fused_adamw_bf16<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)param, m, v, (__nv_bfloat16*)grad,
        beta1, beta2, oneMinusBeta1, oneMinusBeta2,
        eps, alpha, lrWd, n);
    return cudaGetLastError();
}

} // extern "C"

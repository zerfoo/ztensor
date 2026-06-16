// elementwise_bf16.cu -- bfloat16 CUDA kernels for elementwise tensor operations.
// Uses __nv_bfloat16/__nv_bfloat162 types from cuda_bf16.h for 2-wide SIMD
// processing. bf16 shares f32's 8-bit exponent (same dynamic range), so these
// kernels do NOT reopen the ADR-072 forward-conditioning cliff; the 7-bit
// mantissa is the only precision difference vs f32.
//
// Transcendental unary ops (tanh, sqrt, exp, log) and all reductions
// (scaled_softmax) accumulate in FP32 for precision, matching the FP16 kernels
// and the no-fast-math torch-numerics convention (see Makefile: NO global
// --use_fast_math; expf/tanhf/logf stay IEEE-accurate).

#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ---------- Binary elementwise (no broadcasting, __nv_bfloat162 SIMD) ----------

__global__ void kernel_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < n) {
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
        __nv_bfloat162* c2 = reinterpret_cast<__nv_bfloat162*>(c);
        c2[idx] = __hadd2(a2[idx], b2[idx]);
    } else if (idx2 < n) {
        c[idx2] = __hadd(a[idx2], b[idx2]);
    }
}

__global__ void kernel_sub_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < n) {
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
        __nv_bfloat162* c2 = reinterpret_cast<__nv_bfloat162*>(c);
        c2[idx] = __hsub2(a2[idx], b2[idx]);
    } else if (idx2 < n) {
        c[idx2] = __hsub(a[idx2], b[idx2]);
    }
}

__global__ void kernel_mul_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < n) {
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
        __nv_bfloat162* c2 = reinterpret_cast<__nv_bfloat162*>(c);
        c2[idx] = __hmul2(a2[idx], b2[idx]);
    } else if (idx2 < n) {
        c[idx2] = __hmul(a[idx2], b[idx2]);
    }
}

__global__ void kernel_div_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < n) {
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
        __nv_bfloat162* c2 = reinterpret_cast<__nv_bfloat162*>(c);
        c2[idx] = __h2div(a2[idx], b2[idx]);
    } else if (idx2 < n) {
        c[idx2] = __hdiv(a[idx2], b[idx2]);
    }
}

// ---------- Unary elementwise (FP32 transcendental, bf16 in/out) ----------
// tanh, sqrt, exp, log compute in FP32 then round to bf16. This matches the
// no-fast-math convention: the bf16 result equals round_to_bf16(f(f32(x))),
// which is the parity oracle the bf16-vs-f32 gate checks.

__global__ void kernel_tanh_bf16(const __nv_bfloat16* a, __nv_bfloat16* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2bfloat16(tanhf(__bfloat162float(a[idx])));
    }
}

__global__ void kernel_sqrt_bf16(const __nv_bfloat16* a, __nv_bfloat16* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2bfloat16(sqrtf(__bfloat162float(a[idx])));
    }
}

__global__ void kernel_rsqrt_bf16(const __nv_bfloat16* a, __nv_bfloat16* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2bfloat16(rsqrtf(__bfloat162float(a[idx])));
    }
}

__global__ void kernel_exp_bf16(const __nv_bfloat16* a, __nv_bfloat16* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2bfloat16(expf(__bfloat162float(a[idx])));
    }
}

__global__ void kernel_log_bf16(const __nv_bfloat16* a, __nv_bfloat16* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2bfloat16(logf(__bfloat162float(a[idx])));
    }
}

// ---------- Scaled Softmax bf16 ----------
// input/output are __nv_bfloat16, reductions in FP32 for precision.
// output = softmax(input * scale) along the axis.

__global__ void kernel_scaled_softmax_bf16(const __nv_bfloat16* input, __nv_bfloat16* output,
                                           int outer, int inner, int axisSize,
                                           float scale) {
    int stripe = blockIdx.x;
    int o = stripe / inner;
    int in_ = stripe % inner;
    int base = o * axisSize * inner + in_;
    int step = inner;

    extern __shared__ float sdata[];

    // Phase 1: Find max along axis (FP32 accumulation)
    float local_max = -INFINITY;
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        float val = __bfloat162float(input[base + k * step]) * scale;
        if (val > local_max) local_max = val;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float val = sdata[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (other > val) val = other;
        }
        if (threadIdx.x == 0) sdata[0] = val;
    }
    __syncthreads();
    float max_val = sdata[0];
    __syncthreads();

    // Phase 2: Compute exp(x * scale - max) and sum (FP32 accumulation)
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        int idx = base + k * step;
        float ex = expf(__bfloat162float(input[idx]) * scale - max_val);
        output[idx] = __float2bfloat16(ex);
        local_sum += ex;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float val = sdata[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) sdata[0] = val;
    }
    __syncthreads();
    float sum_val = sdata[0];
    __syncthreads();

    // Phase 3: Normalize (divide by FP32 sum, round result to bf16)
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        int idx = base + k * step;
        float v = __bfloat162float(output[idx]) / sum_val;
        output[idx] = __float2bfloat16(v);
    }
}

// ---------- F32 <-> BF16 conversion kernels ----------

__global__ void kernel_f32_to_bf16(const float* src, __nv_bfloat16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

__global__ void kernel_bf16_to_f32(const __nv_bfloat16* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

// ---------- Launcher helpers ----------

static inline void grid_config_bf16(int n, int* grid, int* block) {
    *block = 256;
    // Each thread handles 2 elements for binary ops.
    *grid = ((n + 1) / 2 + *block - 1) / *block;
}

static inline void grid_config_bf16_unary(int n, int* grid, int* block) {
    *block = 256;
    *grid = (n + *block - 1) / *block;
}

static inline float bits_to_float_bf16(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------- Launcher functions (extern "C" for CGO / purego) ----------

extern "C" {

cudaError_t launch_add_bf16(const void* a, const void* b, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_bf16(n, &grid, &block);
    kernel_add_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (__nv_bfloat16*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_sub_bf16(const void* a, const void* b, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_bf16(n, &grid, &block);
    kernel_sub_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (__nv_bfloat16*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_mul_bf16(const void* a, const void* b, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_bf16(n, &grid, &block);
    kernel_mul_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (__nv_bfloat16*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_div_bf16(const void* a, const void* b, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_bf16(n, &grid, &block);
    kernel_div_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (__nv_bfloat16*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_tanh_bf16(const void* a, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_bf16_unary(n, &grid, &block);
    kernel_tanh_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)a, (__nv_bfloat16*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_sqrt_bf16(const void* a, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_bf16_unary(n, &grid, &block);
    kernel_sqrt_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)a, (__nv_bfloat16*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_rsqrt_bf16(const void* a, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_bf16_unary(n, &grid, &block);
    kernel_rsqrt_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)a, (__nv_bfloat16*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_exp_bf16(const void* a, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_bf16_unary(n, &grid, &block);
    kernel_exp_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)a, (__nv_bfloat16*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_log_bf16(const void* a, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_bf16_unary(n, &grid, &block);
    kernel_log_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)a, (__nv_bfloat16*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_scaled_softmax_bf16(const void* input, void* output,
                                       int outer, int inner, int axisSize,
                                       unsigned int scale_bits,
                                       cudaStream_t stream) {
    float scale = bits_to_float_bf16(scale_bits);
    int block = 1;
    while (block < axisSize && block < 256) block <<= 1;
    int numStripes = outer * inner;
    size_t smem = block * sizeof(float);
    kernel_scaled_softmax_bf16<<<numStripes, block, smem, stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output, outer, inner, axisSize, scale);
    return cudaGetLastError();
}

cudaError_t launch_f32_to_bf16(const void* src, void* dst, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_f32_to_bf16<<<grid, block, 0, stream>>>((const float*)src, (__nv_bfloat16*)dst, n);
    return cudaGetLastError();
}

cudaError_t launch_bf16_to_f32(const void* src, void* dst, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_bf16_to_f32<<<grid, block, 0, stream>>>((const __nv_bfloat16*)src, (float*)dst, n);
    return cudaGetLastError();
}

} // extern "C"

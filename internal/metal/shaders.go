//go:build darwin

package metal

// combinedMSLSource contains all Metal compute shaders compiled into a single
// library. Each kernel function is named to match the Go dispatch call site.
const combinedMSLSource = `
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Element-wise binary ops: c[i] = op(a[i], b[i])
// ============================================================================

kernel void kernel_add(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float       *c [[buffer(2)]],
    constant uint      &n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = a[tid] + b[tid];
}

kernel void kernel_sub(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float       *c [[buffer(2)]],
    constant uint      &n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = a[tid] - b[tid];
}

kernel void kernel_mul(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float       *c [[buffer(2)]],
    constant uint      &n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = a[tid] * b[tid];
}

kernel void kernel_div(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float       *c [[buffer(2)]],
    constant uint      &n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = a[tid] / b[tid];
}

kernel void kernel_pow(
    device const float *base [[buffer(0)]],
    device const float *exp  [[buffer(1)]],
    device float       *c    [[buffer(2)]],
    constant uint      &n    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = pow(base[tid], exp[tid]);
}

// ============================================================================
// Element-wise unary ops: c[i] = op(a[i])
// ============================================================================

kernel void kernel_exp(
    device const float *a [[buffer(0)]],
    device float       *c [[buffer(1)]],
    constant uint      &n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = exp(a[tid]);
}

kernel void kernel_log(
    device const float *a [[buffer(0)]],
    device float       *c [[buffer(1)]],
    constant uint      &n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = log(a[tid]);
}

kernel void kernel_sqrt(
    device const float *a [[buffer(0)]],
    device float       *c [[buffer(1)]],
    constant uint      &n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = sqrt(a[tid]);
}

kernel void kernel_rsqrt(
    device const float *a [[buffer(0)]],
    device float       *c [[buffer(1)]],
    constant uint      &n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = rsqrt(a[tid]);
}

kernel void kernel_sin(
    device const float *a [[buffer(0)]],
    device float       *c [[buffer(1)]],
    constant uint      &n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = sin(a[tid]);
}

kernel void kernel_cos(
    device const float *a [[buffer(0)]],
    device float       *c [[buffer(1)]],
    constant uint      &n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = cos(a[tid]);
}

kernel void kernel_tanh(
    device const float *a [[buffer(0)]],
    device float       *c [[buffer(1)]],
    constant uint      &n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = tanh(a[tid]);
}

kernel void kernel_tanh_prime(
    device const float *a        [[buffer(0)]],
    device const float *upstream [[buffer(1)]],
    device float       *c        [[buffer(2)]],
    constant uint      &n        [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        float t = tanh(a[tid]);
        c[tid] = (1.0f - t * t) * upstream[tid];
    }
}

// ============================================================================
// Scalar ops: c[i] = op(a[i], scalar)
// ============================================================================

kernel void kernel_add_scalar(
    device const float *a      [[buffer(0)]],
    device float       *c      [[buffer(1)]],
    constant float     &scalar [[buffer(2)]],
    constant uint      &n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = a[tid] + scalar;
}

kernel void kernel_sub_scalar(
    device const float *a      [[buffer(0)]],
    device float       *c      [[buffer(1)]],
    constant float     &scalar [[buffer(2)]],
    constant uint      &n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = a[tid] - scalar;
}

kernel void kernel_mul_scalar(
    device const float *a      [[buffer(0)]],
    device float       *c      [[buffer(1)]],
    constant float     &scalar [[buffer(2)]],
    constant uint      &n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = a[tid] * scalar;
}

kernel void kernel_div_scalar(
    device const float *a      [[buffer(0)]],
    device float       *c      [[buffer(1)]],
    constant float     &scalar [[buffer(2)]],
    constant uint      &n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = a[tid] / scalar;
}

kernel void kernel_pow_scalar(
    device const float *a      [[buffer(0)]],
    device float       *c      [[buffer(1)]],
    constant float     &scalar [[buffer(2)]],
    constant uint      &n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = pow(a[tid], scalar);
}

// ============================================================================
// Fill: data[i] = value
// ============================================================================

kernel void kernel_fill(
    device float   *data  [[buffer(0)]],
    constant float &value [[buffer(1)]],
    constant uint  &n     [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) data[tid] = value;
}

// ============================================================================
// RMSNorm: output[r,d] = input[r,d] * rsqrt(mean(input[r,:]^2) + eps) * weight[d]
// One threadgroup per row, uses shared memory for parallel reduction.
// ============================================================================

kernel void kernel_rmsnorm(
    device const float *input   [[buffer(0)]],
    device const float *weight  [[buffer(1)]],
    device float       *output  [[buffer(2)]],
    device float       *scales  [[buffer(3)]],
    constant float     &eps     [[buffer(4)]],
    constant uint      &D       [[buffer(5)]],
    threadgroup float  *shared  [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    uint base = row * D;

    // Compute partial sum of squares.
    float partial = 0.0f;
    for (uint i = lid; i < D; i += tpg) {
        float v = input[base + i];
        partial += v * v;
    }
    shared[lid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_scale = rsqrt(shared[0] / float(D) + eps);
    if (lid == 0 && scales != nullptr) {
        scales[row] = rms_scale;
    }

    // Apply normalization.
    for (uint i = lid; i < D; i += tpg) {
        output[base + i] = input[base + i] * rms_scale * weight[i];
    }
}

// ============================================================================
// Softmax: per-row softmax along axis.
// input/output: [outer, axisSize, inner] — softmax over axisSize dim.
// One threadgroup per (outer, inner) pair.
// ============================================================================

kernel void kernel_softmax(
    device const float *input    [[buffer(0)]],
    device float       *output   [[buffer(1)]],
    constant uint      &outer    [[buffer(2)]],
    constant uint      &inner    [[buffer(3)]],
    constant uint      &axisSize [[buffer(4)]],
    threadgroup float  *shared   [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    uint o = gid / inner;
    uint i = gid % inner;
    if (o >= outer) return;

    uint stride = inner;

    // Find max.
    float maxVal = -INFINITY;
    for (uint k = lid; k < axisSize; k += tpg) {
        float v = input[o * axisSize * stride + k * stride + i];
        maxVal = max(maxVal, v);
    }
    shared[lid] = maxVal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rowMax = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute exp sum.
    float expSum = 0.0f;
    for (uint k = lid; k < axisSize; k += tpg) {
        expSum += exp(input[o * axisSize * stride + k * stride + i] - rowMax);
    }
    shared[lid] = expSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = shared[0];

    // Write output.
    for (uint k = lid; k < axisSize; k += tpg) {
        uint idx = o * axisSize * stride + k * stride + i;
        output[idx] = exp(input[idx] - rowMax) / sum;
    }
}

// ============================================================================
// ScaledSoftmax: softmax(input * scale)
// ============================================================================

kernel void kernel_scaled_softmax(
    device const float *input    [[buffer(0)]],
    device float       *output   [[buffer(1)]],
    constant uint      &outer    [[buffer(2)]],
    constant uint      &inner    [[buffer(3)]],
    constant uint      &axisSize [[buffer(4)]],
    constant float     &scale    [[buffer(5)]],
    threadgroup float  *shared   [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    uint o = gid / inner;
    uint i = gid % inner;
    if (o >= outer) return;

    uint stride = inner;

    // Find max of scaled values.
    float maxVal = -INFINITY;
    for (uint k = lid; k < axisSize; k += tpg) {
        float v = input[o * axisSize * stride + k * stride + i] * scale;
        maxVal = max(maxVal, v);
    }
    shared[lid] = maxVal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rowMax = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute exp sum.
    float expSum = 0.0f;
    for (uint k = lid; k < axisSize; k += tpg) {
        expSum += exp(input[o * axisSize * stride + k * stride + i] * scale - rowMax);
    }
    shared[lid] = expSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = shared[0];

    for (uint k = lid; k < axisSize; k += tpg) {
        uint idx = o * axisSize * stride + k * stride + i;
        output[idx] = exp(input[idx] * scale - rowMax) / sum;
    }
}

// ============================================================================
// FusedRoPE: Rotary Positional Embedding
// input/output: [batch * seqLen * headDim]
// cos/sin: [seqLen * cosStride]
// ============================================================================

kernel void kernel_fused_rope(
    device const float *input     [[buffer(0)]],
    device const float *cosAngles [[buffer(1)]],
    device const float *sinAngles [[buffer(2)]],
    device float       *output    [[buffer(3)]],
    constant uint      &batch     [[buffer(4)]],
    constant uint      &seqLen    [[buffer(5)]],
    constant uint      &headDim   [[buffer(6)]],
    constant uint      &halfRotary [[buffer(7)]],
    constant uint      &cosStride  [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = batch * seqLen * headDim;
    if (tid >= total) return;

    uint d = tid % headDim;
    uint pos = (tid / headDim) % seqLen;

    if (d < halfRotary) {
        float c = cosAngles[pos * cosStride + d];
        float s = sinAngles[pos * cosStride + d];
        float x0 = input[tid];
        float x1 = input[tid + halfRotary];
        output[tid] = x0 * c - x1 * s;
        output[tid + halfRotary] = x0 * s + x1 * c;
    } else if (d >= 2 * halfRotary) {
        // Pass-through for non-rotary dimensions.
        output[tid] = input[tid];
    }
}

// ============================================================================
// FusedSwiGLU: output[i] = w1[i] * sigmoid(w1[i]) * w3[i]
// ============================================================================

kernel void kernel_fused_swiglu(
    device const float *w1     [[buffer(0)]],
    device const float *w3     [[buffer(1)]],
    device float       *output [[buffer(2)]],
    constant uint      &n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        float x = w1[tid];
        output[tid] = x / (1.0f + exp(-x)) * w3[tid];
    }
}

// ============================================================================
// FusedAddRMSNorm: sum = input + residual, output = rmsnorm(sum, weight, eps)
// One threadgroup per row.
// ============================================================================

kernel void kernel_fused_add_rmsnorm(
    device const float *input    [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    device const float *weight   [[buffer(2)]],
    device float       *normedOut [[buffer(3)]],
    device float       *sumOut    [[buffer(4)]],
    constant float     &eps       [[buffer(5)]],
    constant uint      &D         [[buffer(6)]],
    threadgroup float  *shared    [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    uint base = row * D;

    // First pass: compute sum and partial sum-of-squares.
    float partial = 0.0f;
    for (uint i = lid; i < D; i += tpg) {
        float s = input[base + i] + residual[base + i];
        sumOut[base + i] = s;
        partial += s * s;
    }
    shared[lid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_scale = rsqrt(shared[0] / float(D) + eps);

    // Second pass: normalize.
    for (uint i = lid; i < D; i += tpg) {
        normedOut[base + i] = sumOut[base + i] * rms_scale * weight[i];
    }
}

// ============================================================================
// FusedNormAdd: output = rmsnorm(input, weight, eps) + residual
// One threadgroup per row.
// ============================================================================

kernel void kernel_fused_norm_add(
    device const float *input    [[buffer(0)]],
    device const float *weight   [[buffer(1)]],
    device const float *residual [[buffer(2)]],
    device float       *output   [[buffer(3)]],
    constant float     &eps      [[buffer(4)]],
    constant uint      &D        [[buffer(5)]],
    threadgroup float  *shared   [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    uint base = row * D;

    float partial = 0.0f;
    for (uint i = lid; i < D; i += tpg) {
        float v = input[base + i];
        partial += v * v;
    }
    shared[lid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_scale = rsqrt(shared[0] / float(D) + eps);

    for (uint i = lid; i < D; i += tpg) {
        output[base + i] = input[base + i] * rms_scale * weight[i] + residual[base + i];
    }
}

// ============================================================================
// SgemvM1: y = A * x, where A is [M, N] and x is [N], y is [M].
// One threadgroup per row (M).
// ============================================================================

kernel void kernel_sgemv_m1(
    device float       *y      [[buffer(0)]],
    device const float *A      [[buffer(1)]],
    device const float *x      [[buffer(2)]],
    constant uint      &M      [[buffer(3)]],
    constant uint      &N      [[buffer(4)]],
    threadgroup float  *shared [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    if (row >= M) return;
    uint base = row * N;

    float partial = 0.0f;
    for (uint j = lid; j < N; j += tpg) {
        partial += A[base + j] * x[j];
    }
    shared[lid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) y[row] = shared[0];
}

// ============================================================================
// Gather: output[i,:] = table[indices[i],:] (embedding lookup)
// ============================================================================

kernel void kernel_gather(
    device const float  *table   [[buffer(0)]],
    device const long   *indices [[buffer(1)]],
    device float        *output  [[buffer(2)]],
    constant uint       &N       [[buffer(3)]],
    constant uint       &D       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = N * D;
    if (tid >= total) return;
    uint i = tid / D;
    uint d = tid % D;
    long idx = indices[i];
    output[tid] = table[idx * D + d];
}

// ============================================================================
// Transpose2D: out[c,r] = in[r,c]
// ============================================================================

kernel void kernel_transpose2d(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    constant uint      &rows   [[buffer(2)]],
    constant uint      &cols   [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = rows * cols;
    if (tid >= total) return;
    uint r = tid / cols;
    uint c = tid % cols;
    output[c * rows + r] = input[tid];
}

// ============================================================================
// SumAxis: output[outer][inner] = sum(input[outer][k][inner], k=0..axisSize-1)
// ============================================================================

kernel void kernel_sum_axis(
    device const float *input    [[buffer(0)]],
    device float       *output   [[buffer(1)]],
    constant uint      &outer    [[buffer(2)]],
    constant uint      &inner    [[buffer(3)]],
    constant uint      &axisSize [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = outer * inner;
    if (tid >= total) return;
    uint o = tid / inner;
    uint i = tid % inner;

    float sum = 0.0f;
    for (uint k = 0; k < axisSize; k++) {
        sum += input[o * axisSize * inner + k * inner + i];
    }
    output[tid] = sum;
}

// ============================================================================
// Repeat: replicate along axis.
// ============================================================================

kernel void kernel_repeat(
    device const float *src       [[buffer(0)]],
    device float       *dst       [[buffer(1)]],
    constant uint      &outerSize [[buffer(2)]],
    constant uint      &axisDim   [[buffer(3)]],
    constant uint      &innerSize [[buffer(4)]],
    constant uint      &reps      [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = outerSize * axisDim * reps * innerSize;
    if (tid >= total) return;

    uint inner = tid % innerSize;
    uint tmp = tid / innerSize;
    uint repAxis = tmp % (axisDim * reps);
    uint outer = tmp / (axisDim * reps);
    uint srcAxis = repAxis % axisDim;
    uint srcIdx = outer * axisDim * innerSize + srcAxis * innerSize + inner;
    dst[tid] = src[srcIdx];
}

// ============================================================================
// 2D Broadcast ops: c[r,c] = op(a[r*saRow+c*saCol], b[r*sbRow+c*sbCol])
// ============================================================================

kernel void kernel_add_broadcast(
    device const float *a     [[buffer(0)]],
    device const float *b     [[buffer(1)]],
    device float       *c     [[buffer(2)]],
    constant uint      &saRow [[buffer(3)]],
    constant uint      &saCol [[buffer(4)]],
    constant uint      &sbRow [[buffer(5)]],
    constant uint      &sbCol [[buffer(6)]],
    constant uint      &M     [[buffer(7)]],
    constant uint      &D     [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = M * D;
    if (tid >= total) return;
    uint r = tid / D;
    uint col = tid % D;
    c[tid] = a[r * saRow + col * saCol] + b[r * sbRow + col * sbCol];
}

kernel void kernel_sub_broadcast(
    device const float *a     [[buffer(0)]],
    device const float *b     [[buffer(1)]],
    device float       *c     [[buffer(2)]],
    constant uint      &saRow [[buffer(3)]],
    constant uint      &saCol [[buffer(4)]],
    constant uint      &sbRow [[buffer(5)]],
    constant uint      &sbCol [[buffer(6)]],
    constant uint      &M     [[buffer(7)]],
    constant uint      &D     [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = M * D;
    if (tid >= total) return;
    uint r = tid / D;
    uint col = tid % D;
    c[tid] = a[r * saRow + col * saCol] - b[r * sbRow + col * sbCol];
}

kernel void kernel_mul_broadcast(
    device const float *a     [[buffer(0)]],
    device const float *b     [[buffer(1)]],
    device float       *c     [[buffer(2)]],
    constant uint      &saRow [[buffer(3)]],
    constant uint      &saCol [[buffer(4)]],
    constant uint      &sbRow [[buffer(5)]],
    constant uint      &sbCol [[buffer(6)]],
    constant uint      &M     [[buffer(7)]],
    constant uint      &D     [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = M * D;
    if (tid >= total) return;
    uint r = tid / D;
    uint col = tid % D;
    c[tid] = a[r * saRow + col * saCol] * b[r * sbRow + col * sbCol];
}

kernel void kernel_div_broadcast(
    device const float *a     [[buffer(0)]],
    device const float *b     [[buffer(1)]],
    device float       *c     [[buffer(2)]],
    constant uint      &saRow [[buffer(3)]],
    constant uint      &saCol [[buffer(4)]],
    constant uint      &sbRow [[buffer(5)]],
    constant uint      &sbCol [[buffer(6)]],
    constant uint      &M     [[buffer(7)]],
    constant uint      &D     [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = M * D;
    if (tid >= total) return;
    uint r = tid / D;
    uint col = tid % D;
    c[tid] = a[r * saRow + col * saCol] / b[r * sbRow + col * sbCol];
}

// ============================================================================
// Argmax: find index of max in float32 array.
// Two-pass: first reduce to per-block max+idx, then final reduce.
// ============================================================================

kernel void kernel_argmax_pass1(
    device const float *input  [[buffer(0)]],
    device float       *vals   [[buffer(1)]],
    device int         *idxs   [[buffer(2)]],
    constant uint      &n      [[buffer(3)]],
    threadgroup float  *svals  [[threadgroup(0)]],
    threadgroup int    *sidxs  [[threadgroup(1)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    float bestVal = -INFINITY;
    int bestIdx = 0;
    for (uint i = gid * tpg + lid; i < n; i += tpg * ((n + tpg - 1) / tpg)) {
        if (i < n && input[i] > bestVal) {
            bestVal = input[i];
            bestIdx = int(i);
        }
    }
    svals[lid] = bestVal;
    sidxs[lid] = bestIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) {
            if (svals[lid + s] > svals[lid]) {
                svals[lid] = svals[lid + s];
                sidxs[lid] = sidxs[lid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        vals[gid] = svals[0];
        idxs[gid] = sidxs[0];
    }
}

kernel void kernel_argmax_pass2(
    device const float *vals   [[buffer(0)]],
    device const int   *idxs   [[buffer(1)]],
    device int         *result [[buffer(2)]],
    constant uint      &nBlocks [[buffer(3)]],
    threadgroup float  *svals  [[threadgroup(0)]],
    threadgroup int    *sidxs  [[threadgroup(1)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    float bestVal = -INFINITY;
    int bestIdx = 0;
    for (uint i = lid; i < nBlocks; i += tpg) {
        if (vals[i] > bestVal) {
            bestVal = vals[i];
            bestIdx = idxs[i];
        }
    }
    svals[lid] = bestVal;
    sidxs[lid] = bestIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) {
            if (svals[lid + s] > svals[lid]) {
                svals[lid] = svals[lid + s];
                sidxs[lid] = sidxs[lid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) result[0] = sidxs[0];
}

// ============================================================================
// FusedQKNormRoPE: per-head RMSNorm + RoPE for Q and K heads.
// input: [totalHeads, headDim], output: [totalHeads, headDim]
// One threadgroup per head.
// ============================================================================

kernel void kernel_fused_qk_norm_rope(
    device const float *input      [[buffer(0)]],
    device const float *weightQ    [[buffer(1)]],
    device const float *weightK    [[buffer(2)]],
    device const float *cosAngles  [[buffer(3)]],
    device const float *sinAngles  [[buffer(4)]],
    device float       *output     [[buffer(5)]],
    constant float     &eps        [[buffer(6)]],
    constant uint      &totalHeads [[buffer(7)]],
    constant uint      &headDim    [[buffer(8)]],
    constant uint      &numQHeads  [[buffer(9)]],
    constant uint      &halfRotary [[buffer(10)]],
    threadgroup float  *shared     [[threadgroup(0)]],
    uint head [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]]
) {
    if (head >= totalHeads) return;
    uint base = head * headDim;
    bool isQ = (head < numQHeads);
    device const float *w = isQ ? weightQ : weightK;

    // RMSNorm reduction.
    float partial = 0.0f;
    for (uint i = lid; i < headDim; i += tpg) {
        float v = input[base + i];
        partial += v * v;
    }
    shared[lid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_scale = rsqrt(shared[0] / float(headDim) + eps);

    // Apply norm then RoPE.
    for (uint i = lid; i < headDim; i += tpg) {
        float normed = input[base + i] * rms_scale * w[i];
        if (i < halfRotary) {
            float normed2 = input[base + i + halfRotary] * rms_scale * w[i + halfRotary];
            float c = cosAngles[i];
            float s = sinAngles[i];
            output[base + i] = normed * c - normed2 * s;
            output[base + i + halfRotary] = normed * s + normed2 * c;
        } else if (i >= 2 * halfRotary) {
            output[base + i] = normed;
        }
    }
}
`

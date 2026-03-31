// fused_softmax_vmul.cu -- CUDA kernel for fused softmax + V multiply.
// Computes output = softmax(scores * scale) @ V in a single kernel launch,
// avoiding materialization of the attention weights tensor.
//
// Decode-optimized (seqQ=1): each block handles one (batch*head) pair.
// scores: [BH, 1, seqKV]  (row vector per head)
// V:      [BH, seqKV, D]
// output: [BH, 1, D]
//
// Algorithm per block:
//   1. Load scores into shared memory, applying scale
//   2. Parallel reduction for max (numerical stability)
//   3. Subtract max, exponentiate, parallel reduction for sum
//   4. Normalize to get softmax weights in shared memory
//   5. For each d in [0, D): compute weighted sum over seqKV
//   6. Write output[bh, 0, d]

#include <cuda_runtime.h>
#include <math.h>

__global__ void kernel_fused_softmax_vmul(
    const float* __restrict__ scores,   // [BH, seqKV]
    const float* __restrict__ V,        // [BH, seqKV, D]
    float* __restrict__ output,         // [BH, D]
    float scale,
    int seqKV,
    int D
) {
    int bh = blockIdx.x;  // one block per (batch, head) pair

    extern __shared__ float sdata[];
    // sdata[0..blockDim.x-1] used for reductions
    // sdata[blockDim.x..blockDim.x+seqKV-1] used for softmax weights

    float* smem_reduce = sdata;
    float* smem_weights = sdata + blockDim.x;

    const float* my_scores = scores + bh * seqKV;
    const float* my_V = V + bh * seqKV * D;
    float* my_output = output + bh * D;

    // Phase 1: Load scores into shared memory (with scale) and find max.
    float local_max = -INFINITY;
    for (int k = threadIdx.x; k < seqKV; k += blockDim.x) {
        float val = __ldg(&my_scores[k]) * scale;
        smem_weights[k] = val;
        if (val > local_max) local_max = val;
    }
    smem_reduce[threadIdx.x] = local_max;
    __syncthreads();

    // Inter-warp max reduction via shared memory.
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            if (smem_reduce[threadIdx.x + s] > smem_reduce[threadIdx.x])
                smem_reduce[threadIdx.x] = smem_reduce[threadIdx.x + s];
        }
        __syncthreads();
    }
    // Final warp max reduction using __shfl_down_sync.
    if (threadIdx.x < 32) {
        float val = smem_reduce[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (other > val) val = other;
        }
        if (threadIdx.x == 0) smem_reduce[0] = val;
    }
    __syncthreads();
    float max_val = smem_reduce[0];
    __syncthreads();

    // Phase 2: Subtract max, exponentiate, accumulate sum.
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < seqKV; k += blockDim.x) {
        float ex = expf(smem_weights[k] - max_val);
        smem_weights[k] = ex;
        local_sum += ex;
    }
    smem_reduce[threadIdx.x] = local_sum;
    __syncthreads();

    // Inter-warp sum reduction via shared memory.
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            smem_reduce[threadIdx.x] += smem_reduce[threadIdx.x + s];
        }
        __syncthreads();
    }
    // Final warp sum reduction using __shfl_down_sync.
    if (threadIdx.x < 32) {
        float val = smem_reduce[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) smem_reduce[0] = val;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem_reduce[0];
    __syncthreads();

    // Phase 3: Normalize weights in shared memory.
    for (int k = threadIdx.x; k < seqKV; k += blockDim.x) {
        smem_weights[k] *= inv_sum;
    }
    __syncthreads();

    // Phase 4: Weighted sum over V for each head dimension d.
    // Each thread handles a subset of d values.
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < seqKV; k++) {
            acc += smem_weights[k] * __ldg(&my_V[k * D + d]);
        }
        my_output[d] = acc;
    }
}

// bits_to_float reinterprets a uint32 bit pattern as float32.
// Used because the purego/ccall calling convention passes all arguments
// through integer registers.
static inline float bits_to_float(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------- Launcher function (extern "C" for CGO / purego) ----------

extern "C" {

cudaError_t launch_fused_softmax_vmul_f32(
    const float* scores,    // [BH, 1, seqKV]
    const float* V,         // [BH, seqKV, D]
    float* output,          // [BH, 1, D]
    unsigned int scale_bits,// float scale as uint32 bits
    int BH,                 // batch * heads
    int seqKV,              // key/value sequence length
    int D,                  // head dimension
    cudaStream_t stream
) {
    float scale = bits_to_float(scale_bits);

    // Block size: next power of 2 up to min(max(seqKV, D), 256).
    int maxDim = seqKV > D ? seqKV : D;
    int block = 1;
    while (block < maxDim && block < 256) block <<= 1;

    // Shared memory: blockDim.x floats for reductions + seqKV floats for weights.
    size_t smem = (block + seqKV) * sizeof(float);

    kernel_fused_softmax_vmul<<<BH, block, smem, stream>>>(
        scores, V, output, scale, seqKV, D
    );
    return cudaGetLastError();
}

} // extern "C"

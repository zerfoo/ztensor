// fused_repeat_interleave.cu -- CUDA kernel for fused GQA key/value head expansion.
// Expands tensor from [B, numKV, S, D] to [B, numQ, S, D] by repeating
// each KV head `rep` times along the head dimension, where numQ = numKV * rep.
// Replaces the Reshape -> Repeat -> Reshape chain with a single kernel launch.

#include <cuda_runtime.h>

__global__ void kernel_repeat_interleave(
    const float* __restrict__ input,   // [B, numKV, S, D]
    float* __restrict__ output,        // [B, numQ, S, D]
    int B, int numKV, int S, int D, int rep
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numQ = numKV * rep;
    int total = B * numQ * S * D;
    if (idx >= total) return;

    // Decompose flat index into [b, qHead, s, d]
    int d = idx % D;
    int s = (idx / D) % S;
    int qHead = (idx / (D * S)) % numQ;
    int b = idx / (D * S * numQ);

    // Map output head to source KV head
    int kvHead = qHead / rep;

    // Read from input
    int srcIdx = ((b * numKV + kvHead) * S + s) * D + d;
    output[idx] = __ldg(&input[srcIdx]);
}

// ---------- Launcher function (extern "C" for CGO / purego) ----------

extern "C" {

cudaError_t launch_repeat_interleave_f32(
    const float* input, float* output,
    int B, int numKV, int S, int D, int rep,
    cudaStream_t stream
) {
    int numQ = numKV * rep;
    int total = B * numQ * S * D;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_repeat_interleave<<<grid, block, 0, stream>>>(
        input, output, B, numKV, S, D, rep
    );
    return cudaGetLastError();
}

} // extern "C"

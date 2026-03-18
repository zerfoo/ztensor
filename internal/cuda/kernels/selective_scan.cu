// selective_scan.cu -- CUDA kernel for Mamba/SSM selective scan (S6).
// Computes the linear recurrence: h[t] = A*h[t-1] + B[:,t]*x[:,t]
// and output: y[t] = C[:,t]*h[t] + D*x[:,t]
//
// Each thread handles one (batch, d_model) pair and scans sequentially
// over the sequence dimension. d_model * batch is typically large enough
// to saturate GPU occupancy while seq_len is moderate (<=4096 in Mamba).

#include <cuda_runtime.h>

// Sequential scan kernel: each thread processes one (batch, d_model) pair,
// maintaining d_state hidden states across the sequence.
//
// Memory layout (row-major):
//   x:  [batch, d_model, seq_len]
//   A:  [d_model, d_state]
//   B:  [batch, d_state, seq_len]
//   C:  [batch, d_state, seq_len]
//   D:  [d_model]  (may be NULL for no skip connection)
//   y:  [batch, d_model, seq_len]
//
// d_state must be <= 64 (fits in registers for Mamba's typical d_state=16).
__global__ void kernel_selective_scan_forward(
    const float* __restrict__ x,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D,
    float* __restrict__ y,
    int batch,
    int d_model,
    int d_state,
    int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_model;
    if (idx >= total) return;

    int b = idx / d_model;
    int d = idx % d_model;

    const float* x_bd = x + (b * d_model + d) * seq_len;
    float* y_bd = y + (b * d_model + d) * seq_len;

    float d_skip = (D != NULL) ? D[d] : 0.0f;

    // Hidden state: h[s] for each d_state dimension.
    // Mamba uses d_state=16; we support up to 64.
    float h[64];
    for (int s = 0; s < d_state && s < 64; s++) {
        h[s] = 0.0f;
    }

    for (int t = 0; t < seq_len; t++) {
        float x_t = x_bd[t];
        float y_t = d_skip * x_t;

        for (int s = 0; s < d_state && s < 64; s++) {
            float a_ds = A[d * d_state + s];
            float b_t = B[(b * d_state + s) * seq_len + t];
            float c_t = C[(b * d_state + s) * seq_len + t];

            h[s] = a_ds * h[s] + b_t * x_t;
            y_t += c_t * h[s];
        }

        y_bd[t] = y_t;
    }
}

// ---------- Launcher function (extern "C" for dlsym) ----------

extern "C" {

cudaError_t launch_selective_scan_forward(
    const float* x,
    const float* A,
    const float* B,
    const float* C,
    const float* D,
    float* y,
    int batch,
    int d_model,
    int d_state,
    int seq_len,
    cudaStream_t stream)
{
    int total = batch * d_model;
    const int BLOCK = 256;
    int numBlocks = (total + BLOCK - 1) / BLOCK;

    kernel_selective_scan_forward<<<numBlocks, BLOCK, 0, stream>>>(
        x, A, B, C, D, y, batch, d_model, d_state, seq_len);
    return cudaGetLastError();
}

} // extern "C"

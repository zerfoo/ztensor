// gather_q8.cu -- CUDA kernel for Q8_0 embedding table gather (lookup).
// Dequantizes only the requested rows on GPU, avoiding full table decode.
//
// Q8Storage block format (36 bytes per 32 values):
//   [0:4]   float32 scale
//   [4:36]  32 x int8 quantized values
//
// Note: this matches Zerfoo's Q8Storage format (float32 scale), NOT
// the GGUF Q8_0 format (fp16 scale, 34 bytes). Q8Storage converts
// the fp16 scale to float32 during GGUF loading.
//
// Dequantization: value[i] = scale * quant[i]
//
// The table is stored as contiguous Q8_0 blocks: V rows of (D/32) blocks each.
// Each thread block handles one index (one row lookup).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Q8Storage uses 36-byte blocks: float32 scale (4 bytes) + 32 x int8 (32 bytes).
// This differs from GGUF Q8_0 (34 bytes with fp16 scale) because Q8Storage
// converts the fp16 scale to float32 during loading.
#define Q8_BLOCK_SIZE  32
#define Q8_BLOCK_BYTES 36

// kernel_gather_q8: output[i, :] = dequant(q8_table[indices[i], :])
// q8_table: raw Q8_0 bytes [V * blocks_per_row * 34]
// indices: [N] int32 token IDs
// output: [N, D] float32
__global__ void kernel_gather_q8(
    const uint8_t* __restrict__ q8_table,
    const int*     __restrict__ indices,
    float*         __restrict__ output,
    int N, int D, int V)
{
    int row = blockIdx.x;
    if (row >= N) return;

    int idx = indices[row];
    if (idx < 0) idx = 0;
    if (idx >= V) idx = V - 1;

    int blocks_per_row = D / Q8_BLOCK_SIZE;
    const uint8_t* row_data = q8_table + (size_t)idx * blocks_per_row * Q8_BLOCK_BYTES;
    float* dst = output + (size_t)row * D;

    // Each thread handles a strided subset of Q8 blocks.
    for (int bi = threadIdx.x; bi < blocks_per_row; bi += blockDim.x) {
        const uint8_t* blk = row_data + bi * Q8_BLOCK_BYTES;

        // Read float32 scale using byte-wise load (ARM64 alignment safety).
        uint32_t s_bits = (uint32_t)__ldg(&blk[0]) | ((uint32_t)__ldg(&blk[1]) << 8)
                        | ((uint32_t)__ldg(&blk[2]) << 16) | ((uint32_t)__ldg(&blk[3]) << 24);
        float scale = *reinterpret_cast<const float*>(&s_bits);

        const int8_t* quants = (const int8_t*)(blk + 4);
        int base = bi * Q8_BLOCK_SIZE;

        // Dequantize 32 values.
        #pragma unroll
        for (int j = 0; j < Q8_BLOCK_SIZE; j++) {
            dst[base + j] = scale * (float)__ldg(&quants[j]);
        }
    }
}

// ---------- Launcher ----------

extern "C" {

cudaError_t launch_gather_q8_f32(
    const uint8_t* q8_table,
    const int*     indices,
    float*         output,
    int N, int D, int V,
    cudaStream_t stream)
{
    // One block per index. Threads handle blocks within the row.
    int blocks_per_row = D / Q8_BLOCK_SIZE;
    int block_size = blocks_per_row;
    if (block_size > 256) block_size = 256;
    // Round up to power of 2.
    int b = 1;
    while (b < block_size) b <<= 1;
    if (b > 256) b = 256;

    kernel_gather_q8<<<N, b, 0, stream>>>(q8_table, indices, output, N, D, V);
    return cudaGetLastError();
}

} // extern "C"

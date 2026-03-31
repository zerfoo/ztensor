/* Fused repeat-interleave kernel interface for GQA head expansion.
 *
 * Expands [B, numKV, S, D] to [B, numQ, S, D] where numQ = numKV * rep,
 * replacing the Reshape -> Repeat -> Reshape chain with one kernel launch.
 */
#ifndef FUSED_REPEAT_INTERLEAVE_H
#define FUSED_REPEAT_INTERLEAVE_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* launch_repeat_interleave_f32 expands KV heads for GQA in one kernel launch.
 *
 * input:  device pointer to [B, numKV, S, D] float array.
 * output: device pointer to [B, numQ, S, D] float array (numQ = numKV * rep).
 * B:      batch size.
 * numKV:  number of key/value heads.
 * S:      sequence length.
 * D:      head dimension.
 * rep:    replication factor (numQ / numKV).
 * stream: CUDA stream (pass NULL for default stream).
 */
cudaError_t launch_repeat_interleave_f32(
    const float* input, float* output,
    int B, int numKV, int S, int D, int rep,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_REPEAT_INTERLEAVE_H */

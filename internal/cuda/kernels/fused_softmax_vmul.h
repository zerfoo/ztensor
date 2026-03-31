/* Fused softmax + V multiply kernel interface for decode attention.
 *
 * Computes output = softmax(scores * scale) @ V in a single kernel launch,
 * avoiding materialization of the attention weights tensor.
 * Decode-optimized (seqQ=1): each block handles one (batch, head) pair.
 */
#ifndef FUSED_SOFTMAX_VMUL_H
#define FUSED_SOFTMAX_VMUL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* launch_fused_softmax_vmul_f32 computes softmax(scores*scale) @ V.
 *
 * scores:     device pointer to [BH, seqKV] float array.
 * V:          device pointer to [BH, seqKV, D] float array.
 * output:     device pointer to [BH, D] float array.
 * scale_bits: pre-softmax scale as uint32 (reinterpret_cast<uint32>(float)).
 * BH:         batch * heads.
 * seqKV:      key/value sequence length.
 * D:          head dimension.
 * stream:     CUDA stream (pass NULL for default stream).
 */
cudaError_t launch_fused_softmax_vmul_f32(
    const float* scores, const float* V, float* output,
    unsigned int scale_bits,
    int BH, int seqKV, int D,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_SOFTMAX_VMUL_H */

/* Fused on-device AdamW mixed-precision update kernel interface.
 *
 * Performs the full AdamW step in place on device (ADR 070 end state, ADR 075
 * lever L1): param, m, grad are f32 device buffers; v is an f64 device sidecar.
 * No host<->device round-trip of any optimizer state. The second moment stays
 * in f64 to preserve f32 training stability (the GPU "CrossAsset cliff" fix).
 */
#ifndef FUSED_ADAMW_H
#define FUSED_ADAMW_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fused_adamw_f32 applies one AdamW step in one kernel launch, in place.
 *
 * param:         device pointer to f32[n] -- updated in place.
 * m:             device pointer to f32[n] first moment -- updated in place.
 * v:             device pointer to f64[n] second moment -- updated in place.
 * grad:          device pointer to f32[n] gradient -- read then zeroed in place.
 * beta1, beta2:  Adam decay rates.
 * oneMinusBeta1, oneMinusBeta2: precomputed (1-beta) terms.
 * eps:           epsilon (added to sqrt(v)).
 * alpha:         lr * bias-correction (numer/denom), precomputed on host.
 * lrWd:          lr * weightDecay, precomputed on host.
 * n:             number of elements.
 * stream:        CUDA stream (pass NULL for default stream).
 */
cudaError_t fused_adamw_f32(
    float* param, float* m, double* v, float* grad,
    double beta1, double beta2,
    double oneMinusBeta1, double oneMinusBeta2,
    double eps, double alpha, double lrWd,
    int n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_ADAMW_H */

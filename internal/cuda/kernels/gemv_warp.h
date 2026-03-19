#ifndef GEMV_WARP_H
#define GEMV_WARP_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t launch_gemv_warp_f32(
    float* y, const float* A, const float* x,
    int M, int N, cudaStream_t stream);

cudaError_t launch_gemv_warp_f16(
    void* y, const void* A, const void* x,
    int M, int N, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GEMV_WARP_H */

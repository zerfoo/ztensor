/* fused_encoder_fwd.h -- Fused PatchTST encoder layer forward pass.
 *
 * A single host-side orchestrator function that replaces ~78 discrete
 * Engine operations per encoder layer with ~14 internal sub-operations
 * (cuBLAS GEMMs + custom CUDA kernels) launched on the same stream.
 *
 * Buffer indices for the ptrs[] and cache[] arrays are defined as enums
 * so Go bindings can reference the same layout.
 */

#ifndef FUSED_ENCODER_FWD_H
#define FUSED_ENCODER_FWD_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Weight pointer indices (16 pointers). */
enum FusedEncoderWeight {
    FEW_QW = 0, FEW_QB,
    FEW_KW,     FEW_KB,
    FEW_VW,     FEW_VB,
    FEW_OW,     FEW_OB,
    FEW_FFN1W,  FEW_FFN1B,
    FEW_FFN2W,  FEW_FFN2B,
    FEW_NORM1W, FEW_NORM1B,
    FEW_NORM2W, FEW_NORM2B,
    FEW_COUNT   /* = 16 */
};

/* Forward cache buffer indices (pre-allocated by Go, written by kernel).
 * The backward pass reads these to compute gradients. */
enum FusedEncoderFwdBuf {
    FEB_NORMED1 = 0,    /* [totalRows, dModel]                    */
    FEB_LN1_INVSTD,     /* [totalRows]                            */
    FEB_Q,              /* [totalRows, dModel]                    */
    FEB_K,              /* [totalRows, dModel]                    */
    FEB_V,              /* [totalRows, dModel]                    */
    FEB_QH,             /* [bsC*nHeads, numPatches, headDim]      */
    FEB_KH,             /* [bsC*nHeads, numPatches, headDim]      */
    FEB_VH,             /* [bsC*nHeads, numPatches, headDim]      */
    FEB_ATTN_SCORES,    /* [bsC*nHeads, numPatches, numPatches]   */
    FEB_ATTN_OUT_H,     /* [bsC*nHeads, numPatches, headDim]      */
    FEB_ATTN_OUT,       /* [totalRows, dModel]                    */
    FEB_X_RES1,         /* [totalRows, dModel]                    */
    FEB_NORMED2,        /* [totalRows, dModel]                    */
    FEB_LN2_INVSTD,     /* [totalRows]                            */
    FEB_FFN1_PRE,       /* [totalRows, ffnDim]                    */
    FEB_FFN1_OUT,       /* [totalRows, ffnDim]                    */
    FEB_COUNT           /* = 16 */
};

/* fused_encoder_fwd_f32 -- execute one encoder layer forward pass.
 *
 * Parameters:
 *   cublas_handle  cuBLAS handle with stream already set
 *   weights        FEW_COUNT pointers to layer weight device memory
 *   bufs           FEB_COUNT pointers to pre-allocated cache buffers
 *   input          [totalRows, dModel] layer input (device ptr)
 *   output         [totalRows, dModel] layer output (device ptr, may alias input)
 *   totalRows      bsC * numPatches
 *   dModel         embedding dimension
 *   nHeads         number of attention heads
 *   headDim        dModel / nHeads
 *   ffnDim         feed-forward hidden dimension (typically dModel * 4)
 *   bsC            batch_size * channels
 *   numPatches     sequence length (number of patches)
 *   stream         CUDA stream
 *
 * Returns cudaSuccess on success.
 */
cudaError_t fused_encoder_fwd_f32(
    void*        cublas_handle,
    const void** weights,
    void**       bufs,
    const float* input,
    float*       output,
    int totalRows, int dModel, int nHeads, int headDim, int ffnDim,
    int bsC, int numPatches,
    cudaStream_t stream);

/* fused_encoder_fwd_scratch_bytes -- compute minimum buffer sizes.
 *
 * Returns the total bytes needed across all FEB_COUNT buffers.
 * Caller can use this to validate pre-allocation.
 */
long long fused_encoder_fwd_scratch_bytes(
    int totalRows, int dModel, int nHeads, int headDim, int ffnDim,
    int bsC, int numPatches);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_ENCODER_FWD_H */

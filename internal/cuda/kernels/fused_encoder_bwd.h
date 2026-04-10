/* fused_encoder_bwd.h -- Fused PatchTST encoder layer backward pass.
 *
 * Companion to fused_encoder_fwd.h. Computes gradients for all weights
 * and the input gradient in a single host-side orchestrator call.
 * Reads from the forward cache (FEB_* buffers) to avoid recomputation.
 */

#ifndef FUSED_ENCODER_BWD_H
#define FUSED_ENCODER_BWD_H

#include <cuda_runtime.h>
#include "fused_encoder_fwd.h"  /* FEW_*, FEB_* enums */

#ifdef __cplusplus
extern "C" {
#endif

/* Weight gradient output indices (16 gradient tensors, same order as FEW_*). */
enum FusedEncoderGrad {
    FEG_DQW = 0, FEG_DQB,
    FEG_DKW,     FEG_DKB,
    FEG_DVW,     FEG_DVB,
    FEG_DOW,     FEG_DOB,
    FEG_DFFN1W,  FEG_DFFN1B,
    FEG_DFFN2W,  FEG_DFFN2B,
    FEG_DNORM1W, FEG_DNORM1B,
    FEG_DNORM2W, FEG_DNORM2B,
    FEG_COUNT    /* = 16 */
};

/* Weight transpose indices (pre-computed by Go for backward efficiency). */
enum FusedEncoderWeightT {
    FEWT_QWT = 0,   /* qW^T [dModel, dModel] */
    FEWT_KWT,       /* kW^T [dModel, dModel] */
    FEWT_VWT,       /* vW^T [dModel, dModel] */
    FEWT_OWT,       /* oW^T [dModel, dModel] */
    FEWT_FFN1WT,    /* ffn1W^T [ffnDim, dModel] */
    FEWT_FFN2WT,    /* ffn2W^T [dModel, ffnDim] */
    FEWT_COUNT      /* = 6 */
};

/* Backward scratch buffer indices. */
enum FusedEncoderBwdBuf {
    FEBB_DFFN1_OUT = 0,  /* [totalRows, ffnDim]                   */
    FEBB_DFFN1_PRE,      /* [totalRows, ffnDim]                   */
    FEBB_DNORMED2,       /* [totalRows, dModel]                   */
    FEBB_DX_RES1,        /* [totalRows, dModel]                   */
    FEBB_DATTN_OUT,      /* [totalRows, dModel]                   */
    FEBB_DATTN_OUT_H,    /* [bnh, numPatches, headDim]            */
    FEBB_DQH,            /* [bnh, numPatches, headDim]            */
    FEBB_DKH,            /* [bnh, numPatches, headDim]            */
    FEBB_DVH,            /* [bnh, numPatches, headDim]            */
    FEBB_DSCORES,        /* [bnh, numPatches, numPatches]         */
    FEBB_DQ,             /* [totalRows, dModel]                   */
    FEBB_DK,             /* [totalRows, dModel]                   */
    FEBB_DV,             /* [totalRows, dModel]                   */
    FEBB_DNORMED1,       /* [totalRows, dModel]                   */
    FEBB_TEMP,           /* [max(totalRows*dModel, totalRows*ffnDim)] scratch */
    FEBB_COUNT           /* = 15 */
};

/* fused_encoder_bwd_f32 -- compute gradients for one encoder layer.
 *
 * Parameters:
 *   cublas_handle  cuBLAS handle with stream already set
 *   weights        FEW_COUNT pointers to layer weights (read-only)
 *   weight_t       FEWT_COUNT pointers to pre-transposed weights (read-only)
 *   fwd_bufs       FEB_COUNT pointers to forward cache (read-only)
 *   bwd_bufs       FEBB_COUNT pointers to backward scratch (read-write)
 *   grads          FEG_COUNT pointers to gradient accumulators (ACCUMULATED, not zeroed)
 *   dOutput        [totalRows, dModel] upstream gradient (read-only)
 *   dInput         [totalRows, dModel] input gradient output
 *   input          [totalRows, dModel] original layer input (for LN1 backward)
 *   totalRows..numPatches  dimension parameters
 *   stream         CUDA stream
 */
cudaError_t fused_encoder_bwd_f32(
    void*        cublas_handle,
    const void** weights,
    const void** weight_t,
    const void** fwd_bufs,
    void**       bwd_bufs,
    void**       grads,
    const float* dOutput,
    float*       dInput,
    const float* input,
    int totalRows, int dModel, int nHeads, int headDim, int ffnDim,
    int bsC, int numPatches,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_ENCODER_BWD_H */

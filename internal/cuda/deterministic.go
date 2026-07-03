package cuda

import "os"

// Deterministic-reductions debug mode (zerfoo
// docs/plan-gpu-training-hardening.md T4.1; scope documented in
// docs/design.md "ZTENSOR_DETERMINISTIC scope").
//
// When ZTENSOR_DETERMINISTIC=1:
//
//   - cuBLAS handles (internal/cublas.CreateHandle) are created with
//     CUBLAS_WORKSPACE_CONFIG forced to a fixed value (if the process has not
//     already set one) and math mode CUBLAS_PEDANTIC_MATH, which per NVIDIA's
//     cuBLAS documentation forgoes algorithm choices -- including TF32
//     tensor-core downcasting and any workspace-dependent split-K/atomics
//     reduction strategy -- that are not required to reproduce bit-identical
//     results for a fixed problem size and GPU. This is the same lever
//     PyTorch's torch.use_deterministic_algorithms(True) documents for
//     cuBLAS GEMMs.
//   - CPU fp32 reductions (numeric.Sum, compute.CPUEngine Sum/Softmax,
//     xblas RMSNorm sum-of-squares, including the arm64 NEON SIMD path) were
//     already made fixed-order and deterministic UNCONDITIONALLY by T135.2;
//     this flag does not change CPU behavior.
//   - GPU custom kernels (softmax, RMSNorm, GEMV, flash attention/decode,
//     AdamW) reduce via warp-shuffle/tree patterns inside a launch
//     configuration that is a pure function of input shape, with no
//     cross-block atomics; they are deterministic already and this flag does
//     not change them.
//   - fused_encoder_bwd.cu's dScale/dBias LayerNorm-style bias-gradient
//     accumulation uses atomicAdd across row blocks with NO deterministic
//     variant. compute.GPUEngine.FusedEncoderBackward refuses to run under
//     this flag instead of silently returning order-dependent gradients --
//     see docs/design.md for the honest exclusion.
//
// Off by default (debug/verification tool); expect a measurable GEMM
// slowdown when enabled, since CUBLAS_PEDANTIC_MATH forgoes TF32 tensor
// cores.
var deterministicEnabled = os.Getenv("ZTENSOR_DETERMINISTIC") == "1"

// DeterministicEnabled reports whether ZTENSOR_DETERMINISTIC=1 was set at
// process start.
func DeterministicEnabled() bool { return deterministicEnabled }

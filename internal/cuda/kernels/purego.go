package kernels

import (
	"fmt"
	"log/slog"
	"sync"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// KernelLib holds dlopen'd function pointers for custom CUDA kernels
// compiled into libkernels.so.
type KernelLib struct {
	handle uintptr

	// elementwise binary
	launchAdd, launchSub, launchMul, launchDiv, launchPow uintptr

	// elementwise scalar
	launchAddScalar, launchMulScalar, launchDivScalar uintptr
	launchSubScalar, launchPowScalar                  uintptr

	// elementwise unary
	launchExp, launchLog, launchSqrt, launchRsqrt, launchSin, launchCos, launchTanh uintptr
	launchTanhPrime                                                                 uintptr

	// elementwise special
	launchFill, launchSumAxis, launchSoftmax uintptr

	// broadcast
	launchAddBroadcast, launchSubBroadcast uintptr
	launchMulBroadcast, launchDivBroadcast uintptr

	// broadcast 4D
	launchAddBroadcast4D, launchSubBroadcast4D uintptr
	launchMulBroadcast4D, launchDivBroadcast4D uintptr

	// rmsnorm
	launchRMSNorm uintptr

	// gather
	launchGather      uintptr
	launchGatherI32   uintptr
	launchGatherQ8F32 uintptr

	// transpose
	launchTranspose2D, launchTransposeND uintptr

	// repeat
	launchRepeat uintptr

	// gemm_q4
	launchGemmQ4F32 uintptr

	// gemv_q4k (fused dequant+GEMV for Q4_K_M)
	launchGemvQ4KF32      uintptr
	launchGemvQ4KDp4aF32  uintptr
	launchGemvQ4KSm121F32 uintptr // sm_121 optimized path (Blackwell GB10)
	checkGemvQ4KSm121     uintptr // capability probe

	// gemv_q5k (fused dequant+GEMV for Q5_K_M)
	launchGemvQ5KF32 uintptr

	// gemv_q6k (fused dequant+GEMV for Q6_K)
	launchGemvQ6KF32 uintptr

	// gemv_q5_0 (fused dequant+GEMV for Q5_0)
	launchGemvQ5_0F32 uintptr

	// dequant_q4k (Q4_K to F32 for non-GEMV cuBLAS path)
	launchDequantQ4KF32 uintptr

	// gemm_q8
	launchGemmQ8F32 uintptr

	// argmax
	launchArgmax uintptr

	// fused_rope
	launchFusedRoPEF32 uintptr

	// fused_swiglu
	launchFusedSwiGLUF32 uintptr

	// fused_repeat_interleave (GQA KV head expansion)
	launchRepeatInterleaveF32 uintptr

	// fused_add_rmsnorm
	launchFusedAddRMSNormF32 uintptr

	// fused_norm_add
	launchFusedNormAddF32 uintptr

	// fused_qk_norm_rope
	launchFusedQKNormRoPEF32 uintptr

	// scaled_softmax
	launchScaledSoftmaxF32 uintptr

	// flash_attention
	launchFlashAttentionF32       uintptr
	launchFlashAttentionDecodeF32 uintptr

	// flash_attention2
	launchFlashAttention2F32       uintptr
	launchFlashAttention2DecodeF32 uintptr

	// FP16 elementwise
	launchAddFP16, launchSubFP16, launchMulFP16, launchDivFP16 uintptr

	// FP16 rmsnorm
	launchRMSNormFP16 uintptr

	// FP16 scaled_softmax
	launchScaledSoftmaxFP16 uintptr

	// FP16 conversion
	launchF32ToFP16 uintptr
	launchFP16ToF32 uintptr

	// FP8 ops
	launchDequantFP8E4M3ToFP16 uintptr
	launchFP8Add               uintptr
	launchFP8Mul               uintptr
	launchFP8RMSNorm           uintptr

	// FP8 GEMM (cublasLt, sm_89+)
	launchFP8Gemm uintptr

	// counter
	launchIncrementCounter uintptr
	launchResetCounter     uintptr

	// offset_memcpy
	launchOffsetMemcpy     uintptr
	launchOffsetMemcpyFP16 uintptr

	// rope_select
	launchRoPESelect uintptr

	// sgemv_m1
	launchSgemvM1 uintptr

	// selective_scan
	launchSelectiveScanForward uintptr

	// flash_decode (split-KV)
	launchFlashDecodeSplitKVF32 uintptr

	// paged_attention
	launchPagedAttentionF32 uintptr

	// ragged_attention
	launchRaggedAttentionF32 uintptr

	// fp4_gemv (NVFP4 E2M1 fused dequant+GEMV, sm_100+)
	launchFP4GemvF16 uintptr

	// gemv_warp (warp-specialized GEMV for decode phase)
	launchGemvWarpF32 uintptr
	launchGemvWarpF16 uintptr

	// fused_softmax_vmul (fused softmax + V multiply for decode attention)
	launchFusedSoftmaxVMulF32 uintptr

	// ternary_gemv (ternary {-1,0,+1} packed 2-bit GEMV)
	launchTernaryGemvF32 uintptr

	// nsa_attention (fused three-path Native Sparse Attention)
	launchNSAAttentionF32 uintptr

	// kv_dequant (Q4/Q3 KV cache dequantization)
	launchKVDequantQ4F32 uintptr
	launchKVDequantQ3F32 uintptr

	// iq_dequant (importance-weighted quantization dequantization)
	launchIQDequantNLF32  uintptr
	launchIQDequant3SF32  uintptr
	launchIQDequant2XXSF32 uintptr
}

var (
	kernelLib     *KernelLib
	kernelLibOnce sync.Once
	errKernelLib  error
)

// openKernelLib loads libkernels.so and resolves all kernel function pointers.
func openKernelLib() (*KernelLib, error) {
	kernelLibOnce.Do(func() {
		if !cuda.Available() {
			errKernelLib = fmt.Errorf("kernels: cuda not available")
			return
		}
		lib, err := cuda.DlopenKernels()
		if err != nil {
			errKernelLib = err
			return
		}
		k := &KernelLib{handle: lib}
		syms := []struct {
			name string
			dest *uintptr
		}{
			// elementwise binary
			{"launch_add", &k.launchAdd},
			{"launch_sub", &k.launchSub},
			{"launch_mul", &k.launchMul},
			{"launch_div", &k.launchDiv},
			{"launch_pow", &k.launchPow},
			// elementwise scalar
			{"launch_add_scalar", &k.launchAddScalar},
			{"launch_mul_scalar", &k.launchMulScalar},
			{"launch_div_scalar", &k.launchDivScalar},
			{"launch_sub_scalar", &k.launchSubScalar},
			{"launch_pow_scalar", &k.launchPowScalar},
			// elementwise unary
			{"launch_exp", &k.launchExp},
			{"launch_log", &k.launchLog},
			{"launch_sqrt", &k.launchSqrt},
			{"launch_rsqrt", &k.launchRsqrt},
			{"launch_sin", &k.launchSin},
			{"launch_cos", &k.launchCos},
			{"launch_tanh", &k.launchTanh},
			{"launch_tanh_prime", &k.launchTanhPrime},
			// elementwise special
			{"launch_fill", &k.launchFill},
			{"launch_sum_axis", &k.launchSumAxis},
			{"launch_softmax", &k.launchSoftmax},
			// broadcast
			{"launch_add_broadcast", &k.launchAddBroadcast},
			{"launch_sub_broadcast", &k.launchSubBroadcast},
			{"launch_mul_broadcast", &k.launchMulBroadcast},
			{"launch_div_broadcast", &k.launchDivBroadcast},
			// broadcast 4D
			{"launch_add_broadcast4d", &k.launchAddBroadcast4D},
			{"launch_sub_broadcast4d", &k.launchSubBroadcast4D},
			{"launch_mul_broadcast4d", &k.launchMulBroadcast4D},
			{"launch_div_broadcast4d", &k.launchDivBroadcast4D},
			// rmsnorm
			{"launch_rmsnorm", &k.launchRMSNorm},
			// gather
			{"launch_gather", &k.launchGather},
			{"launch_gather_i32", &k.launchGatherI32},
			{"launch_gather_q8_f32", &k.launchGatherQ8F32},
			// transpose
			{"launch_transpose_2d", &k.launchTranspose2D},
			{"launch_transpose_nd", &k.launchTransposeND},
			// repeat
			{"launch_repeat", &k.launchRepeat},
			// gemm_q4
			{"gemm_q4_f32", &k.launchGemmQ4F32},
			// gemv_q4k (fused dequant+GEMV for Q4_K_M)
			{"gemv_q4k_f32", &k.launchGemvQ4KF32},
			{"gemv_q4k_dp4a_f32", &k.launchGemvQ4KDp4aF32},
			// gemv_q4k sm_121 optimized path (Blackwell GB10 / DGX Spark)
			{"gemv_q4k_sm121_f32", &k.launchGemvQ4KSm121F32},
			{"gemv_q4k_check_sm121", &k.checkGemvQ4KSm121},
			// gemv_q5k (fused dequant+GEMV for Q5_K_M)
			{"gemv_q5k_f32", &k.launchGemvQ5KF32},
			// gemv_q6k (fused dequant+GEMV for Q6_K)
			{"gemv_q6k_f32", &k.launchGemvQ6KF32},
			// gemv_q5_0 (fused dequant+GEMV for Q5_0)
			{"gemv_q5_0_f32", &k.launchGemvQ5_0F32},
			// dequant_q4k (Q4_K to F32 for non-GEMV cuBLAS path)
			{"dequant_q4k_f32", &k.launchDequantQ4KF32},
			// gemm_q8
			{"gemm_q8_f32", &k.launchGemmQ8F32},
			// argmax
			{"launch_argmax", &k.launchArgmax},
			// fused_rope
			{"fused_rope_f32", &k.launchFusedRoPEF32},
			// fused_swiglu
			{"fused_swiglu_f32", &k.launchFusedSwiGLUF32},
			// fused_repeat_interleave (GQA KV head expansion)
			{"launch_repeat_interleave_f32", &k.launchRepeatInterleaveF32},
		// fused_add_rmsnorm
		{"fused_add_rmsnorm_f32", &k.launchFusedAddRMSNormF32},
		// fused_norm_add
		{"fused_norm_add_f32", &k.launchFusedNormAddF32},
		// fused_qk_norm_rope
		{"fused_qk_norm_rope_f32", &k.launchFusedQKNormRoPEF32},
		// scaled_softmax
		{"scaled_softmax_f32", &k.launchScaledSoftmaxF32},
		// flash_attention
		{"flash_attention_forward_f32", &k.launchFlashAttentionF32},
		{"flash_attention_decode_f32", &k.launchFlashAttentionDecodeF32},
		// flash_attention2
		{"flash_attention2_forward_f32", &k.launchFlashAttention2F32},
		{"flash_attention2_decode_f32", &k.launchFlashAttention2DecodeF32},
		// FP16 elementwise
		{"launch_add_fp16", &k.launchAddFP16},
		{"launch_sub_fp16", &k.launchSubFP16},
		{"launch_mul_fp16", &k.launchMulFP16},
		{"launch_div_fp16", &k.launchDivFP16},
		// FP16 rmsnorm
		{"launch_rmsnorm_fp16", &k.launchRMSNormFP16},
		// FP16 scaled_softmax
		{"launch_scaled_softmax_fp16", &k.launchScaledSoftmaxFP16},
		// FP16 conversion
		{"launch_f32_to_fp16", &k.launchF32ToFP16},
		{"launch_fp16_to_f32", &k.launchFP16ToF32},
		// FP8 ops
		{"launch_dequant_fp8e4m3_to_fp16", &k.launchDequantFP8E4M3ToFP16},
		{"launch_fp8_add", &k.launchFP8Add},
		{"launch_fp8_mul", &k.launchFP8Mul},
		{"launch_fp8_rmsnorm", &k.launchFP8RMSNorm},
		// FP8 GEMM (cublasLt, sm_89+)
		{"launch_fp8_gemm", &k.launchFP8Gemm},
		// counter
		{"launch_increment_counter", &k.launchIncrementCounter},
		{"launch_reset_counter", &k.launchResetCounter},
		// offset_memcpy
		{"launch_offset_memcpy", &k.launchOffsetMemcpy},
		{"launch_offset_memcpy_fp16", &k.launchOffsetMemcpyFP16},
		// rope_select
		{"launch_rope_select", &k.launchRoPESelect},
		// sgemv_m1
		{"launch_sgemv_m1", &k.launchSgemvM1},
		// selective_scan
		{"launch_selective_scan_forward", &k.launchSelectiveScanForward},
		// flash_decode (split-KV)
		{"flash_decode_splitkv_f32", &k.launchFlashDecodeSplitKVF32},
		// paged_attention
		{"paged_attention_forward_f32", &k.launchPagedAttentionF32},
		// ragged_attention
		{"ragged_attention_forward_f32", &k.launchRaggedAttentionF32},
		// fp4_gemv (NVFP4 E2M1 fused dequant+GEMV, sm_100+)
		{"fp4_gemv_f16", &k.launchFP4GemvF16},
		// gemv_warp (warp-specialized GEMV for decode phase)
		{"launch_gemv_warp_f32", &k.launchGemvWarpF32},
		{"launch_gemv_warp_f16", &k.launchGemvWarpF16},
		// fused_softmax_vmul (fused softmax + V multiply for decode attention)
		{"launch_fused_softmax_vmul_f32", &k.launchFusedSoftmaxVMulF32},
		// ternary_gemv (ternary {-1,0,+1} packed 2-bit GEMV)
		{"ternary_gemv_f32", &k.launchTernaryGemvF32},
		// nsa_attention (fused three-path Native Sparse Attention)
		{"nsa_attention_f32", &k.launchNSAAttentionF32},
		// kv_dequant (Q4/Q3 KV cache dequantization)
		{"kv_dequant_q4_f32", &k.launchKVDequantQ4F32},
		{"kv_dequant_q3_f32", &k.launchKVDequantQ3F32},
		// iq_dequant (importance-weighted quantization)
		{"iq_dequant_nl_f32", &k.launchIQDequantNLF32},
		{"iq_dequant_3s_f32", &k.launchIQDequant3SF32},
		{"iq_dequant_2xxs_f32", &k.launchIQDequant2XXSF32},
		}
		// Optional symbols: missing is non-fatal (kernel not compiled yet).
		optionalSyms := map[string]bool{
			"gemv_q4k_dp4a_f32":              true,
			"gemv_q4k_sm121_f32":             true, // sm_121 optimized; requires Blackwell
			"gemv_q4k_check_sm121":           true, // capability probe; may be absent on older builds
			"gemv_q5k_f32":                    true,
			"gemv_q6k_f32":                    true,
			"gemv_q5_0_f32":                   true,
			"flash_attention_decode_f32":      true,
			"flash_attention2_forward_f32":   true,
			"flash_attention2_decode_f32":    true,
			"launch_f32_to_fp16":              true,
			"launch_fp16_to_f32":              true,
			"launch_dequant_fp8e4m3_to_fp16":  true,
			"launch_fp8_add":                  true,
			"launch_fp8_mul":                  true,
			"launch_fp8_rmsnorm":              true,
			"launch_fp8_gemm":                true,
			"launch_increment_counter":        true,
			"launch_reset_counter":            true,
			"launch_offset_memcpy":            true,
			"launch_offset_memcpy_fp16":       true,
			"launch_rope_select":              true,
			"launch_sgemv_m1":                 true,
			"launch_selective_scan_forward":   true,
			"flash_decode_splitkv_f32":        true,
			"paged_attention_forward_f32":     true,
			"ragged_attention_forward_f32":    true,
			"fp4_gemv_f16":                    true,
			"launch_gemv_warp_f32":             true,
			"launch_gemv_warp_f16":             true,
			"launch_repeat_interleave_f32":    true,
			"launch_gather_q8_f32":            true,
			"launch_fused_softmax_vmul_f32":   true,
			"ternary_gemv_f32":                true,
			"nsa_attention_f32":               true,
			"kv_dequant_q4_f32":               true,
			"kv_dequant_q3_f32":               true,
			"iq_dequant_nl_f32":               true,
			"iq_dequant_3s_f32":               true,
			"iq_dequant_2xxs_f32":             true,
		}
		for _, s := range syms {
			ptr, dlErr := cuda.Dlsym(lib, s.name)
			if dlErr != nil {
				if optionalSyms[s.name] {
					continue // leave function pointer as 0; callers check before use
				}
				errKernelLib = fmt.Errorf("kernels: dlsym %s: %w", s.name, dlErr)
				return
			}
			*s.dest = ptr
		}
		kernelLib = k
	})
	return kernelLib, errKernelLib
}

var klibWarnOnce sync.Once

func klib() *KernelLib {
	k, err := openKernelLib()
	if err != nil {
		klibWarnOnce.Do(func() {
			slog.Warn("cuda kernels not available -- GPU operations will fail",
				"error", err,
				"hint", "build with: cd internal/cuda/kernels && make CUDA_ARCH=<your_arch> shared")
		})
	}
	return k
}

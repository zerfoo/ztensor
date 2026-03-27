//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// KVDequantQ4 performs fused Q4 dequantization and gather for KV cache attention.
// packed: GPU pointer to Q4-packed KV data
// scales: GPU pointer to per-group scale factors
// output: GPU pointer to dequantized float32 output
// indices: GPU pointer to gather indices (which KV positions to read)
// numIndices: number of positions to gather
// dim: KV head dimension
// groupSize: quantization group size (128)
func KVDequantQ4(packed, scales, output, indices unsafe.Pointer, numIndices, dim, groupSize int, stream unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("kv_dequant_q4_f32 kernel: kernels not available")
	}
	if k.launchKVDequantQ4F32 == 0 {
		return fmt.Errorf("kv_dequant_q4_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchKVDequantQ4F32,
		uintptr(packed), uintptr(scales), uintptr(output), uintptr(indices),
		uintptr(numIndices), uintptr(dim), uintptr(groupSize),
		uintptr(stream))
	return checkKernel(ret, "kv_dequant_q4_f32")
}

// KVDequantQ3 performs fused Q3 codebook dequantization and gather.
// packed: GPU pointer to Q3-packed KV data
// centroids: GPU pointer to per-group codebook centroids (8 per group)
// output: GPU pointer to dequantized float32 output
// indices: GPU pointer to gather indices
// numIndices: number of positions to gather
// dim: KV head dimension
// groupSize: quantization group size (128)
func KVDequantQ3(packed, centroids, output, indices unsafe.Pointer, numIndices, dim, groupSize int, stream unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("kv_dequant_q3_f32 kernel: kernels not available")
	}
	if k.launchKVDequantQ3F32 == 0 {
		return fmt.Errorf("kv_dequant_q3_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchKVDequantQ3F32,
		uintptr(packed), uintptr(centroids), uintptr(output), uintptr(indices),
		uintptr(numIndices), uintptr(dim), uintptr(groupSize),
		uintptr(stream))
	return checkKernel(ret, "kv_dequant_q3_f32")
}

// IsKVDequantQ4Supported returns true if the Q4 KV dequant kernel is loaded.
func IsKVDequantQ4Supported() bool {
	k := klib()
	return k != nil && k.launchKVDequantQ4F32 != 0
}

// IsKVDequantQ3Supported returns true if the Q3 KV dequant kernel is loaded.
func IsKVDequantQ3Supported() bool {
	k := klib()
	return k != nil && k.launchKVDequantQ3F32 != 0
}

//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// NSAAttentionForward performs fused three-path Native Sparse Attention on GPU.
//
// Q:          [batch*numHeads, seqQ, headDim] query tensor
// K:          [batch, seqKV, numKVHeads*headDim] key cache
// V:          [batch, seqKV, numKVHeads*headDim] value cache
// O:          [batch*numHeads, seqQ, headDim] output
// gateCoarse: [numHeads] sigmoid gate for coarse path
// gateFine:   [numHeads] sigmoid gate for fine path
// gateWindow: [numHeads] sigmoid gate for window path
//
// blockSize:  coarse block size (tokens per block)
// topBlocks:  number of top blocks to select (coarse path)
// topTokens:  number of top tokens to select (fine path)
// windowSize: sliding window size
func NSAAttentionForward(
	Q, K, V, O unsafe.Pointer,
	gateCoarse, gateFine, gateWindow unsafe.Pointer,
	numBH, seqQ, seqKV, headDim int,
	numQueryHeads, numKVHeads int,
	blockSize, topBlocks, topTokens, windowSize int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("nsa_attention_f32 kernel: kernels not available")
	}
	if k.launchNSAAttentionF32 == 0 {
		return fmt.Errorf("nsa_attention_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchNSAAttentionF32,
		uintptr(Q), uintptr(K), uintptr(V), uintptr(O),
		uintptr(gateCoarse), uintptr(gateFine), uintptr(gateWindow),
		uintptr(numBH), uintptr(seqQ), uintptr(seqKV), uintptr(headDim),
		uintptr(numQueryHeads), uintptr(numKVHeads),
		uintptr(blockSize), uintptr(topBlocks), uintptr(topTokens), uintptr(windowSize),
		uintptr(stream))
	return checkKernel(ret, "nsa_attention_f32")
}

// IsNSAAttentionSupported returns true if the fused NSA kernel is loaded.
func IsNSAAttentionSupported() bool {
	k := klib()
	return k != nil && k.launchNSAAttentionF32 != 0
}

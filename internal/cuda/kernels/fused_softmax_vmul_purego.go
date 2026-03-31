//go:build !cuda

package kernels

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// FusedSoftmaxVMulF32 computes softmax(scores * scale) @ V in a single kernel.
// scores: [BH, seqKV], V: [BH, seqKV, D], output: [BH, D].
func FusedSoftmaxVMulF32(
	scores, V, output unsafe.Pointer,
	scale float32,
	BH, seqKV, D int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_softmax_vmul_f32 kernel: kernels not available")
	}
	scaleBits := math.Float32bits(scale)
	ret := cuda.Ccall(k.launchFusedSoftmaxVMulF32,
		uintptr(scores), uintptr(V), uintptr(output),
		uintptr(scaleBits),
		uintptr(BH), uintptr(seqKV), uintptr(D),
		uintptr(stream))
	return checkKernel(ret, "fused_softmax_vmul_f32")
}

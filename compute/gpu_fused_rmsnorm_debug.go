package compute

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/gpuapi"
)

// fusedRMSNormDebug is gated on ZERFOO_GQA_DEBUG=1 (same env var used by the
// zerfoo-side GQA / PLE instrumentation) so we can bisect the corrupting
// sub-step inside FusedRMSNormGPU for E98.T98.2.2 without spamming prod logs.
var fusedRMSNormDebug = os.Getenv("ZERFOO_GQA_DEBUG") == "1"

// fusedRMSNormProbe force-syncs the stream and issues a 1-byte D2H cudaMemcpy
// from ptr. If CUDA is in a sticky error state, either call surfaces the error
// immediately, which lets us pin the offending sub-step inside
// FusedRMSNormGPU. No-op when the debug env gate is off.
func fusedRMSNormProbe(tag string, runtime gpuapi.Runtime, stream gpuapi.Stream, ptr unsafe.Pointer, byteLen int) {
	if !fusedRMSNormDebug {
		return
	}
	var syncErr error
	if stream != nil {
		syncErr = stream.Synchronize()
	}
	var memcpyErr error
	if ptr != nil && byteLen > 0 && runtime != nil {
		var probe [1]byte
		memcpyErr = runtime.Memcpy(unsafe.Pointer(&probe[0]), ptr, 1, gpuapi.MemcpyDeviceToHost)
	}
	fmt.Fprintf(os.Stderr, "[RMS_DBG] %s gpuPtr=%p bytes=%d sync=%v memcpy=%v\n",
		tag, ptr, byteLen, syncErr, memcpyErr)
}

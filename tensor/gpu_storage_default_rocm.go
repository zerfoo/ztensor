//go:build rocm && !cuda

package tensor

import (
	"sync"

	"github.com/zerfoo/ztensor/internal/gpuapi"
)

// defaultRuntime is lazily initialized on first use. It provides backward
// compatibility for callers that do not supply a runtime explicitly.
var (
	defaultRuntime     gpuapi.Runtime
	defaultRuntimeOnce sync.Once
)

func getDefaultRuntime() gpuapi.Runtime {
	defaultRuntimeOnce.Do(func() {
		defaultRuntime = gpuapi.NewROCmRuntime()
	})
	return defaultRuntime
}

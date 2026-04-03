package graph

import "github.com/zerfoo/ztensor/internal/pjrt"

// NewPJRTClient loads a PJRT plugin shared library and creates a client.
// pluginPath is the bare library filename (e.g. "pjrt_c_api_cpu_plugin.so")
// or an absolute path. The returned client must be closed when no longer
// needed (typically after closing all PJRTPlans that reference it).
func NewPJRTClient(pluginPath string) (*pjrt.Client, error) {
	lib, err := pjrt.Load(pluginPath)
	if err != nil {
		return nil, err
	}
	return pjrt.NewClient(lib)
}

package cublas

import (
	"os"
	"sync"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// TestSetMathMode_NilHandle exercises the argument-validation path of
// SetMathMode, which does not require a live cuBLAS context, so it runs even
// on hosts without CUDA (e.g. darwin dev machines).
func TestSetMathMode_NilHandle(t *testing.T) {
	if err := SetMathMode(nil, CublasPedanticMath); err == nil {
		t.Fatal("SetMathMode(nil, ...): expected an error, got nil")
	}
}

// TestEnsureDeterministicWorkspaceConfig_SetsWhenUnset proves the
// ZTENSOR_DETERMINISTIC=1 workspace-config lever (T4.1): CUBLAS_WORKSPACE_CONFIG
// is set to a fixed value when the process has not already set one. Does not
// require CUDA -- this only touches the env var, not the cuBLAS library.
func TestEnsureDeterministicWorkspaceConfig_SetsWhenUnset(t *testing.T) {
	origWorkspace, hadWorkspace := os.LookupEnv("CUBLAS_WORKSPACE_CONFIG")
	_ = os.Unsetenv("CUBLAS_WORKSPACE_CONFIG")
	t.Cleanup(func() {
		if hadWorkspace {
			_ = os.Setenv("CUBLAS_WORKSPACE_CONFIG", origWorkspace)
		} else {
			_ = os.Unsetenv("CUBLAS_WORKSPACE_CONFIG")
		}
	})

	// workspaceConfigOnce is process-lifetime sync.Once; reset it so this
	// test's Do body actually runs regardless of test execution order.
	workspaceConfigOnce = sync.Once{}

	ensureDeterministicWorkspaceConfig()

	if got := os.Getenv("CUBLAS_WORKSPACE_CONFIG"); got == "" {
		t.Fatal("ensureDeterministicWorkspaceConfig: CUBLAS_WORKSPACE_CONFIG still unset")
	}
}

// TestEnsureDeterministicWorkspaceConfig_LeavesExistingValue confirms an
// operator-provided CUBLAS_WORKSPACE_CONFIG is never overwritten -- the
// process-level env var, set before the process starts, is the authoritative
// source per the documented cuBLAS/PyTorch determinism contract.
func TestEnsureDeterministicWorkspaceConfig_LeavesExistingValue(t *testing.T) {
	origWorkspace, hadWorkspace := os.LookupEnv("CUBLAS_WORKSPACE_CONFIG")
	const want = ":16:8"
	_ = os.Setenv("CUBLAS_WORKSPACE_CONFIG", want)
	t.Cleanup(func() {
		if hadWorkspace {
			_ = os.Setenv("CUBLAS_WORKSPACE_CONFIG", origWorkspace)
		} else {
			_ = os.Unsetenv("CUBLAS_WORKSPACE_CONFIG")
		}
	})

	workspaceConfigOnce = sync.Once{}
	ensureDeterministicWorkspaceConfig()

	if got := os.Getenv("CUBLAS_WORKSPACE_CONFIG"); got != want {
		t.Fatalf("ensureDeterministicWorkspaceConfig overwrote an existing value: got %q, want %q", got, want)
	}
}

// TestCreateHandle_DeterministicMode is the GPU proof that CreateHandle does
// not fail under ZTENSOR_DETERMINISTIC=1 (best-effort setup: a warning, never
// a hard error, when cublasSetMathMode or the workspace env var cannot be
// honored). Requires cuBLAS; skips on hosts without a GPU.
func TestCreateHandle_DeterministicMode(t *testing.T) {
	if !Available() {
		t.Skip("cuBLAS not available")
	}
	restore := cuda.SetDeterministicEnabledForTesting(true)
	defer restore()

	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle under ZTENSOR_DETERMINISTIC=1: %v", err)
	}
	defer func() { _ = h.Destroy() }()

	if err := SetMathMode(h, CublasPedanticMath); err != nil {
		t.Fatalf("SetMathMode(CUBLAS_PEDANTIC_MATH) on a live handle: %v", err)
	}
}

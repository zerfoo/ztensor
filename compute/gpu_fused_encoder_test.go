package compute

import (
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// TestFusedEncoderBackward_RefusesUnderDeterministicMode proves the
// ZTENSOR_DETERMINISTIC=1 honesty guard (T4.1): fused_encoder_bwd.cu's
// dScale/dBias gradient accumulation uses atomicAdd across row blocks with no
// deterministic variant, so FusedEncoderBackward must refuse to run under the
// flag instead of silently returning order-dependent gradients. The guard
// fires before any CUDA/cuBLAS handle is touched, so this runs without a GPU.
func TestFusedEncoderBackward_RefusesUnderDeterministicMode(t *testing.T) {
	restore := cuda.SetDeterministicEnabledForTesting(true)
	defer restore()

	e := &GPUEngine[float32]{}
	err := e.FusedEncoderBackward(nil, nil, nil, nil, nil, nil, nil, nil, 0, 0, 0, 0, 0, 0, 0)
	if err == nil {
		t.Fatal("FusedEncoderBackward: expected an error under ZTENSOR_DETERMINISTIC=1, got nil")
	}
	if !strings.Contains(err.Error(), "ZTENSOR_DETERMINISTIC") {
		t.Fatalf("FusedEncoderBackward error does not name ZTENSOR_DETERMINISTIC as the cause: %v", err)
	}
}

// TestFusedEncoderBackward_AllowedWithoutDeterministicMode confirms the guard
// is inert (falls through to the ordinary cuBLAS-handle-missing error, not
// the determinism error) when the flag is unset.
func TestFusedEncoderBackward_AllowedWithoutDeterministicMode(t *testing.T) {
	restore := cuda.SetDeterministicEnabledForTesting(false)
	defer restore()

	e := &GPUEngine[float32]{}
	err := e.FusedEncoderBackward(nil, nil, nil, nil, nil, nil, nil, nil, 0, 0, 0, 0, 0, 0, 0)
	if err == nil {
		t.Fatal("FusedEncoderBackward: expected an error (no cuBLAS handle on a bare engine), got nil")
	}
	if strings.Contains(err.Error(), "ZTENSOR_DETERMINISTIC") {
		t.Fatalf("FusedEncoderBackward: determinism guard fired with the flag unset: %v", err)
	}
}

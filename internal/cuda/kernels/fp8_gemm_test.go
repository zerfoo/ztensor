package kernels

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// TestFP8GemmSignature verifies that the FP8Gemm function has the expected signature.
func TestFP8GemmSignature(t *testing.T) {
	_ = assignFunc[func(a, b, c unsafe.Pointer, m, k, n int, scaleA, scaleB float32, stream unsafe.Pointer) error](FP8Gemm)
}

// TestFP8GemmIsFP8GemmSupportedSignature verifies the capability check function.
func TestFP8GemmIsFP8GemmSupportedSignature(t *testing.T) {
	_ = assignFunc[func() bool](IsFP8GemmSupported)
}

// TestFP8GemmKernelLibSymbol verifies that openKernelLib resolves the FP8 GEMM symbol.
func TestFP8GemmKernelLibSymbol(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	k, err := openKernelLib()
	if err != nil {
		t.Fatalf("openKernelLib: %v", err)
	}
	if k == nil {
		t.Fatal("openKernelLib returned nil without error")
	}

	if !IsFP8GemmSupported() {
		t.Skip("FP8 GEMM not supported (requires sm_89+)")
	}

	if k.launchFP8Gemm == 0 {
		t.Error("launchFP8Gemm symbol not resolved")
	}
}

// TestFP8GemmGracefulWithoutCUDA verifies that FP8Gemm returns an error when CUDA is not available.
func TestFP8GemmGracefulWithoutCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("CUDA available, skipping graceful-failure test")
	}

	err := FP8Gemm(nil, nil, nil, 1, 1, 1, 1.0, 1.0, nil)
	if err == nil {
		t.Error("FP8Gemm should return error without CUDA")
	}
}

// TestFP8GemmCorrectness computes a small FP8 matmul and verifies against FP32 reference.
// A[M,K] * B[K,N] = C[M,N], with dequant scales applied.
// Skips if no CUDA or sm_89+ not available.
func TestFP8GemmCorrectness(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	if !IsFP8GemmSupported() {
		t.Skip("FP8 GEMM not supported (requires sm_89+)")
	}

	const M, K, N = 4, 8, 4
	const scaleA float32 = 1.0
	const scaleB float32 = 1.0

	// Create small FP8 E4M3 matrices with known values.
	// Use simple integer-representable values for exact comparison.
	// FP8 E4M3 can exactly represent small integers: 0, 1, 2, 3, 4, ...
	//
	// E4M3 encoding for small values:
	// 0 = 0x00, 1.0 = 0x38, 2.0 = 0x40, -1.0 = 0xB8
	// We use 1.0 (0x38) for simplicity.

	aData := make([]byte, M*K)
	bData := make([]byte, K*N)

	// Fill A with 1.0 (E4M3: exp=7, mant=0 -> 0b0_0111_000 = 0x38).
	for i := range aData {
		aData[i] = 0x38 // 1.0 in FP8 E4M3
	}
	// Fill B with 1.0.
	for i := range bData {
		bData[i] = 0x38 // 1.0 in FP8 E4M3
	}

	// Expected result: C[i,j] = sum(A[i,:] * B[:,j]) * scaleA * scaleB
	// = K * 1.0 * 1.0 * 1.0 * 1.0 = K = 8.0
	// In FP16, 8.0 is exactly representable.
	expectedVal := float32(K) * scaleA * scaleB

	// Allocate device memory.
	sizeA := M * K
	sizeB := K * N
	sizeC := M * N * 2 // FP16 = 2 bytes per element

	devA, err := cuda.Malloc(sizeA)
	if err != nil {
		t.Fatalf("Malloc A: %v", err)
	}
	defer cuda.Free(devA) //nolint:errcheck

	devB, err := cuda.Malloc(sizeB)
	if err != nil {
		t.Fatalf("Malloc B: %v", err)
	}
	defer cuda.Free(devB) //nolint:errcheck

	devC, err := cuda.Malloc(sizeC)
	if err != nil {
		t.Fatalf("Malloc C: %v", err)
	}
	defer cuda.Free(devC) //nolint:errcheck

	// Copy inputs to device.
	if err := cuda.Memcpy(devA, unsafe.Pointer(&aData[0]), sizeA, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A: %v", err)
	}
	if err := cuda.Memcpy(devB, unsafe.Pointer(&bData[0]), sizeB, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy B: %v", err)
	}

	// Launch FP8 GEMM (nil stream = default stream, synchronous with subsequent Memcpy).
	if err := FP8Gemm(devA, devB, devC, M, K, N, scaleA, scaleB, nil); err != nil {
		t.Fatalf("FP8Gemm: %v", err)
	}

	// Copy result back (Memcpy on default stream is synchronous).
	cData := make([]uint16, M*N)
	if err := cuda.Memcpy(unsafe.Pointer(&cData[0]), devC, sizeC, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy C: %v", err)
	}

	// Convert FP16 to FP32 and check.
	// FP16 format: 1 sign, 5 exponent (bias=15), 10 mantissa.
	maxDiff := float32(0)
	outputRange := float32(0)
	for i := 0; i < M*N; i++ {
		got := fp16ToFloat32(cData[i])
		diff := float32(math.Abs(float64(got - expectedVal)))
		if diff > maxDiff {
			maxDiff = diff
		}
		if got > outputRange {
			outputRange = got
		}
	}

	// Acceptance: max diff < 0.1% of output range.
	threshold := outputRange * 0.001
	if maxDiff > threshold {
		t.Errorf("FP8 GEMM correctness: max diff %.6f > threshold %.6f (0.1%% of range %.2f)",
			maxDiff, threshold, outputRange)
	}
}

// fp16ToFloat32 converts a raw FP16 bit pattern to float32.
func fp16ToFloat32(bits uint16) float32 {
	sign := uint32((bits >> 15) & 1)
	exp := uint32((bits >> 10) & 0x1F)
	mant := uint32(bits & 0x03FF)

	switch {
	case exp == 0 && mant == 0:
		// Zero.
		return math.Float32frombits(sign << 31)
	case exp == 0:
		// Subnormal: val = (-1)^sign * 2^(-14) * (mant/1024).
		f := float32(mant) / 1024.0 * (1.0 / 16384.0) // 2^(-14) = 1/16384
		if sign != 0 {
			f = -f
		}
		return f
	case exp == 0x1F:
		// Inf or NaN.
		if mant == 0 {
			return math.Float32frombits((sign << 31) | 0x7F800000)
		}
		return math.Float32frombits((sign << 31) | 0x7FC00000) // NaN
	default:
		// Normal: rebias from FP16 (bias=15) to FP32 (bias=127).
		f32Exp := exp - 15 + 127
		f32Mant := mant << 13
		return math.Float32frombits((sign << 31) | (f32Exp << 23) | f32Mant)
	}
}

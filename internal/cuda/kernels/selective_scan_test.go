package kernels

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// selectiveScanCPU is the sequential CPU reference implementation.
// x:  [batch, d_model, seq_len]
// A:  [d_model, d_state]
// B:  [batch, d_state, seq_len]
// C:  [batch, d_state, seq_len]
// D:  [d_model] (may be nil)
// Returns y: [batch, d_model, seq_len]
func selectiveScanCPU(x, A, B, C, D []float32, batch, dModel, dState, seqLen int) []float32 {
	y := make([]float32, batch*dModel*seqLen)

	for b := 0; b < batch; b++ {
		for d := 0; d < dModel; d++ {
			// Hidden state for this (batch, d_model).
			h := make([]float32, dState)

			for t := 0; t < seqLen; t++ {
				xIdx := (b*dModel+d)*seqLen + t
				xt := x[xIdx]

				var yt float32
				if D != nil {
					yt = D[d] * xt
				}

				for s := 0; s < dState; s++ {
					aDS := A[d*dState+s]
					bt := B[(b*dState+s)*seqLen+t]
					ct := C[(b*dState+s)*seqLen+t]

					h[s] = aDS*h[s] + bt*xt
					yt += ct * h[s]
				}

				y[xIdx] = yt
			}
		}
	}
	return y
}

func TestSelectiveScan(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	if !IsSelectiveScanSupported() {
		t.Skip("selective scan kernel not compiled")
	}

	const (
		batch  = 1
		dModel = 4
		dState = 2
		seqLen = 8
	)

	rng := rand.New(rand.NewSource(42))
	randSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}

	x := randSlice(batch * dModel * seqLen)
	A := randSlice(dModel * dState)
	// Make A values in (-1, 0) range for stability (typical in Mamba).
	for i := range A {
		A[i] = -float32(math.Abs(float64(A[i])))
	}
	B := randSlice(batch * dState * seqLen)
	C := randSlice(batch * dState * seqLen)
	D := randSlice(dModel)

	// CPU reference.
	yCPU := selectiveScanCPU(x, A, B, C, D, batch, dModel, dState, seqLen)

	// GPU computation.
	yGPU := runSelectiveScanGPU(t, x, A, B, C, D, batch, dModel, dState, seqLen)

	// Compare.
	maxDiff := float32(0)
	for i := range yCPU {
		diff := float32(math.Abs(float64(yCPU[i] - yGPU[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 1e-5 {
		t.Errorf("max diff = %e, want < 1e-5", maxDiff)
		// Print first few mismatches.
		for i := range yCPU {
			diff := float32(math.Abs(float64(yCPU[i] - yGPU[i])))
			if diff > 1e-5 {
				t.Errorf("  y[%d]: cpu=%f gpu=%f diff=%e", i, yCPU[i], yGPU[i], diff)
				if i > 10 {
					break
				}
			}
		}
	}
}

func TestSelectiveScanBatch(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	if !IsSelectiveScanSupported() {
		t.Skip("selective scan kernel not compiled")
	}

	const (
		batch  = 4
		dModel = 16
		dState = 4
		seqLen = 32
	)

	rng := rand.New(rand.NewSource(123))
	randSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}

	x := randSlice(batch * dModel * seqLen)
	A := randSlice(dModel * dState)
	for i := range A {
		A[i] = -float32(math.Abs(float64(A[i])))
	}
	B := randSlice(batch * dState * seqLen)
	C := randSlice(batch * dState * seqLen)
	D := randSlice(dModel)

	yCPU := selectiveScanCPU(x, A, B, C, D, batch, dModel, dState, seqLen)
	yGPU := runSelectiveScanGPU(t, x, A, B, C, D, batch, dModel, dState, seqLen)

	maxDiff := float32(0)
	for i := range yCPU {
		diff := float32(math.Abs(float64(yCPU[i] - yGPU[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 1e-5 {
		t.Errorf("max diff = %e, want < 1e-5", maxDiff)
	}
	t.Logf("batch=%d d_model=%d d_state=%d seq_len=%d max_diff=%e",
		batch, dModel, dState, seqLen, maxDiff)
}

func TestSelectiveScanNoD(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	if !IsSelectiveScanSupported() {
		t.Skip("selective scan kernel not compiled")
	}

	const (
		batch  = 2
		dModel = 8
		dState = 4
		seqLen = 16
	)

	rng := rand.New(rand.NewSource(99))
	randSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}

	x := randSlice(batch * dModel * seqLen)
	A := randSlice(dModel * dState)
	for i := range A {
		A[i] = -float32(math.Abs(float64(A[i])))
	}
	B := randSlice(batch * dState * seqLen)
	C := randSlice(batch * dState * seqLen)

	// No D (skip connection).
	yCPU := selectiveScanCPU(x, A, B, C, nil, batch, dModel, dState, seqLen)
	yGPU := runSelectiveScanGPU(t, x, A, B, C, nil, batch, dModel, dState, seqLen)

	maxDiff := float32(0)
	for i := range yCPU {
		diff := float32(math.Abs(float64(yCPU[i] - yGPU[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 1e-5 {
		t.Errorf("max diff = %e, want < 1e-5", maxDiff)
	}
}

func TestSelectiveScanGracefulWithoutCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("CUDA available, skipping graceful-failure test")
	}
	err := SelectiveScanForward(nil, nil, nil, nil, nil, nil, 1, 1, 1, 1, nil)
	if err == nil {
		t.Error("SelectiveScanForward should return error without CUDA")
	}
}

// runSelectiveScanGPU allocates device memory, copies inputs, runs the kernel,
// and returns the output as a host slice.
func runSelectiveScanGPU(t *testing.T, x, A, B, C, D []float32,
	batch, dModel, dState, seqLen int) []float32 {
	t.Helper()

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	allocAndCopy := func(data []float32) unsafe.Pointer {
		size := len(data) * 4
		ptr, err := cuda.Malloc(size)
		if err != nil {
			t.Fatalf("Malloc(%d): %v", size, err)
		}
		if err := cuda.Memcpy(ptr, unsafe.Pointer(&data[0]), size, cuda.MemcpyHostToDevice); err != nil {
			t.Fatalf("Memcpy H2D: %v", err)
		}
		return ptr
	}

	devX := allocAndCopy(x)
	defer func() { _ = cuda.Free(devX) }()
	devA := allocAndCopy(A)
	defer func() { _ = cuda.Free(devA) }()
	devB := allocAndCopy(B)
	defer func() { _ = cuda.Free(devB) }()
	devC := allocAndCopy(C)
	defer func() { _ = cuda.Free(devC) }()

	var devD unsafe.Pointer
	if D != nil {
		devD = allocAndCopy(D)
		defer func() { _ = cuda.Free(devD) }()
	}

	ySize := batch * dModel * seqLen * 4
	devY, err := cuda.Malloc(ySize)
	if err != nil {
		t.Fatalf("Malloc(y): %v", err)
	}
	defer func() { _ = cuda.Free(devY) }()

	if err := SelectiveScanForward(devX, devA, devB, devC, devD, devY,
		batch, dModel, dState, seqLen, stream.Ptr()); err != nil {
		t.Fatalf("SelectiveScanForward: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	result := make([]float32, batch*dModel*seqLen)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devY, ySize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	return result
}

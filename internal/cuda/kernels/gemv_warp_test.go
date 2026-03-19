//go:build cuda

package kernels

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// cpuGemv computes y = A*x on the CPU for reference.
func cpuGemv(A []float32, x []float32, M, N int) []float32 {
	y := make([]float32, M)
	for i := range M {
		var sum float64
		for j := range N {
			sum += float64(A[i*N+j]) * float64(x[j])
		}
		y[i] = float32(sum)
	}
	return y
}

// buildGemvTestData constructs a known A[M x N] matrix and x[N] vector.
func buildGemvTestData(M, N int) (A, x []float32) {
	A = make([]float32, M*N)
	x = make([]float32, N)
	for i := range N {
		x[i] = float32(i%17-8) * 0.05
	}
	for i := range M {
		for j := range N {
			A[i*N+j] = float32(math.Sin(float64(i*N+j)*0.03)) * 1.5
		}
	}
	return A, x
}

func TestGemvWarpF32_Correctness(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	cases := []struct {
		name string
		M, N int
	}{
		{"small_32x64", 32, 64},
		{"medium_128x512", 128, 512},
		{"gemma3_1b_1536x1536", 1536, 1536},
		{"gemma3_1b_6144x1536", 6144, 1536},
		{"odd_N_127x255", 127, 255},
		{"large_4096x4096", 4096, 4096},
		{"single_row_1x1024", 1, 1024},
		{"tiny_4x4", 4, 4},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			A, x := buildGemvTestData(tc.M, tc.N)
			ref := cpuGemv(A, x, tc.M, tc.N)

			stream, err := cuda.CreateStream()
			if err != nil {
				t.Fatalf("CreateStream: %v", err)
			}
			defer func() { _ = stream.Destroy() }()

			devA, err := cuda.Malloc(tc.M * tc.N * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc A: %v", err)
			}
			defer func() { _ = cuda.Free(devA) }()

			devX, err := cuda.Malloc(tc.N * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc x: %v", err)
			}
			defer func() { _ = cuda.Free(devX) }()

			devY, err := cuda.Malloc(tc.M * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc y: %v", err)
			}
			defer func() { _ = cuda.Free(devY) }()

			if err := cuda.Memcpy(devA, unsafe.Pointer(&A[0]), tc.M*tc.N*4, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy A: %v", err)
			}
			if err := cuda.Memcpy(devX, unsafe.Pointer(&x[0]), tc.N*4, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy x: %v", err)
			}

			if err := GemvWarpF32(devY, devA, devX, tc.M, tc.N, stream.Ptr()); err != nil {
				t.Fatalf("GemvWarpF32: %v", err)
			}

			if err := stream.Synchronize(); err != nil {
				t.Fatalf("Synchronize: %v", err)
			}

			got := make([]float32, tc.M)
			if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, tc.M*4, cuda.MemcpyDeviceToHost); err != nil {
				t.Fatalf("Memcpy y: %v", err)
			}

			maxRelErr := 0.0
			for i := range got {
				absRef := math.Abs(float64(ref[i]))
				diff := math.Abs(float64(got[i] - ref[i]))
				var relErr float64
				if absRef > 1e-6 {
					relErr = diff / absRef
				} else {
					relErr = diff
				}
				if relErr > maxRelErr {
					maxRelErr = relErr
				}
				if relErr > 1e-4 {
					t.Errorf("y[%d] = %f, want %f (rel err %e)", i, got[i], ref[i], relErr)
					if t.Failed() {
						break
					}
				}
			}
			t.Logf("max relative error: %e", maxRelErr)
		})
	}
}

func TestGemvWarpF32_TallSkinny(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	// Decode-phase shapes: tall matrix * single vector (N=1 in output sense,
	// but the matrix is M x K where K is the hidden dim).
	cases := []struct {
		name string
		M, N int
	}{
		{"decode_4096x1", 4096, 1},
		{"decode_4096x4096", 4096, 4096},
		{"decode_11008x4096", 11008, 4096},
		{"decode_8192x1", 8192, 1},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			A, x := buildGemvTestData(tc.M, tc.N)
			ref := cpuGemv(A, x, tc.M, tc.N)

			stream, err := cuda.CreateStream()
			if err != nil {
				t.Fatalf("CreateStream: %v", err)
			}
			defer func() { _ = stream.Destroy() }()

			devA, err := cuda.Malloc(tc.M * tc.N * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc A: %v", err)
			}
			defer func() { _ = cuda.Free(devA) }()

			devX, err := cuda.Malloc(tc.N * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc x: %v", err)
			}
			defer func() { _ = cuda.Free(devX) }()

			devY, err := cuda.Malloc(tc.M * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc y: %v", err)
			}
			defer func() { _ = cuda.Free(devY) }()

			if err := cuda.Memcpy(devA, unsafe.Pointer(&A[0]), tc.M*tc.N*4, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy A: %v", err)
			}
			if err := cuda.Memcpy(devX, unsafe.Pointer(&x[0]), tc.N*4, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy x: %v", err)
			}

			if err := GemvWarpF32(devY, devA, devX, tc.M, tc.N, stream.Ptr()); err != nil {
				t.Fatalf("GemvWarpF32: %v", err)
			}

			if err := stream.Synchronize(); err != nil {
				t.Fatalf("Synchronize: %v", err)
			}

			got := make([]float32, tc.M)
			if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, tc.M*4, cuda.MemcpyDeviceToHost); err != nil {
				t.Fatalf("Memcpy y: %v", err)
			}

			maxRelErr := 0.0
			for i := range got {
				absRef := math.Abs(float64(ref[i]))
				diff := math.Abs(float64(got[i] - ref[i]))
				var relErr float64
				if absRef > 1e-6 {
					relErr = diff / absRef
				} else {
					relErr = diff
				}
				if relErr > maxRelErr {
					maxRelErr = relErr
				}
				if relErr > 1e-4 {
					t.Errorf("y[%d] = %f, want %f (rel err %e)", i, got[i], ref[i], relErr)
					if t.Failed() {
						break
					}
				}
			}
			t.Logf("max relative error: %e", maxRelErr)
		})
	}
}

// float32ToFloat16 converts a float32 to IEEE 754 half-precision (uint16).
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 16) & 0x8000
	exp := int((bits>>23)&0xff) - 127 + 15
	frac := bits & 0x007fffff

	if exp <= 0 {
		return uint16(sign)
	}
	if exp >= 31 {
		return uint16(sign | 0x7c00)
	}
	return uint16(sign | uint32(exp)<<10 | (frac >> 13))
}

// float16ToFloat32 converts an IEEE 754 half-precision (uint16) to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff

	if exp == 0 {
		return math.Float32frombits(sign)
	}
	if exp == 31 {
		return math.Float32frombits(sign | 0x7f800000 | (frac << 13))
	}
	return math.Float32frombits(sign | ((exp-15+127)<<23) | (frac << 13))
}

func TestGemvWarpF16_Correctness(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	cases := []struct {
		name string
		M, N int
	}{
		{"small_32x64", 32, 64},
		{"medium_128x512", 128, 512},
		{"decode_1536x1536", 1536, 1536},
		{"even_N_256x256", 256, 256},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Build test data in float32, then convert to FP16.
			Af32, xf32 := buildGemvTestData(tc.M, tc.N)

			// Scale down to avoid FP16 overflow.
			for i := range Af32 {
				Af32[i] *= 0.1
			}
			for i := range xf32 {
				xf32[i] *= 0.1
			}

			// Convert to FP16.
			Af16 := make([]uint16, tc.M*tc.N)
			xf16 := make([]uint16, tc.N)
			for i := range Af32 {
				Af16[i] = float32ToFloat16(Af32[i])
			}
			for i := range xf32 {
				xf16[i] = float32ToFloat16(xf32[i])
			}

			// CPU reference using the actual FP16 values (round-tripped).
			ref := make([]float32, tc.M)
			for i := range tc.M {
				var sum float64
				for j := range tc.N {
					a := float64(float16ToFloat32(Af16[i*tc.N+j]))
					x := float64(float16ToFloat32(xf16[j]))
					sum += a * x
				}
				ref[i] = float32(sum)
			}

			stream, err := cuda.CreateStream()
			if err != nil {
				t.Fatalf("CreateStream: %v", err)
			}
			defer func() { _ = stream.Destroy() }()

			devA, err := cuda.Malloc(tc.M * tc.N * 2)
			if err != nil {
				t.Fatalf("cuda.Malloc A: %v", err)
			}
			defer func() { _ = cuda.Free(devA) }()

			devX, err := cuda.Malloc(tc.N * 2)
			if err != nil {
				t.Fatalf("cuda.Malloc x: %v", err)
			}
			defer func() { _ = cuda.Free(devX) }()

			devY, err := cuda.Malloc(tc.M * 2)
			if err != nil {
				t.Fatalf("cuda.Malloc y: %v", err)
			}
			defer func() { _ = cuda.Free(devY) }()

			if err := cuda.Memcpy(devA, unsafe.Pointer(&Af16[0]), tc.M*tc.N*2, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy A: %v", err)
			}
			if err := cuda.Memcpy(devX, unsafe.Pointer(&xf16[0]), tc.N*2, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy x: %v", err)
			}

			if err := GemvWarpF16(devY, devA, devX, tc.M, tc.N, stream.Ptr()); err != nil {
				t.Fatalf("GemvWarpF16: %v", err)
			}

			if err := stream.Synchronize(); err != nil {
				t.Fatalf("Synchronize: %v", err)
			}

			gotFP16 := make([]uint16, tc.M)
			if err := cuda.Memcpy(unsafe.Pointer(&gotFP16[0]), devY, tc.M*2, cuda.MemcpyDeviceToHost); err != nil {
				t.Fatalf("Memcpy y: %v", err)
			}

			maxRelErr := 0.0
			for i := range gotFP16 {
				got := float16ToFloat32(gotFP16[i])
				absRef := math.Abs(float64(ref[i]))
				diff := math.Abs(float64(got - ref[i]))
				var relErr float64
				if absRef > 1e-4 {
					relErr = diff / absRef
				} else {
					relErr = diff
				}
				if relErr > maxRelErr {
					maxRelErr = relErr
				}
				// FP16 has less precision, allow 1e-2 relative error.
				if relErr > 1e-2 {
					t.Errorf("y[%d] = %f, want %f (rel err %e)", i, got, ref[i], relErr)
					if t.Failed() {
						break
					}
				}
			}
			t.Logf("max relative error: %e", maxRelErr)
		})
	}
}

func BenchmarkGemvWarpF32_4096(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}

	M, N := 4096, 4096
	A, x := buildGemvTestData(M, N)

	stream, err := cuda.CreateStream()
	if err != nil {
		b.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	devA, _ := cuda.Malloc(M * N * 4)
	defer func() { _ = cuda.Free(devA) }()
	devX, _ := cuda.Malloc(N * 4)
	defer func() { _ = cuda.Free(devX) }()
	devY, _ := cuda.Malloc(M * 4)
	defer func() { _ = cuda.Free(devY) }()

	_ = cuda.Memcpy(devA, unsafe.Pointer(&A[0]), M*N*4, cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devX, unsafe.Pointer(&x[0]), N*4, cuda.MemcpyHostToDevice)

	b.ResetTimer()
	for b.Loop() {
		_ = GemvWarpF32(devY, devA, devX, M, N, stream.Ptr())
	}
	_ = stream.Synchronize()

	elapsed := b.Elapsed()
	flops := 2.0 * float64(M) * float64(N) * float64(b.N)
	gflops := flops / elapsed.Seconds() / 1e9
	b.ReportMetric(gflops, "GFLOPS")
}

package kernels

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// nvfp4LUT is the E2M1 magnitude lookup table matching tensor/quantized.go.
var nvfp4LUT = [8]float32{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

// buildFP4TestData constructs M rows of NVFP4 packed blocks, FP16 input vector,
// float16 scales, and computes the FP32 reference GEMV output.
// K must be a multiple of 16.
func buildFP4TestData(M, K int) (packed []byte, scalesU16 []uint16, xFP16 []uint16, ref []float32) {
	const blockSize = 16

	blocksPerRow := K / blockSize

	// Generate float32 weight values and input vector.
	weights := make([]float32, M*K)
	xF32 := make([]float32, K)
	for i := range xF32 {
		xF32[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
	}
	for i := range weights {
		weights[i] = float32(math.Sin(float64(i)*0.03)) * 2.0
	}

	// Quantize weights to NVFP4 E2M1 with block scaling.
	packed = make([]byte, M*blocksPerRow*(blockSize/2))
	scalesU16 = make([]uint16, M*blocksPerRow)

	for row := range M {
		for bi := range blocksPerRow {
			offset := row*K + bi*blockSize
			blockVals := weights[offset : offset+blockSize]

			// Find absmax for block.
			var absMax float32
			for _, v := range blockVals {
				a := v
				if a < 0 {
					a = -a
				}
				if a > absMax {
					absMax = a
				}
			}

			// Scale maps [0, absMax] to [0, 6.0].
			var scale float32
			if absMax > 0 {
				scale = absMax / 6.0
			}
			scaleIdx := row*blocksPerRow + bi
			scalesU16[scaleIdx] = float32ToFloat16Bits(scale)

			var invScale float32
			if scale > 0 {
				invScale = 1.0 / scale
			}

			// Quantize each value.
			packOffset := (row*blocksPerRow + bi) * (blockSize / 2)
			for j := range blockSize {
				v := blockVals[j]
				var sign byte
				if v < 0 {
					sign = 1
					v = -v
				}
				code := encodeE2M1Test(v * invScale)
				fp4 := (sign << 3) | code

				byteIdx := j / 2
				if j%2 == 0 {
					packed[packOffset+byteIdx] = (packed[packOffset+byteIdx] & 0xF0) | fp4
				} else {
					packed[packOffset+byteIdx] = (packed[packOffset+byteIdx] & 0x0F) | (fp4 << 4)
				}
			}
		}
	}

	// Convert input to FP16 bits.
	xFP16 = make([]uint16, K)
	for i := range K {
		xFP16[i] = float32ToFloat16Bits(xF32[i])
	}

	// Compute reference: dequantize and dot product.
	ref = make([]float32, M)
	for row := range M {
		var sum float32
		for bi := range blocksPerRow {
			scaleIdx := row*blocksPerRow + bi
			scale := float16BitsToFloat32(scalesU16[scaleIdx])
			packOffset := (row*blocksPerRow + bi) * (blockSize / 2)

			for j := range blockSize {
				byteIdx := j / 2
				var fp4 byte
				if j%2 == 0 {
					fp4 = packed[packOffset+byteIdx] & 0x0F
				} else {
					fp4 = packed[packOffset+byteIdx] >> 4
				}
				signBit := fp4 >> 3
				mag := nvfp4LUT[fp4&0x07]
				val := mag * scale
				if signBit != 0 {
					val = -val
				}

				k := bi*blockSize + j
				xVal := float16BitsToFloat32(xFP16[k])
				sum += val * xVal
			}
		}
		ref[row] = sum
	}

	return packed, scalesU16, xFP16, ref
}

// encodeE2M1Test converts a non-negative float32 to a 3-bit E2M1 code (0-7).
func encodeE2M1Test(absVal float32) byte {
	best := byte(0)
	bestDist := absVal
	for i := byte(1); i < 8; i++ {
		dist := absVal - nvfp4LUT[i]
		if dist < 0 {
			dist = -dist
		}
		if dist < bestDist {
			bestDist = dist
			best = i
		}
	}
	return best
}

func TestFP4GemvF16_Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	if !IsFP4GemvSupported() {
		t.Skip("NVFP4 GEMV requires sm_100+ (Blackwell)")
	}

	M, K := 64, 256
	packed, scalesU16, xFP16, ref := buildFP4TestData(M, K)

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	// Allocate device memory.
	devW, err := cuda.Malloc(len(packed))
	if err != nil {
		t.Fatalf("cuda.Malloc W: %v", err)
	}
	defer func() { _ = cuda.Free(devW) }()

	devScales, err := cuda.Malloc(len(scalesU16) * 2)
	if err != nil {
		t.Fatalf("cuda.Malloc scales: %v", err)
	}
	defer func() { _ = cuda.Free(devScales) }()

	devX, err := cuda.Malloc(K * 2) // FP16
	if err != nil {
		t.Fatalf("cuda.Malloc x: %v", err)
	}
	defer func() { _ = cuda.Free(devX) }()

	devY, err := cuda.Malloc(M * 4) // FP32
	if err != nil {
		t.Fatalf("cuda.Malloc y: %v", err)
	}
	defer func() { _ = cuda.Free(devY) }()

	// Copy to device.
	if err := cuda.Memcpy(devW, unsafe.Pointer(&packed[0]), len(packed), cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy W: %v", err)
	}
	if err := cuda.Memcpy(devScales, unsafe.Pointer(&scalesU16[0]), len(scalesU16)*2, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy scales: %v", err)
	}
	if err := cuda.Memcpy(devX, unsafe.Pointer(&xFP16[0]), K*2, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy x: %v", err)
	}

	// Run kernel.
	if err := FP4GemvF16(devW, devScales, devX, devY, M, K, stream.Ptr()); err != nil {
		t.Fatalf("FP4GemvF16: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	// Copy result back.
	got := make([]float32, M)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, M*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy y: %v", err)
	}

	// Compute MSE.
	var mse float64
	for i := range got {
		diff := float64(got[i] - ref[i])
		mse += diff * diff
	}
	mse /= float64(M)

	// Also compute reference MSE baseline (FP16 dequant vs FP32 baseline).
	var refMag float64
	for i := range ref {
		refMag += float64(ref[i]) * float64(ref[i])
	}
	refMag /= float64(M)

	// Normalized MSE (relative to signal magnitude).
	var nrmse float64
	if refMag > 1e-12 {
		nrmse = mse / refMag
	}

	t.Logf("MSE: %e, ref magnitude: %e, normalized MSE: %e", mse, refMag, nrmse)

	// Accept < 1% normalized MSE.
	if nrmse > 0.01 {
		t.Errorf("normalized MSE %e exceeds 1%% threshold", nrmse)
	}

	// Also check individual relative errors.
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
	}
	t.Logf("max relative error: %e", maxRelErr)
}

func TestFP4GemvF16_MultipleSizes(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	if !IsFP4GemvSupported() {
		t.Skip("NVFP4 GEMV requires sm_100+ (Blackwell)")
	}

	cases := []struct {
		name string
		M, K int
	}{
		{"small_16x16", 16, 16},
		{"medium_64x256", 64, 256},
		{"wide_128x1024", 128, 1024},
		{"tall_512x256", 512, 256},
		{"large_256x2048", 256, 2048},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			packed, scalesU16, xFP16, ref := buildFP4TestData(tc.M, tc.K)

			stream, err := cuda.CreateStream()
			if err != nil {
				t.Fatalf("CreateStream: %v", err)
			}
			defer func() { _ = stream.Destroy() }()

			devW, err := cuda.Malloc(len(packed))
			if err != nil {
				t.Fatalf("cuda.Malloc W: %v", err)
			}
			defer func() { _ = cuda.Free(devW) }()

			devScales, err := cuda.Malloc(len(scalesU16) * 2)
			if err != nil {
				t.Fatalf("cuda.Malloc scales: %v", err)
			}
			defer func() { _ = cuda.Free(devScales) }()

			devX, err := cuda.Malloc(tc.K * 2)
			if err != nil {
				t.Fatalf("cuda.Malloc x: %v", err)
			}
			defer func() { _ = cuda.Free(devX) }()

			devY, err := cuda.Malloc(tc.M * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc y: %v", err)
			}
			defer func() { _ = cuda.Free(devY) }()

			if err := cuda.Memcpy(devW, unsafe.Pointer(&packed[0]), len(packed), cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy W: %v", err)
			}
			if err := cuda.Memcpy(devScales, unsafe.Pointer(&scalesU16[0]), len(scalesU16)*2, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy scales: %v", err)
			}
			if err := cuda.Memcpy(devX, unsafe.Pointer(&xFP16[0]), tc.K*2, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy x: %v", err)
			}

			if err := FP4GemvF16(devW, devScales, devX, devY, tc.M, tc.K, stream.Ptr()); err != nil {
				t.Fatalf("FP4GemvF16: %v", err)
			}

			if err := stream.Synchronize(); err != nil {
				t.Fatalf("Synchronize: %v", err)
			}

			got := make([]float32, tc.M)
			if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, tc.M*4, cuda.MemcpyDeviceToHost); err != nil {
				t.Fatalf("Memcpy y: %v", err)
			}

			var mse, refMag float64
			for i := range got {
				diff := float64(got[i] - ref[i])
				mse += diff * diff
				refMag += float64(ref[i]) * float64(ref[i])
			}
			mse /= float64(tc.M)
			refMag /= float64(tc.M)

			var nrmse float64
			if refMag > 1e-12 {
				nrmse = mse / refMag
			}

			t.Logf("MSE: %e, normalized MSE: %e", mse, nrmse)
			if nrmse > 0.01 {
				t.Errorf("normalized MSE %e exceeds 1%% threshold", nrmse)
			}
		})
	}
}

func TestIsFP4GemvSupported(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	// Just verify the function doesn't panic and returns a boolean.
	supported := IsFP4GemvSupported()
	t.Logf("IsFP4GemvSupported: %v", supported)
}

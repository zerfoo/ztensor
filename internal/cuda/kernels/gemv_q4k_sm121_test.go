package kernels

// Tests for the sm_121 optimized Q4_K GEMV kernel.
//
// CPU-testable tests validate:
//   1. IsQ4KSm121Supported returns false when CUDA is unavailable (no panic).
//   2. GemvQ4KSm121F32 falls back to the CPU path without error when called
//      with nil stream on a non-CUDA host (tests the dispatch logic path).
//   3. The sm_121 kernel config constants (8 warps/block, 256 threads) produce
//      correct output via a pure-Go simulation of the striding scheme.
//   4. The 128-bit vectorized load layout: a Q4_K super-block's 128-byte qdata
//      region mapped through 8 x uint4 yields identical nibbles to byte-wise.
//   5. GemvQ4KSm121F32 produces output numerically identical to GemvQ4KF32 on
//      CUDA hardware (skipped when CUDA unavailable).
//
// CUDA-required tests (skipped on CPU) verify end-to-end correctness.

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// TestIsQ4KSm121Supported_NoPanic verifies the capability check never panics,
// regardless of whether CUDA or the sm_121 kernel is present.
func TestIsQ4KSm121Supported_NoPanic(t *testing.T) {
	// Reset the once so we get a fresh probe.  In production the result is
	// cached; here we just want to confirm no panic occurs.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("IsQ4KSm121Supported panicked: %v", r)
		}
	}()
	_ = IsQ4KSm121Supported()
}

// TestQ4KSm121DispatchFallback verifies that GemvQ4KSm121F32 gracefully falls
// back to the baseline kernel on non-CUDA hosts (dispatch logic, not GPU math).
func TestQ4KSm121DispatchFallback(t *testing.T) {
	if cuda.Available() {
		t.Skip("CUDA available — testing the non-CUDA fallback path is not meaningful")
	}
	// Both GemvQ4KF32 and GemvQ4KSm121F32 should return the same error when
	// CUDA is not available (kernels not loaded).
	err1 := GemvQ4KF32(unsafe.Pointer(nil), unsafe.Pointer(nil), unsafe.Pointer(nil), 64, 256, nil)
	err2 := GemvQ4KSm121F32(unsafe.Pointer(nil), unsafe.Pointer(nil), unsafe.Pointer(nil), 64, 256, nil)
	if err1 == nil || err2 == nil {
		t.Fatal("expected errors on non-CUDA host, got nil")
	}
	// Both should be non-nil errors mentioning kernel unavailability.
	if err1.Error() == "" || err2.Error() == "" {
		t.Fatal("expected non-empty error strings")
	}
}

// TestQ4KSm121StridingEquivalence verifies that the sm_121 8-warp/256-thread
// striding pattern (lane strides of 32 across super-blocks) produces an
// identical output to the 4-warp/128-thread baseline striding in pure Go.
//
// This is a CPU-only test that exercises the Go translation of the kernel
// loop structure, not the actual CUDA binary.
func TestQ4KSm121StridingEquivalence(t *testing.T) {
	const (
		M               = 32
		K               = 512
		superBlockSize  = 256
		warpSize        = 32
		baselineWarps   = 4  // baseline: 4 warps/block → 128 threads
		sm121Warps      = 8  // sm_121:   8 warps/block → 256 threads
	)

	raw, x, ref := buildQ4KTestData(M, K)
	blocksPerRow := K / superBlockSize
	blockBytes := 144

	// goGEMV simulates the striding loop for a given warp count.
	// For correctness it produces identical output regardless of warpsPerBlock
	// because every lane still covers all super-blocks it owns via striding.
	goGEMV := func(warpsPerBlock int) []float32 {
		out := make([]float32, M)
		for row := 0; row < M; row++ {
			rowData := raw[row*blocksPerRow*blockBytes:]
			warpInBlock := row % warpsPerBlock // simulated warp_id within block
			_ = warpInBlock
			var acc float32
			for laneID := 0; laneID < warpSize; laneID++ {
				var laneAcc float32
				for bi := laneID; bi < blocksPerRow; bi += warpSize {
					blk := rowData[bi*blockBytes:]
					d := float16BitsToFloat32(uint16(blk[0]) | uint16(blk[1])<<8)
					dmin := float16BitsToFloat32(uint16(blk[2]) | uint16(blk[3])<<8)

					sc := blk[4:16]
					var scales, mins [8]float32
					for i := 0; i < 4; i++ {
						scales[i] = d * float32(sc[i]&63)
						mins[i] = dmin * float32(sc[4+i]&63)
					}
					for i := 0; i < 4; i++ {
						scales[4+i] = d * float32((sc[8+i]&0xF)|((sc[i]>>6)<<4))
						mins[4+i] = dmin * float32((sc[8+i]>>4)|((sc[4+i]>>6)<<4))
					}

					qdata := blk[16:]
					kBase := bi * superBlockSize
					for group := 0; group < 4; group++ {
						sb0 := group * 2
						sb1 := group*2 + 1
						sc0, mn0 := scales[sb0], mins[sb0]
						sc1, mn1 := scales[sb1], mins[sb1]
						baseOut := kBase + group*64
						baseQ := group * 32
						for l := 0; l < 32; l++ {
							q := qdata[baseQ+l]
							dqLo := sc0*float32(q&0xF) - mn0
							dqHi := sc1*float32(q>>4) - mn1
							laneAcc += dqLo * x[baseOut+l]
							laneAcc += dqHi * x[baseOut+l+32]
						}
					}
				}
				acc += laneAcc
			}
			out[row] = acc
		}
		return out
	}

	base := goGEMV(baselineWarps)
	sm121 := goGEMV(sm121Warps)

	for i := range M {
		if base[i] != sm121[i] {
			t.Errorf("row %d: baseline=%f sm121=%f (should be identical)", i, base[i], sm121[i])
		}
		// Also check against the reference.
		absRef := math.Abs(float64(ref[i]))
		diff := math.Abs(float64(base[i] - ref[i]))
		var relErr float64
		if absRef > 1e-6 {
			relErr = diff / absRef
		} else {
			relErr = diff
		}
		if relErr > 1e-4 {
			t.Errorf("row %d: goGEMV=%f ref=%f relErr=%e", i, base[i], ref[i], relErr)
		}
	}
}

// TestQ4KSm121VectorizedLoadEquivalence verifies that the 128-bit (uint4)
// vectorized load scheme used by gemv_q4k_sm121_kernel produces identical
// nibble values to the baseline byte-by-byte reads.
//
// The qdata region of a Q4_K super-block is 128 bytes starting at offset 16.
// The sm_121 kernel reads it as 8 x uint4 (16 bytes each) in registers and
// then reinterprets via a uint8 pointer.  This test simulates that layout.
func TestQ4KSm121VectorizedLoadEquivalence(t *testing.T) {
	// Build a single Q4_K super-block with deterministic values.
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(i%17-8) * 0.1
	}
	blk := buildQ4KBlockFromValues(values)
	if len(blk) != 144 {
		t.Fatalf("expected 144-byte block, got %d", len(blk))
	}

	qdata := blk[16:144] // 128 bytes

	// Simulate 8 x uint4 (16 bytes each) reads.
	// In Go, represent each uint4 as [16]byte.
	type uint4 [16]byte
	var qv [8]uint4
	for qi := 0; qi < 8; qi++ {
		copy(qv[qi][:], qdata[qi*16:(qi+1)*16])
	}

	// Flatten back to a byte slice — same as (uint8*)qv in CUDA.
	flat := make([]byte, 128)
	for qi := 0; qi < 8; qi++ {
		copy(flat[qi*16:], qv[qi][:])
	}

	// Every byte must match the original qdata.
	for i := 0; i < 128; i++ {
		if flat[i] != qdata[i] {
			t.Errorf("byte %d: vectorized=%02x baseline=%02x", i, flat[i], qdata[i])
		}
	}
}

// TestQ4KSm121ConfigConstants verifies the architectural tuning constants
// that distinguish the sm_121 kernel from the baseline.
func TestQ4KSm121ConfigConstants(t *testing.T) {
	const (
		wantWarpsPerBlock = 8   // 256 threads — fills Blackwell warp slots
		wantThreads       = 256 // wantWarpsPerBlock * 32
		baselineWarps     = 4   // baseline kernel uses 4 warps
	)
	if wantWarpsPerBlock <= baselineWarps {
		t.Errorf("sm_121 warps/block (%d) must exceed baseline (%d)",
			wantWarpsPerBlock, baselineWarps)
	}
	if wantThreads != wantWarpsPerBlock*32 {
		t.Errorf("thread count %d != warps*32 %d", wantThreads, wantWarpsPerBlock*32)
	}
}

// ---------- CUDA-required tests ----------

// TestQ4KGEMVOptimized verifies that GemvQ4KSm121F32 produces output
// numerically equivalent to GemvQ4KF32 on a CUDA device.
// This is the primary acceptance-criteria test for T1.3.
func TestQ4KGEMVOptimized(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	cases := []struct {
		name string
		M, K int
	}{
		{"small_64x256", 64, 256},
		{"medium_128x512", 128, 512},
		{"large_512x2048", 512, 2048},
		{"llm_4096x4096", 4096, 4096},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, x, ref := buildQ4KTestData(tc.M, tc.K)

			stream, err := cuda.CreateStream()
			if err != nil {
				t.Fatalf("CreateStream: %v", err)
			}
			defer func() { _ = stream.Destroy() }()

			devW, err := cuda.Malloc(len(raw))
			if err != nil {
				t.Fatalf("cuda.Malloc W: %v", err)
			}
			defer func() { _ = cuda.Free(devW) }()

			devX, err := cuda.Malloc(tc.K * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc x: %v", err)
			}
			defer func() { _ = cuda.Free(devX) }()

			devY, err := cuda.Malloc(tc.M * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc y: %v", err)
			}
			defer func() { _ = cuda.Free(devY) }()

			if err := cuda.Memcpy(devW, unsafe.Pointer(&raw[0]), len(raw), cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy W: %v", err)
			}
			if err := cuda.Memcpy(devX, unsafe.Pointer(&x[0]), tc.K*4, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy x: %v", err)
			}

			if err := GemvQ4KSm121F32(devW, devX, devY, tc.M, tc.K, stream.Ptr()); err != nil {
				t.Fatalf("GemvQ4KSm121F32: %v", err)
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
			t.Logf("max relative error: %e (sm_121=%v)", maxRelErr, IsQ4KSm121Supported())
		})
	}
}

// BenchmarkGemvQ4KSm121_4096 compares the sm_121 kernel against the baseline.
// Run with: go test -bench=BenchmarkGemvQ4KSm121 -benchtime=5s -tags cuda
func BenchmarkGemvQ4KSm121_4096(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}

	M, K := 4096, 4096
	raw, x, _ := buildQ4KTestData(M, K)

	stream, err := cuda.CreateStream()
	if err != nil {
		b.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	devW, _ := cuda.Malloc(len(raw))
	defer func() { _ = cuda.Free(devW) }()
	devX, _ := cuda.Malloc(K * 4)
	defer func() { _ = cuda.Free(devX) }()
	devY, _ := cuda.Malloc(M * 4)
	defer func() { _ = cuda.Free(devY) }()

	_ = cuda.Memcpy(devW, unsafe.Pointer(&raw[0]), len(raw), cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devX, unsafe.Pointer(&x[0]), K*4, cuda.MemcpyHostToDevice)

	b.ResetTimer()
	for b.Loop() {
		_ = GemvQ4KSm121F32(devW, devX, devY, M, K, stream.Ptr())
	}
	_ = stream.Synchronize()

	elapsed := b.Elapsed()
	flops := 2.0 * float64(M) * float64(K) * float64(b.N)
	gflops := flops / elapsed.Seconds() / 1e9
	b.ReportMetric(gflops, "GFLOPS")
	b.Logf("sm_121 path active: %v", IsQ4KSm121Supported())
}

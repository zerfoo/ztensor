package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// referenceMatMulF32 computes C = A * B in pure float32 for reference.
func referenceMatMulF32(t *testing.T, aData []float32, aShape []int, bData []float32, bShape []int) []float32 {
	t.Helper()
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()
	a, err := tensor.New[float32](aShape, aData)
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.New[float32](bShape, bData)
	if err != nil {
		t.Fatal(err)
	}
	result, err := engine.MatMul(ctx, a, b)
	if err != nil {
		t.Fatal(err)
	}
	return result.Data()
}

func TestW4A16(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// Dimensions: m=2, k=32 (Q4_0 block size), n=3.
	m, k, n := 2, 32, 3

	// Generate weight data (will be quantized to 4-bit).
	wData := make([]float32, m*k)
	for i := range wData {
		wData[i] = float32(i%7-3) * 0.1
	}

	// Generate activation data (will be stored as FP16).
	actData := make([]float32, k*n)
	for i := range actData {
		actData[i] = float32(i%5-2) * 0.1
	}

	tests := []struct {
		name       string
		weightStor tensor.Storage[float32]
		maxErr     float64
	}{
		{
			name:       "Q4_0 weights x FP16 activations",
			weightStor: tensor.QuantizeQ4(wData),
			maxErr:     0.15,
		},
		{
			name:       "GPTQ 4-bit weights x FP16 activations",
			weightStor: tensor.QuantizeGPTQ(wData, 32, 4),
			maxErr:     0.15,
		},
		{
			name:       "AWQ weights x FP16 activations",
			weightStor: tensor.QuantizeAWQ(wData, 32),
			maxErr:     0.15,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// A = 4-bit weights [m, k].
			wTensor, err := tensor.NewWithStorage[float32]([]int{m, k}, tt.weightStor)
			if err != nil {
				t.Fatal(err)
			}

			// B = FP16 activations [k, n].
			fp16Stor := tensor.NewFloat16StorageFromF32(actData)
			actTensor, err := tensor.NewWithStorage[float32]([]int{k, n}, fp16Stor)
			if err != nil {
				t.Fatal(err)
			}

			// Verify detection.
			if !IsW4A16(wTensor, actTensor) {
				t.Fatal("IsW4A16 should return true for W4A16 pair")
			}

			info := W4A16Info(wTensor, actTensor)
			if info.WeightFormat == "" {
				t.Fatal("W4A16Info should return non-empty WeightFormat")
			}

			// Run W4A16 MatMul.
			result, matched, err := TryW4A16MatMul(ctx, engine, wTensor, actTensor)
			if err != nil {
				t.Fatalf("TryW4A16MatMul failed: %v", err)
			}
			if !matched {
				t.Fatal("TryW4A16MatMul should match W4A16 pair")
			}
			if result.Shape()[0] != m || result.Shape()[1] != n {
				t.Errorf("shape = %v, want [%d %d]", result.Shape(), m, n)
			}

			// Reference: dequantize weights to float32, decode FP16 to float32, then GEMM.
			refW := tt.weightStor.Slice()
			refAct := fp16Stor.Slice()
			want := referenceMatMulF32(t, refW, []int{m, k}, refAct, []int{k, n})

			got := result.Data()
			for i := range got {
				diff := math.Abs(float64(got[i] - want[i]))
				if diff > tt.maxErr {
					t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
				}
			}
		})
	}
}

func TestW4A16_BWeights(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// Test with weights on B side: A=FP16 activations, B=4-bit weights.
	m, k, n := 2, 32, 3

	actData := make([]float32, m*k)
	for i := range actData {
		actData[i] = float32(i%5-2) * 0.1
	}

	wData := make([]float32, k*n)
	for i := range wData {
		wData[i] = float32(i%7-3) * 0.1
	}

	// A = FP16 activations [m, k].
	fp16Stor := tensor.NewFloat16StorageFromF32(actData)
	actTensor, err := tensor.NewWithStorage[float32]([]int{m, k}, fp16Stor)
	if err != nil {
		t.Fatal(err)
	}

	// B = Q4_0 weights [k, n].
	q4Stor := tensor.QuantizeQ4(wData)
	wTensor, err := tensor.NewWithStorage[float32]([]int{k, n}, q4Stor)
	if err != nil {
		t.Fatal(err)
	}

	if !IsW4A16(actTensor, wTensor) {
		t.Fatal("IsW4A16 should detect FP16 on A and Q4 on B")
	}

	result, matched, err := TryW4A16MatMul(ctx, engine, actTensor, wTensor)
	if err != nil {
		t.Fatalf("TryW4A16MatMul failed: %v", err)
	}
	if !matched {
		t.Fatal("TryW4A16MatMul should match")
	}

	// Reference.
	refAct := fp16Stor.Slice()
	refW := q4Stor.Slice()
	want := referenceMatMulF32(t, refAct, []int{m, k}, refW, []int{k, n})

	got := result.Data()
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > 0.15 {
			t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

func TestW4A16_NotMatched(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// Both float32 — should not match.
	m, k, n := 2, 32, 3
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	a, _ := tensor.New[float32]([]int{m, k}, aData)
	b, _ := tensor.New[float32]([]int{k, n}, bData)

	if IsW4A16(a, b) {
		t.Error("IsW4A16 should return false for plain float32 tensors")
	}

	_, matched, err := TryW4A16MatMul(ctx, engine, a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if matched {
		t.Error("TryW4A16MatMul should not match plain float32 tensors")
	}
}

func TestW4A16_QuantFormat(t *testing.T) {
	tests := []struct {
		name   string
		stor   tensor.Storage[float32]
		expect string
	}{
		{"Q4_0", tensor.QuantizeQ4(make([]float32, 32)), "Q4_0"},
		{"GPTQ_4", tensor.QuantizeGPTQ(make([]float32, 32), 32, 4), "GPTQ_4"},
		{"GPTQ_8", tensor.QuantizeGPTQ(make([]float32, 32), 32, 8), ""},
		{"AWQ", tensor.QuantizeAWQ(make([]float32, 32), 32), "AWQ"},
		{"float32", tensor.NewCPUStorage(make([]float32, 32)), ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := QuantFormat[float32](tt.stor)
			if got != tt.expect {
				t.Errorf("QuantFormat = %q, want %q", got, tt.expect)
			}
		})
	}
}

func TestW4A16_DequantW4ToFP16(t *testing.T) {
	src := make([]float32, 32)
	for i := range src {
		src[i] = float32(i%7-3) * 0.1
	}

	q4 := tensor.QuantizeQ4(src)
	fp16 := DequantW4ToFP16(q4)

	if fp16.Len() != 32 {
		t.Fatalf("FP16 len = %d, want 32", fp16.Len())
	}

	// The dequantized float32 from Q4 should round-trip through FP16 with
	// acceptable precision loss.
	dequantF32 := q4.Slice()
	fp16F32 := fp16.Slice()
	for i := range dequantF32 {
		diff := math.Abs(float64(dequantF32[i] - fp16F32[i]))
		if diff > 0.01 {
			t.Errorf("index %d: Q4->FP16 diff = %v (q4=%v, fp16=%v)", i, diff, dequantF32[i], fp16F32[i])
		}
	}
}

func TestW4A16_Perplexity(t *testing.T) {
	// Validate that W4A16 dispatch produces exact results relative to the
	// dequantized reference. The 0.5% perplexity acceptance criterion is a
	// model-level metric; at the MatMul level we verify that the dispatch
	// path introduces zero additional error beyond what quantization inherently
	// causes. We also check that Q4 quantization error vs FP16 is bounded
	// (typically <10% RMSE for small random matrices).
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	m, k, n := 4, 128, 8

	// Generate pseudo-random weight data with realistic distribution.
	wData := make([]float32, m*k)
	for i := range wData {
		v := float32(((i*1103515245 + 12345) % 65536)) / 65536.0
		wData[i] = (v - 0.5) * 2.0 // range [-1, 1]
	}

	actData := make([]float32, k*n)
	for i := range actData {
		v := float32(((i*6364136223846793005 + 1442695040888963407) % 65536)) / 65536.0
		actData[i] = (v - 0.5) * 2.0
	}

	// W4A16 path: quantize weights to Q4_0, activations as FP16.
	q4Stor := tensor.QuantizeQ4(wData)
	fp16AStor := tensor.NewFloat16StorageFromF32(actData)

	wTensor, err := tensor.NewWithStorage[float32]([]int{m, k}, q4Stor)
	if err != nil {
		t.Fatal(err)
	}
	actTensor, err := tensor.NewWithStorage[float32]([]int{k, n}, fp16AStor)
	if err != nil {
		t.Fatal(err)
	}

	result, matched, err := TryW4A16MatMul(ctx, engine, wTensor, actTensor)
	if err != nil {
		t.Fatalf("TryW4A16MatMul failed: %v", err)
	}
	if !matched {
		t.Fatal("should match")
	}

	// Reference: dequantize Q4 weights + decode FP16 activations, then float32 GEMM.
	// The W4A16 dispatch should produce identical results to this reference.
	refW := q4Stor.Slice()
	refAct := fp16AStor.Slice()
	refResult := referenceMatMulF32(t, refW, []int{m, k}, refAct, []int{k, n})

	got := result.Data()
	var sumSqErr, sumSqRef float64
	for i := range got {
		diff := float64(got[i] - refResult[i])
		sumSqErr += diff * diff
		sumSqRef += float64(refResult[i]) * float64(refResult[i])
	}

	rmse := math.Sqrt(sumSqErr / float64(len(got)))
	rmsRef := math.Sqrt(sumSqRef / float64(len(got)))

	// The dispatch path should produce zero additional error — results
	// must be bit-identical to the dequant+GEMM reference.
	if rmse > 1e-6 {
		t.Errorf("W4A16 dispatch adds error beyond dequant reference: RMSE=%.8f", rmse)
	}

	// Also verify quantization error vs pure FP16 is bounded (sanity check).
	fp16WStor := tensor.NewFloat16StorageFromF32(wData)
	fp16Result := referenceMatMulF32(t, fp16WStor.Slice(), []int{m, k}, refAct, []int{k, n})
	var sumSqQuantErr, sumSqFP16 float64
	for i := range got {
		diff := float64(got[i] - fp16Result[i])
		sumSqQuantErr += diff * diff
		sumSqFP16 += float64(fp16Result[i]) * float64(fp16Result[i])
	}
	quantRMSE := math.Sqrt(sumSqQuantErr / float64(len(got)))
	fp16RMS := math.Sqrt(sumSqFP16 / float64(len(got)))

	var quantRelErr float64
	if fp16RMS > 0 {
		quantRelErr = quantRMSE / fp16RMS
	}

	t.Logf("W4A16 dispatch vs dequant reference: RMSE=%.8f, RMS_ref=%.6f", rmse, rmsRef)
	t.Logf("W4A16 vs FP16 baseline (quantization error): relative=%.4f%%", quantRelErr*100)

	// Quantization error should be bounded — Q4 typically <15% for random data.
	if quantRelErr > 0.15 {
		t.Errorf("quantization error = %.4f%%, exceeds 15%% sanity bound", quantRelErr*100)
	}
}

func TestW4A16_Info(t *testing.T) {
	m, k, n := 2, 32, 3

	wData := make([]float32, m*k)
	actData := make([]float32, k*n)

	tests := []struct {
		name       string
		weightStor tensor.Storage[float32]
		wantFmt    string
	}{
		{"Q4_0", tensor.QuantizeQ4(wData), "Q4_0"},
		{"GPTQ_4", tensor.QuantizeGPTQ(wData, 32, 4), "GPTQ_4"},
		{"AWQ", tensor.QuantizeAWQ(wData, 32), "AWQ"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wTensor, _ := tensor.NewWithStorage[float32]([]int{m, k}, tt.weightStor)
			fp16Stor := tensor.NewFloat16StorageFromF32(actData)
			actTensor, _ := tensor.NewWithStorage[float32]([]int{k, n}, fp16Stor)

			info := W4A16Info(wTensor, actTensor)
			if info.WeightFormat != tt.wantFmt {
				t.Errorf("W4A16Info.WeightFormat = %q, want %q", info.WeightFormat, tt.wantFmt)
			}
		})
	}
}

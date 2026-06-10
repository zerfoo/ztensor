package oracle

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestEncodeDecodeRoundTripF32 round-trips float32 values through the raw
// little-endian codec. f32-representable values must survive exactly.
func TestEncodeDecodeRoundTripF32(t *testing.T) {
	vals := []float64{0, 1, -1, 0.5, -2.25, float64(float32(3.14159)), math.MaxFloat32, -math.SmallestNonzeroFloat32}
	raw, err := EncodeTensor(DTypeFloat32, vals)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 4*len(vals) {
		t.Fatalf("encoded %d bytes, want %d", len(raw), 4*len(vals))
	}
	got, err := DecodeTensor(DTypeFloat32, raw)
	if err != nil {
		t.Fatal(err)
	}
	for i := range vals {
		if got[i] != vals[i] {
			t.Fatalf("element %d: got %v, want %v", i, got[i], vals[i])
		}
	}
}

// TestEncodeDecodeRoundTripF64 round-trips float64 values bit-exactly.
func TestEncodeDecodeRoundTripF64(t *testing.T) {
	vals := []float64{0, math.Pi, -math.E, 1e-300, -1e300, math.SmallestNonzeroFloat64}
	raw, err := EncodeTensor(DTypeFloat64, vals)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 8*len(vals) {
		t.Fatalf("encoded %d bytes, want %d", len(raw), 8*len(vals))
	}
	got, err := DecodeTensor(DTypeFloat64, raw)
	if err != nil {
		t.Fatal(err)
	}
	for i := range vals {
		if got[i] != vals[i] {
			t.Fatalf("element %d: got %v, want %v", i, got[i], vals[i])
		}
	}
}

// TestEncodeLittleEndian pins the byte order: 1.0f32 is 00 00 80 3F little-
// endian (what numpy dtype('<f4') reads), and 1.0f64 is 00..00 F0 3F.
func TestEncodeLittleEndian(t *testing.T) {
	raw32, err := EncodeTensor(DTypeFloat32, []float64{1.0})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw32, []byte{0x00, 0x00, 0x80, 0x3F}) {
		t.Fatalf("f32 1.0 encoded as % X, want 00 00 80 3F", raw32)
	}
	raw64, err := EncodeTensor(DTypeFloat64, []float64{1.0})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw64, []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F}) {
		t.Fatalf("f64 1.0 encoded as % X, want 00 00 00 00 00 00 F0 3F", raw64)
	}
}

// TestDecodeRejectsBadInput covers unsupported dtypes and truncated payloads.
func TestDecodeRejectsBadInput(t *testing.T) {
	if _, err := EncodeTensor("int8", []float64{1}); err == nil {
		t.Fatal("EncodeTensor accepted an unsupported dtype")
	}
	if _, err := DecodeTensor(DTypeFloat32, []byte{1, 2, 3}); err == nil {
		t.Fatal("DecodeTensor accepted a truncated f32 payload")
	}
	if _, err := DecodeTensor(DTypeFloat64, make([]byte, 12)); err == nil {
		t.Fatal("DecodeTensor accepted a truncated f64 payload")
	}
}

// TestBundleWriteReadRoundTrip writes a complete bundle (manifest + tensor
// files in both dtypes) and reads it back through the public API.
func TestBundleWriteReadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	in := TensorRef{Name: "x0", File: "input_0.bin", Shape: []int{2, 2}, DType: DTypeFloat32}
	fwd := TensorRef{Name: "forward", File: "forward.bin", Shape: []int{2, 2}, DType: DTypeFloat64}
	up := TensorRef{Name: "upstream", File: "upstream.bin", Shape: []int{2, 2}, DType: DTypeFloat32}
	gin := TensorRef{Name: "x0", File: "grad_input_0.bin", Shape: []int{2, 2}, DType: DTypeFloat32}

	inVals := []float64{0.25, -1.5, 2, -3}
	fwdVals := []float64{math.Pi, -math.E, 0.1, 1e-12}
	upVals := []float64{1, -1, 0.5, -0.5}
	ginVals := []float64{4, 3, 2, 1}
	for _, w := range []struct {
		ref  TensorRef
		vals []float64
	}{{in, inVals}, {fwd, fwdVals}, {up, upVals}, {gin, ginVals}} {
		if err := WriteTensorFile(filepath.Join(dir, w.ref.File), w.ref.DType, w.vals); err != nil {
			t.Fatal(err)
		}
	}

	m := &Manifest{
		Op:         "RoundTrip",
		TorchExpr:  "torch.tanh(x0)",
		DType:      DTypeFloat32,
		Seed:       42,
		Tolerance:  defaultTolerance,
		Inputs:     []TensorRef{in},
		Upstream:   up,
		Forward:    fwd,
		InputGrads: []TensorRef{gin},
	}
	if err := WriteBundle(dir, m); err != nil {
		t.Fatal(err)
	}

	b, err := ReadBundle(dir)
	if err != nil {
		t.Fatal(err)
	}
	if b.Manifest.FormatVersion != FormatVersion {
		t.Fatalf("format_version = %d, want %d", b.Manifest.FormatVersion, FormatVersion)
	}
	if b.Manifest.Op != "RoundTrip" || b.Manifest.TorchExpr != "torch.tanh(x0)" || b.Manifest.Seed != 42 {
		t.Fatalf("manifest did not round-trip: %+v", b.Manifest)
	}
	if b.Manifest.Tolerance != defaultTolerance {
		t.Fatalf("tolerance did not round-trip: %+v", b.Manifest.Tolerance)
	}

	gotIn, err := b.Tensor(b.Manifest.Inputs[0])
	if err != nil {
		t.Fatal(err)
	}
	for i := range inVals {
		if gotIn[i] != inVals[i] {
			t.Fatalf("input[%d] = %v, want %v", i, gotIn[i], inVals[i])
		}
	}
	// f64 forward must round-trip bit-exactly, including the tiny 1e-12.
	gotFwd, err := b.Tensor(b.Manifest.Forward)
	if err != nil {
		t.Fatal(err)
	}
	for i := range fwdVals {
		if gotFwd[i] != fwdVals[i] {
			t.Fatalf("forward[%d] = %v, want %v", i, gotFwd[i], fwdVals[i])
		}
	}
}

// TestReadTensorFileShapeMismatch rejects a file whose element count
// disagrees with the manifest shape.
func TestReadTensorFileShapeMismatch(t *testing.T) {
	dir := t.TempDir()
	ref := TensorRef{Name: "x0", File: "x.bin", Shape: []int{3, 3}, DType: DTypeFloat32}
	if err := WriteTensorFile(filepath.Join(dir, ref.File), ref.DType, []float64{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadTensorFile(dir, ref); err == nil {
		t.Fatal("ReadTensorFile accepted a 4-element file for a 3x3 shape")
	}
}

// TestReadBundleRejectsUnknownVersion guards forward compatibility.
func TestReadBundleRejectsUnknownVersion(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(`{"format_version": 99, "op": "X"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadBundle(dir); err == nil {
		t.Fatal("ReadBundle accepted format_version 99")
	}
}

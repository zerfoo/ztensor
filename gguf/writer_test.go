package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// testReader provides minimal GGUF v3 reading for round-trip verification.
type testReader struct {
	data []byte
	pos  int
}

func newTestReader(data []byte) *testReader { return &testReader{data: data} }

func (r *testReader) u32() uint32 {
	v := binary.LittleEndian.Uint32(r.data[r.pos:])
	r.pos += 4
	return v
}

func (r *testReader) u64() uint64 {
	v := binary.LittleEndian.Uint64(r.data[r.pos:])
	r.pos += 8
	return v
}

func (r *testReader) str() string {
	n := int(r.u64())
	s := string(r.data[r.pos : r.pos+n])
	r.pos += n
	return s
}

func (r *testReader) readBytes(n int) []byte {
	b := make([]byte, n)
	copy(b, r.data[r.pos:r.pos+n])
	r.pos += n
	return b
}

func (r *testReader) alignTo(a int) {
	if rem := r.pos % a; rem != 0 {
		r.pos += a - rem
	}
}

func TestEmptyWriter(t *testing.T) {
	w := NewWriter()
	var buf bytes.Buffer
	if err := w.Write(&buf); err != nil {
		t.Fatalf("Write: %v", err)
	}

	r := newTestReader(buf.Bytes())
	if got := r.u32(); got != Magic {
		t.Fatalf("magic = %#x, want %#x", got, Magic)
	}
	if got := r.u32(); got != Version {
		t.Fatalf("version = %d, want %d", got, Version)
	}
	if got := r.u64(); got != 0 {
		t.Fatalf("tensor_count = %d, want 0", got)
	}
	if got := r.u64(); got != 0 {
		t.Fatalf("metadata_kv_count = %d, want 0", got)
	}
	// Minimal valid GGUF: 4+4+8+8 = 24 bytes.
	if buf.Len() != 24 {
		t.Fatalf("empty file size = %d, want 24", buf.Len())
	}
}

func TestMagicAndVersionBytes(t *testing.T) {
	w := NewWriter()
	var buf bytes.Buffer
	if err := w.Write(&buf); err != nil {
		t.Fatalf("Write: %v", err)
	}
	raw := buf.Bytes()

	// Verify "GGUF" appears as bytes 47, 47, 55, 46 (little-endian 0x46554747).
	if raw[0] != 0x47 || raw[1] != 0x47 || raw[2] != 0x55 || raw[3] != 0x46 {
		t.Fatalf("magic bytes = [%#x, %#x, %#x, %#x], want [0x47, 0x47, 0x55, 0x46]",
			raw[0], raw[1], raw[2], raw[3])
	}
	// Version 3 in LE.
	if raw[4] != 3 || raw[5] != 0 || raw[6] != 0 || raw[7] != 0 {
		t.Fatalf("version bytes = [%d, %d, %d, %d], want [3, 0, 0, 0]",
			raw[4], raw[5], raw[6], raw[7])
	}
}

func TestMetadataRoundTrip(t *testing.T) {
	w := NewWriter()
	w.AddMetadataString("general.name", "test-model")
	w.AddMetadataUint32("general.file_type", 7)
	w.AddMetadataFloat32("score", 0.95)
	w.AddMetadataBool("quantized", true)
	w.AddMetadataUint64("total_size", 1<<40)
	w.AddMetadataStringArray("general.tags", []string{"llm", "test"})
	w.AddMetadataUint32Array("general.dims", []uint32{128, 256})

	var buf bytes.Buffer
	if err := w.Write(&buf); err != nil {
		t.Fatalf("Write: %v", err)
	}

	r := newTestReader(buf.Bytes())
	r.u32() // magic
	r.u32() // version
	r.u64() // tensor count
	kvCount := r.u64()
	if kvCount != 7 {
		t.Fatalf("metadata count = %d, want 7", kvCount)
	}

	// 1: string
	if key := r.str(); key != "general.name" {
		t.Fatalf("key = %q, want general.name", key)
	}
	if vt := r.u32(); vt != MetaTypeString {
		t.Fatalf("type = %d, want %d", vt, MetaTypeString)
	}
	if val := r.str(); val != "test-model" {
		t.Fatalf("value = %q, want test-model", val)
	}

	// 2: uint32
	if key := r.str(); key != "general.file_type" {
		t.Fatalf("key = %q", key)
	}
	if vt := r.u32(); vt != MetaTypeUint32 {
		t.Fatalf("type = %d, want %d", vt, MetaTypeUint32)
	}
	if val := r.u32(); val != 7 {
		t.Fatalf("value = %d, want 7", val)
	}

	// 3: float32
	if key := r.str(); key != "score" {
		t.Fatalf("key = %q", key)
	}
	if vt := r.u32(); vt != MetaTypeFloat32 {
		t.Fatalf("type = %d, want %d", vt, MetaTypeFloat32)
	}
	if val := math.Float32frombits(r.u32()); val != 0.95 {
		t.Fatalf("value = %f, want 0.95", val)
	}

	// 4: bool (true)
	if key := r.str(); key != "quantized" {
		t.Fatalf("key = %q", key)
	}
	if vt := r.u32(); vt != MetaTypeBool {
		t.Fatalf("type = %d, want %d", vt, MetaTypeBool)
	}
	if val := r.readBytes(1)[0]; val != 1 {
		t.Fatalf("bool value = %d, want 1", val)
	}

	// 5: uint64
	if key := r.str(); key != "total_size" {
		t.Fatalf("key = %q", key)
	}
	if vt := r.u32(); vt != MetaTypeUint64 {
		t.Fatalf("type = %d, want %d", vt, MetaTypeUint64)
	}
	if val := r.u64(); val != 1<<40 {
		t.Fatalf("value = %d, want %d", val, uint64(1<<40))
	}

	// 6: string array
	if key := r.str(); key != "general.tags" {
		t.Fatalf("key = %q", key)
	}
	if vt := r.u32(); vt != MetaTypeArray {
		t.Fatalf("type = %d, want %d", vt, MetaTypeArray)
	}
	if elemType := r.u32(); elemType != MetaTypeString {
		t.Fatalf("array elem type = %d, want %d", elemType, MetaTypeString)
	}
	if count := r.u64(); count != 2 {
		t.Fatalf("array count = %d, want 2", count)
	}
	if s := r.str(); s != "llm" {
		t.Fatalf("array[0] = %q, want llm", s)
	}
	if s := r.str(); s != "test" {
		t.Fatalf("array[1] = %q, want test", s)
	}

	// 7: uint32 array
	if key := r.str(); key != "general.dims" {
		t.Fatalf("key = %q", key)
	}
	if vt := r.u32(); vt != MetaTypeArray {
		t.Fatalf("type = %d, want %d", vt, MetaTypeArray)
	}
	if elemType := r.u32(); elemType != MetaTypeUint32 {
		t.Fatalf("array elem type = %d, want %d", elemType, MetaTypeUint32)
	}
	if count := r.u64(); count != 2 {
		t.Fatalf("array count = %d, want 2", count)
	}
	if val := r.u32(); val != 128 {
		t.Fatalf("array[0] = %d, want 128", val)
	}
	if val := r.u32(); val != 256 {
		t.Fatalf("array[1] = %d, want 256", val)
	}
}

func TestBoolFalse(t *testing.T) {
	w := NewWriter()
	w.AddMetadataBool("flag", false)

	var buf bytes.Buffer
	if err := w.Write(&buf); err != nil {
		t.Fatalf("Write: %v", err)
	}

	r := newTestReader(buf.Bytes())
	r.u32() // magic
	r.u32() // version
	r.u64() // tensor count
	r.u64() // kv count
	r.str() // key
	r.u32() // type
	if val := r.readBytes(1)[0]; val != 0 {
		t.Fatalf("bool value = %d, want 0", val)
	}
}

func TestAddTensorF32(t *testing.T) {
	w := NewWriter()
	floats := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	w.AddTensorF32("weight", []int{2, 3}, floats)

	var buf bytes.Buffer
	if err := w.Write(&buf); err != nil {
		t.Fatalf("Write: %v", err)
	}

	raw := buf.Bytes()
	r := newTestReader(raw)

	// Header.
	r.u32() // magic
	r.u32() // version
	if tc := r.u64(); tc != 1 {
		t.Fatalf("tensor count = %d, want 1", tc)
	}
	r.u64() // kv count = 0

	// Tensor info.
	if name := r.str(); name != "weight" {
		t.Fatalf("name = %q, want weight", name)
	}
	if nDims := r.u32(); nDims != 2 {
		t.Fatalf("n_dims = %d, want 2", nDims)
	}
	// Reversed: expect 3, 2.
	if d0 := r.u64(); d0 != 3 {
		t.Fatalf("dim[0] = %d, want 3", d0)
	}
	if d1 := r.u64(); d1 != 2 {
		t.Fatalf("dim[1] = %d, want 2", d1)
	}
	if dtype := r.u32(); dtype != TypeF32 {
		t.Fatalf("dtype = %d, want %d", dtype, TypeF32)
	}
	if offset := r.u64(); offset != 0 {
		t.Fatalf("offset = %d, want 0", offset)
	}

	// Alignment padding then tensor data.
	r.alignTo(Alignment)
	if r.pos%Alignment != 0 {
		t.Fatalf("tensor data starts at %d, not aligned to %d", r.pos, Alignment)
	}

	for i, want := range floats {
		got := math.Float32frombits(binary.LittleEndian.Uint32(raw[r.pos+i*4:]))
		if got != want {
			t.Fatalf("float[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestSingleTensorRawBytes(t *testing.T) {
	w := NewWriter()

	shape := []int{2, 3}
	data := make([]byte, 2*3*4)
	for i := range 6 {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i)))
	}

	w.AddTensor("weight", TypeF32, shape, data)

	var buf bytes.Buffer
	if err := w.Write(&buf); err != nil {
		t.Fatalf("Write: %v", err)
	}

	raw := buf.Bytes()
	r := newTestReader(raw)

	r.u32() // magic
	r.u32() // version
	r.u64() // tensor count
	r.u64() // kv count

	// Tensor info.
	r.str() // name
	nDims := r.u32()
	for range nDims {
		r.u64()
	}
	r.u32() // dtype
	r.u64() // offset

	r.alignTo(Alignment)

	got := r.readBytes(len(data))
	if !bytes.Equal(got, data) {
		t.Fatalf("tensor data mismatch")
	}
}

func TestMultipleTensorsAlignment(t *testing.T) {
	w := NewWriter()

	// First tensor: 3 floats (12 bytes -- not aligned to 32).
	data1 := make([]byte, 3*4)
	for i := range 3 {
		binary.LittleEndian.PutUint32(data1[i*4:], math.Float32bits(float32(i)))
	}
	w.AddTensor("a", TypeF32, []int{3}, data1)

	// Second tensor: 5 floats (20 bytes).
	data2 := make([]byte, 5*4)
	for i := range 5 {
		binary.LittleEndian.PutUint32(data2[i*4:], math.Float32bits(float32(i+10)))
	}
	w.AddTensor("b", TypeF32, []int{5}, data2)

	var buf bytes.Buffer
	if err := w.Write(&buf); err != nil {
		t.Fatalf("Write: %v", err)
	}

	raw := buf.Bytes()
	r := newTestReader(raw)

	r.u32() // magic
	r.u32() // version
	r.u64() // tensor count
	r.u64() // kv count

	// Read tensor info for "a".
	r.str() // name
	nDimsA := r.u32()
	for range nDimsA {
		r.u64()
	}
	r.u32() // dtype
	offsetA := r.u64()

	// Read tensor info for "b".
	r.str() // name
	nDimsB := r.u32()
	for range nDimsB {
		r.u64()
	}
	r.u32() // dtype
	offsetB := r.u64()

	if offsetA != 0 {
		t.Fatalf("tensor a offset = %d, want 0", offsetA)
	}

	// 12 bytes of data -> next 32-byte boundary = 32.
	if offsetB%Alignment != 0 {
		t.Fatalf("tensor b offset = %d, not aligned to %d", offsetB, Alignment)
	}
	if offsetB != Alignment {
		t.Fatalf("tensor b offset = %d, want %d", offsetB, Alignment)
	}

	// Verify actual data.
	r.alignTo(Alignment)
	dataStart := r.pos

	gotA := raw[dataStart+int(offsetA) : dataStart+int(offsetA)+len(data1)]
	if !bytes.Equal(gotA, data1) {
		t.Fatalf("tensor a data mismatch")
	}

	gotB := raw[dataStart+int(offsetB) : dataStart+int(offsetB)+len(data2)]
	if !bytes.Equal(gotB, data2) {
		t.Fatalf("tensor b data mismatch")
	}
}

func TestByteForByteRoundTrip(t *testing.T) {
	build := func() []byte {
		w := NewWriter()
		w.AddMetadataString("model", "round-trip-test")
		w.AddMetadataUint32("version", 1)

		data := make([]byte, 4*4)
		for i := range 4 {
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i)*1.5))
		}
		w.AddTensor("embed", TypeF32, []int{4}, data)

		var buf bytes.Buffer
		if err := w.Write(&buf); err != nil {
			panic(err)
		}
		return buf.Bytes()
	}

	first := build()
	second := build()

	if !bytes.Equal(first, second) {
		t.Fatalf("two identical writes produced different output (%d vs %d bytes)", len(first), len(second))
	}
}

func TestInt32Metadata(t *testing.T) {
	w := NewWriter()
	w.AddMetadataInt32("offset", -42)

	var buf bytes.Buffer
	if err := w.Write(&buf); err != nil {
		t.Fatalf("Write: %v", err)
	}

	r := newTestReader(buf.Bytes())
	r.u32() // magic
	r.u32() // version
	r.u64() // tensor count
	r.u64() // kv count
	r.str() // key
	r.u32() // type
	val := int32(r.u32())
	if val != -42 {
		t.Fatalf("int32 value = %d, want -42", val)
	}
}

package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

type kvPair struct {
	key       string
	valueType uint32
	value     any
}

type tensorInfo struct {
	name  string
	dtype int
	shape []int
	data  []byte
}

// Writer buffers GGUF v3 metadata and tensor data, then writes the complete
// file in a single Write call.
type Writer struct {
	metadata []kvPair
	tensors  []tensorInfo
}

// NewWriter returns a new GGUF v3 writer.
func NewWriter() *Writer {
	return &Writer{}
}

// AddMetadataString adds a string metadata key-value pair.
func (w *Writer) AddMetadataString(key, value string) {
	w.metadata = append(w.metadata, kvPair{key: key, valueType: MetaTypeString, value: value})
}

// AddMetadataUint32 adds a uint32 metadata key-value pair.
func (w *Writer) AddMetadataUint32(key string, value uint32) {
	w.metadata = append(w.metadata, kvPair{key: key, valueType: MetaTypeUint32, value: value})
}

// AddMetadataInt32 adds an int32 metadata key-value pair.
func (w *Writer) AddMetadataInt32(key string, value int32) {
	w.metadata = append(w.metadata, kvPair{key: key, valueType: MetaTypeInt32, value: value})
}

// AddMetadataFloat32 adds a float32 metadata key-value pair.
func (w *Writer) AddMetadataFloat32(key string, value float32) {
	w.metadata = append(w.metadata, kvPair{key: key, valueType: MetaTypeFloat32, value: value})
}

// AddMetadataBool adds a bool metadata key-value pair.
func (w *Writer) AddMetadataBool(key string, value bool) {
	w.metadata = append(w.metadata, kvPair{key: key, valueType: MetaTypeBool, value: value})
}

// AddMetadataUint64 adds a uint64 metadata key-value pair.
func (w *Writer) AddMetadataUint64(key string, value uint64) {
	w.metadata = append(w.metadata, kvPair{key: key, valueType: MetaTypeUint64, value: value})
}

// AddMetadataInt64 adds an int64 metadata key-value pair.
func (w *Writer) AddMetadataInt64(key string, value int64) {
	w.metadata = append(w.metadata, kvPair{key: key, valueType: MetaTypeInt64, value: value})
}

// AddMetadataStringArray adds a string array metadata key-value pair.
func (w *Writer) AddMetadataStringArray(key string, values []string) {
	w.metadata = append(w.metadata, kvPair{key: key, valueType: MetaTypeArray, value: stringArray(values)})
}

// AddMetadataUint32Array adds a uint32 array metadata key-value pair.
func (w *Writer) AddMetadataUint32Array(key string, values []uint32) {
	w.metadata = append(w.metadata, kvPair{key: key, valueType: MetaTypeArray, value: uint32Array(values)})
}

// stringArray is a typed wrapper so writeMetadataValue can distinguish array element types.
type stringArray []string

// uint32Array is a typed wrapper so writeMetadataValue can distinguish array element types.
type uint32Array []uint32

// AddTensor registers a tensor with raw byte data. shape uses the caller's
// convention (outermost dimension first); the writer reverses dimensions when
// serializing per the GGUF spec.
func (w *Writer) AddTensor(name string, typ int, shape []int, data []byte) {
	w.tensors = append(w.tensors, tensorInfo{name: name, dtype: typ, shape: shape, data: data})
}

// AddTensorF32 is a convenience method that registers a float32 tensor,
// converting the float32 slice to raw bytes.
func (w *Writer) AddTensorF32(name string, shape []int, data []float32) {
	b := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	w.AddTensor(name, TypeF32, shape, b)
}

// Write outputs the complete GGUF v3 file to out.
func (w *Writer) Write(out io.Writer) error {
	var written int64
	write := func(v any) error {
		if err := binary.Write(out, binary.LittleEndian, v); err != nil {
			return err
		}
		written += int64(binary.Size(v))
		return nil
	}
	writeBytes := func(b []byte) error {
		n, err := out.Write(b)
		written += int64(n)
		return err
	}
	writeString := func(s string) error {
		if err := write(uint64(len(s))); err != nil {
			return err
		}
		return writeBytes([]byte(s))
	}
	writePad := func() error {
		rem := written % Alignment
		if rem == 0 {
			return nil
		}
		pad := make([]byte, Alignment-rem)
		return writeBytes(pad)
	}

	// Header: magic, version, tensor count, KV count.
	if err := write(uint32(Magic)); err != nil {
		return fmt.Errorf("write magic: %w", err)
	}
	if err := write(uint32(Version)); err != nil {
		return fmt.Errorf("write version: %w", err)
	}
	if err := write(uint64(len(w.tensors))); err != nil {
		return fmt.Errorf("write tensor count: %w", err)
	}
	if err := write(uint64(len(w.metadata))); err != nil {
		return fmt.Errorf("write metadata count: %w", err)
	}

	// Metadata KV pairs.
	for _, kv := range w.metadata {
		if err := writeString(kv.key); err != nil {
			return fmt.Errorf("write metadata key %q: %w", kv.key, err)
		}
		if err := write(kv.valueType); err != nil {
			return fmt.Errorf("write metadata type %q: %w", kv.key, err)
		}
		if err := writeMetadataValue(write, writeBytes, writeString, kv); err != nil {
			return fmt.Errorf("write metadata value %q: %w", kv.key, err)
		}
	}

	// Tensor info entries. Compute each tensor's offset relative to the start
	// of the tensor data section.
	var dataOffset uint64
	for _, t := range w.tensors {
		if err := writeString(t.name); err != nil {
			return fmt.Errorf("write tensor name %q: %w", t.name, err)
		}
		nDims := uint32(len(t.shape))
		if err := write(nDims); err != nil {
			return err
		}
		// Reverse dimensions for GGUF (innermost first).
		for i := len(t.shape) - 1; i >= 0; i-- {
			if err := write(uint64(t.shape[i])); err != nil {
				return err
			}
		}
		if err := write(uint32(t.dtype)); err != nil {
			return err
		}
		if err := write(dataOffset); err != nil {
			return err
		}
		dataOffset += uint64(len(t.data))
		if rem := dataOffset % Alignment; rem != 0 {
			dataOffset += Alignment - rem
		}
	}

	// Pad to alignment boundary before tensor data (only if there are tensors).
	if len(w.tensors) > 0 {
		if err := writePad(); err != nil {
			return fmt.Errorf("write header padding: %w", err)
		}
	}

	// Tensor data, each padded to alignment boundary.
	for _, t := range w.tensors {
		if err := writeBytes(t.data); err != nil {
			return fmt.Errorf("write tensor data %q: %w", t.name, err)
		}
		if err := writePad(); err != nil {
			return fmt.Errorf("write tensor padding %q: %w", t.name, err)
		}
	}

	return nil
}

func writeMetadataValue(
	write func(any) error,
	writeBytes func([]byte) error,
	writeString func(string) error,
	kv kvPair,
) error {
	switch kv.valueType {
	case MetaTypeUint32:
		return write(kv.value.(uint32))
	case MetaTypeInt32:
		return write(kv.value.(int32))
	case MetaTypeFloat32:
		return write(math.Float32bits(kv.value.(float32)))
	case MetaTypeBool:
		var b uint8
		if kv.value.(bool) {
			b = 1
		}
		return write(b)
	case MetaTypeString:
		return writeString(kv.value.(string))
	case MetaTypeUint64:
		return write(kv.value.(uint64))
	case MetaTypeInt64:
		return write(kv.value.(int64))
	case MetaTypeArray:
		switch arr := kv.value.(type) {
		case stringArray:
			if err := write(uint32(MetaTypeString)); err != nil {
				return err
			}
			if err := write(uint64(len(arr))); err != nil {
				return err
			}
			for _, s := range arr {
				if err := writeString(s); err != nil {
					return err
				}
			}
			return nil
		case uint32Array:
			if err := write(uint32(MetaTypeUint32)); err != nil {
				return err
			}
			if err := write(uint64(len(arr))); err != nil {
				return err
			}
			for _, v := range arr {
				if err := write(v); err != nil {
					return err
				}
			}
			return nil
		default:
			return fmt.Errorf("unsupported array type %T", kv.value)
		}
	default:
		return fmt.Errorf("unsupported metadata type %d", kv.valueType)
	}
}

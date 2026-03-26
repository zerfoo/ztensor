// Package gguf provides a shared GGUF v3 writer for serializing model files.
// It handles binary format concerns (header, metadata KV pairs, tensor info,
// aligned tensor data) but has zero domain knowledge about model architectures,
// tensor naming, or tokenizer embedding.
package gguf

// GGUF v3 format constants.
const (
	Magic     = 0x46554747 // "GGUF" in little-endian
	Version   = 3
	Alignment = 32 // tensor data alignment in bytes
)

// GGML tensor types.
const (
	TypeF32  = 0
	TypeF16  = 1
	TypeQ4_0 = 2
	TypeQ4_1 = 3
	TypeQ5_0 = 6
	TypeQ5_1 = 7
	TypeQ8_0 = 8
	TypeQ4_K = 12
	TypeQ5_K = 13
	TypeQ6_K = 14
	TypeBF16 = 30
)

// GGUF metadata value types.
const (
	MetaTypeUint8   = 0
	MetaTypeInt8    = 1
	MetaTypeUint16  = 2
	MetaTypeInt16   = 3
	MetaTypeUint32  = 4
	MetaTypeInt32   = 5
	MetaTypeFloat32 = 6
	MetaTypeBool    = 7
	MetaTypeString  = 8
	MetaTypeArray   = 9
	MetaTypeUint64  = 10
	MetaTypeInt64   = 11
	MetaTypeFloat64 = 12
)

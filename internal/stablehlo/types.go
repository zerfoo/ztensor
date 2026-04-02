// Package stablehlo generates StableHLO MLIR text for PJRT compilation.
//
// It provides type mapping (Go types to MLIR tensor type strings),
// SSA value naming (%v0, %v1, ...), shape formatting (tensor<2x3x4xf32>),
// and StableHLO operation name constants.
package stablehlo

import (
	"fmt"
	"strings"
	"sync"
)

// MLIR dtype strings for StableHLO tensor types.
const (
	DTypeF32  = "f32"
	DTypeF64  = "f64"
	DTypeF16  = "f16"
	DTypeBF16 = "bf16"
	DTypeF8   = "f8E4M3FN"
	DTypeI8   = "i8"
	DTypeI16  = "i16"
	DTypeI32  = "i32"
	DTypeI64  = "i64"
	DTypeUI8  = "ui8"
	DTypeUI32 = "ui32"
	DTypeUI64 = "ui64"
	DTypeBool = "i1"
)

// StableHLO operation name constants.
const (
	OpAdd         = "stablehlo.add"
	OpSubtract    = "stablehlo.subtract"
	OpMultiply    = "stablehlo.multiply"
	OpDivide      = "stablehlo.divide"
	OpDotGeneral  = "stablehlo.dot_general"
	OpTranspose   = "stablehlo.transpose"
	OpReshape     = "stablehlo.reshape"
	OpBroadcastIn = "stablehlo.broadcast_in_dim"
	OpReduce      = "stablehlo.reduce"
	OpGather      = "stablehlo.gather"
	OpSlice       = "stablehlo.slice"
	OpConcatenate = "stablehlo.concatenate"
	OpExp         = "stablehlo.exponential"
	OpLog         = "stablehlo.log"
	OpSin         = "stablehlo.sine"
	OpCos         = "stablehlo.cosine"
	OpTanh        = "stablehlo.tanh"
	OpNegate      = "stablehlo.negate"
	OpAbs         = "stablehlo.abs"
	OpSqrt        = "stablehlo.sqrt"
	OpRsqrt       = "stablehlo.rsqrt"
	OpMaximum     = "stablehlo.maximum"
	OpMinimum     = "stablehlo.minimum"
	OpClamp       = "stablehlo.clamp"
	OpSelect      = "stablehlo.select"
	OpCompare     = "stablehlo.compare"
	OpConvert     = "stablehlo.convert"
	OpConstant    = "stablehlo.constant"
	OpIota        = "stablehlo.iota"
	OpPower       = "stablehlo.power"
)

// SSANamer generates monotonically increasing SSA value names (%v0, %v1, ...).
type SSANamer struct {
	mu      sync.Mutex
	counter int
}

// NextName returns the next SSA value name and advances the counter.
func (n *SSANamer) NextName() string {
	n.mu.Lock()
	name := fmt.Sprintf("%%v%d", n.counter)
	n.counter++
	n.mu.Unlock()
	return name
}

// Count returns the current counter value (number of names issued so far).
func (n *SSANamer) Count() int {
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.counter
}

// FormatTensorType formats a MLIR tensor type string from a shape and dtype.
// Example: FormatTensorType([]int{2, 3, 4}, "f32") returns "tensor<2x3x4xf32>".
// For scalar tensors (empty shape), it returns "tensor<f32>".
func FormatTensorType(shape []int, dtype string) string {
	if len(shape) == 0 {
		return "tensor<" + dtype + ">"
	}
	var b strings.Builder
	b.WriteString("tensor<")
	for _, dim := range shape {
		fmt.Fprintf(&b, "%dx", dim)
	}
	b.WriteString(dtype)
	b.WriteByte('>')
	return b.String()
}

// FormatScalarType returns the MLIR scalar type string for a dtype.
// Example: FormatScalarType("f32") returns "f32".
func FormatScalarType(dtype string) string {
	return dtype
}

// GoDTypeToMLIR maps a Go reflect type name to a MLIR dtype string.
// Supported mappings:
//
//	float32  -> f32
//	float64  -> f64
//	float16  -> f16
//	bfloat16 -> bf16
//	float8   -> f8E4M3FN
//	int8     -> i8
//	int16    -> i16
//	int32    -> i32
//	int64    -> i64
//	uint8    -> ui8
//	uint32   -> ui32
//	uint64   -> ui64
//
// Returns the MLIR dtype string and true if the mapping exists, or ("", false) otherwise.
func GoDTypeToMLIR(goType string) (string, bool) {
	dtype, ok := goTypeMap[goType]
	return dtype, ok
}

var goTypeMap = map[string]string{
	"float32":  DTypeF32,
	"float64":  DTypeF64,
	"float16":  DTypeF16,
	"bfloat16": DTypeBF16,
	"float8":   DTypeF8,
	"int8":     DTypeI8,
	"int16":    DTypeI16,
	"int32":    DTypeI32,
	"int64":    DTypeI64,
	"uint8":    DTypeUI8,
	"uint32":   DTypeUI32,
	"uint64":   DTypeUI64,
}

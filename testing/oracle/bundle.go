// Package oracle implements the ztensor side of the PyTorch-as-oracle per-op
// parity harness (zerfoo docs/adr/091, plan T1.3).
//
// The harness runs the same op with the same inputs through ztensor and
// through PyTorch (inside nvcr.io/nvidia/pytorch:26.02-py3 on the DGX GB10)
// and diffs forward AND backward outputs within per-op tolerances. It catches
// numerics-convention divergence -- fast-math approximations, reduction
// ordering, eps placement -- that ztensor's CPU and GPU engines could share,
// which gradcheck (testing/gradcheck) cannot see because both sides of a
// gradcheck comparison are ztensor.
//
// This package is TEST INFRASTRUCTURE: the production stack stays pure Go.
// The only PyTorch dependency is the offline Python runner
// (scripts/oracle/run_oracle.py) executed in the pinned NGC container.
//
// # Case-bundle exchange format (format_version 1)
//
// One directory per op case:
//
//	<dir>/<OpName>/
//	    manifest.json        # everything the runner needs (see Manifest)
//	    input_0.bin ...      # op inputs
//	    param_<name>.bin     # trainable parameter values (if any)
//	    upstream.bin         # upstream gradient dL/dy fed to Backward
//	    forward.bin          # ztensor forward output
//	    grad_input_0.bin ... # ztensor input gradients
//	    grad_param_<name>.bin# ztensor parameter gradients (if any)
//
// Tensor files are raw little-endian IEEE-754 values in row-major (C) order
// with no header; the manifest records shape and dtype ("float32"/"float64",
// matching numpy names). The Python runner reconstructs every input and
// parameter as a torch leaf tensor with requires_grad=True, evaluates
// Manifest.TorchExpr, backprops the recorded upstream gradient, and compares
// torch's forward output and gradients against the recorded ztensor ones.
package oracle

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/zerfoo/ztensor/tensor"
)

// FormatVersion is the current case-bundle format version.
const FormatVersion = 1

// Supported dtype strings (numpy names; the runner maps them to '<f4'/'<f8').
const (
	DTypeFloat32 = "float32"
	DTypeFloat64 = "float64"
)

// TensorRef locates one tensor file within a bundle directory.
type TensorRef struct {
	// Name is the binding name in the torch expression namespace
	// ("x0", "x1", ... for inputs; the parameter name for params).
	Name string `json:"name"`
	// File is the file name relative to the bundle directory.
	File string `json:"file"`
	// Shape is the row-major tensor shape.
	Shape []int `json:"shape"`
	// DType is "float32" or "float64".
	DType string `json:"dtype"`
}

// ParamRef is a TensorRef for a trainable parameter plus the file holding the
// ztensor-recorded gradient for that parameter.
type ParamRef struct {
	TensorRef
	GradFile string `json:"grad_file"`
}

// Tolerance holds the per-op comparison tolerances for forward output and
// gradients. An element passes when |got-ref| <= Atol + Rtol*|ref| (ref is
// the torch value), mirroring numpy.allclose / torch.testing conventions.
type Tolerance struct {
	FwdAtol  float64 `json:"fwd_atol"`
	FwdRtol  float64 `json:"fwd_rtol"`
	GradAtol float64 `json:"grad_atol"`
	GradRtol float64 `json:"grad_rtol"`
}

// Manifest is the bundle's manifest.json: everything the Python runner needs
// to replay the case in torch and judge the diffs.
type Manifest struct {
	FormatVersion int    `json:"format_version"`
	Op            string `json:"op"`
	// TorchExpr is a Python expression over `torch`, the inputs (bound as
	// x0..xn), and the parameters (bound by name), evaluating to the op's
	// forward output. Example: "torch.nn.functional.layer_norm(x0, (4,),
	// weight=gamma.reshape(4), bias=beta.reshape(4), eps=1e-05)".
	TorchExpr string `json:"torch_expr"`
	// DType is the bundle-wide element type ("float32" or "float64").
	DType string `json:"dtype"`
	// Seed is the input/upstream generation seed (provenance only).
	Seed      int64       `json:"seed"`
	Tolerance Tolerance   `json:"tolerance"`
	Inputs    []TensorRef `json:"inputs"`
	Params    []ParamRef  `json:"params,omitempty"`
	Upstream  TensorRef   `json:"upstream"`
	// Forward is the recorded ztensor forward output.
	Forward TensorRef `json:"forward"`
	// InputGrads are the recorded ztensor input gradients, aligned with Inputs.
	InputGrads []TensorRef `json:"input_grads"`
}

// dtypeSize returns the byte width of a supported dtype.
func dtypeSize(dtype string) (int, error) {
	switch dtype {
	case DTypeFloat32:
		return 4, nil
	case DTypeFloat64:
		return 8, nil
	default:
		return 0, fmt.Errorf("oracle: unsupported dtype %q", dtype)
	}
}

func shapeSize(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// EncodeTensor serializes values as raw little-endian bytes of the given
// dtype (row-major order is the caller's responsibility; ztensor tensors are
// already row-major).
func EncodeTensor(dtype string, values []float64) ([]byte, error) {
	width, err := dtypeSize(dtype)
	if err != nil {
		return nil, err
	}
	buf := make([]byte, len(values)*width)
	for i, v := range values {
		switch dtype {
		case DTypeFloat32:
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(v)))
		case DTypeFloat64:
			binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
		}
	}
	return buf, nil
}

// DecodeTensor parses raw little-endian bytes of the given dtype, widening to
// float64 for comparison.
func DecodeTensor(dtype string, raw []byte) ([]float64, error) {
	width, err := dtypeSize(dtype)
	if err != nil {
		return nil, err
	}
	if len(raw)%width != 0 {
		return nil, fmt.Errorf("oracle: %d bytes is not a multiple of %s width %d", len(raw), dtype, width)
	}
	out := make([]float64, len(raw)/width)
	for i := range out {
		switch dtype {
		case DTypeFloat32:
			out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
		case DTypeFloat64:
			out[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
		}
	}
	return out, nil
}

// WriteTensorFile writes values to path as raw little-endian dtype bytes.
func WriteTensorFile(path, dtype string, values []float64) error {
	raw, err := EncodeTensor(dtype, values)
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		return fmt.Errorf("oracle: writing %s: %w", path, err)
	}
	return nil
}

// ReadTensorFile reads a raw little-endian tensor file and validates the
// element count against the reference's shape.
func ReadTensorFile(dir string, ref TensorRef) ([]float64, error) {
	raw, err := os.ReadFile(filepath.Join(dir, ref.File))
	if err != nil {
		return nil, fmt.Errorf("oracle: reading %s: %w", ref.File, err)
	}
	vals, err := DecodeTensor(ref.DType, raw)
	if err != nil {
		return nil, fmt.Errorf("oracle: decoding %s: %w", ref.File, err)
	}
	if want := shapeSize(ref.Shape); len(vals) != want {
		return nil, fmt.Errorf("oracle: %s holds %d elements, shape %v wants %d", ref.File, len(vals), ref.Shape, want)
	}
	return vals, nil
}

// Bundle is one on-disk op case: the parsed manifest plus its directory.
type Bundle struct {
	Dir      string
	Manifest Manifest
}

// WriteBundle writes manifest.json into dir (tensor files are written by the
// generator before this call; WriteBundle is last so a manifest implies a
// complete bundle).
func WriteBundle(dir string, m *Manifest) error {
	if m.FormatVersion == 0 {
		m.FormatVersion = FormatVersion
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("oracle: marshaling manifest for %s: %w", m.Op, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), append(b, '\n'), 0o600); err != nil {
		return fmt.Errorf("oracle: writing manifest for %s: %w", m.Op, err)
	}
	return nil
}

// ReadBundle parses dir/manifest.json.
func ReadBundle(dir string) (*Bundle, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("oracle: reading manifest in %s: %w", dir, err)
	}
	var m Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("oracle: parsing manifest in %s: %w", dir, err)
	}
	if m.FormatVersion != FormatVersion {
		return nil, fmt.Errorf("oracle: bundle %s has format_version %d, this reader supports %d", dir, m.FormatVersion, FormatVersion)
	}
	return &Bundle{Dir: dir, Manifest: m}, nil
}

// Tensor reads one tensor referenced by this bundle's manifest.
func (b *Bundle) Tensor(ref TensorRef) ([]float64, error) {
	return ReadTensorFile(b.Dir, ref)
}

// toFloat64 widens a ztensor tensor's data for serialization.
func toFloat64[T tensor.Float](data []T) []float64 {
	out := make([]float64, len(data))
	for i, v := range data {
		out[i] = float64(v)
	}
	return out
}

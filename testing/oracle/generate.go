package oracle

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/gradcheck"
	"github.com/zerfoo/ztensor/types"
)

// Skip records one registry op the generator could not bundle.
type Skip struct {
	Op     string `json:"op"`
	Reason string `json:"reason"`
}

// Summary reports one generation run.
type Summary struct {
	// Written lists the op names that produced bundles, in registry order.
	Written []string `json:"written"`
	// Skipped lists ops with no clean torch equivalent, with reasons.
	Skipped []Skip `json:"skipped"`
}

// GenerateAll runs every gradcheck registry op forward+backward on the
// float32 CPU engine with deterministic seeded inputs and upstream gradients,
// and writes one case bundle per op under dir (first cut of T1.3; the
// GPU-engine variant runs on the DGX later through the same format). Ops
// without a clean torch equivalent are recorded in Summary.Skipped. A
// generation.json summary is written alongside the bundles.
func GenerateAll(ctx context.Context, dir string) (*Summary, error) {
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, fmt.Errorf("oracle: creating %s: %w", dir, err)
	}
	sum := &Summary{}
	for _, op := range gradcheck.Registry() {
		mapping, ok := torchMap[op.Name]
		if !ok {
			sum.Skipped = append(sum.Skipped, Skip{Op: op.Name, Reason: "no entry in the op->torch mapping table"})
			continue
		}
		if mapping.SkipReason != "" {
			sum.Skipped = append(sum.Skipped, Skip{Op: op.Name, Reason: mapping.SkipReason})
			continue
		}
		if err := generateOne(ctx, dir, op, mapping.Expr); err != nil {
			return nil, fmt.Errorf("oracle: generating bundle for %s: %w", op.Name, err)
		}
		sum.Written = append(sum.Written, op.Name)
	}
	if err := writeSummary(dir, sum); err != nil {
		return nil, err
	}
	return sum, nil
}

func writeSummary(dir string, sum *Summary) error {
	b, err := json.MarshalIndent(sum, "", "  ")
	if err != nil {
		return fmt.Errorf("oracle: marshaling generation summary: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "generation.json"), append(b, '\n'), 0o600); err != nil {
		return fmt.Errorf("oracle: writing generation summary: %w", err)
	}
	return nil
}

// generateOne runs one registry op at f32 and writes its bundle.
func generateOne(ctx context.Context, root string, op gradcheck.OpInfo, expr string) error {
	dir := filepath.Join(root, op.Name)
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return err
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node, err := gradcheck.NewRegistryNode[float32](op.Name, engine)
	if err != nil {
		return err
	}

	// Inputs: the registry's deterministic float64 sampling, rounded to f32.
	// The recorded f32 bytes are canonical -- ztensor and torch both consume
	// exactly these values, so the f64->f32 rounding is not a diff source.
	inputs64, err := op.SampleInputs()
	if err != nil {
		return err
	}
	inputs := make([]*tensor.TensorNumeric[float32], len(inputs64))
	manifest := Manifest{
		FormatVersion: FormatVersion,
		Op:            op.Name,
		TorchExpr:     expr,
		DType:         DTypeFloat32,
		Seed:          op.Seed,
		Tolerance:     toleranceFor(op.Name),
	}
	for i, in64 := range inputs64 {
		data := make([]float32, len(in64.Data()))
		for k, v := range in64.Data() {
			data[k] = float32(v)
		}
		t, err := tensor.New[float32](in64.Shape(), data)
		if err != nil {
			return err
		}
		inputs[i] = t
		ref := TensorRef{
			Name:  fmt.Sprintf("x%d", i),
			File:  fmt.Sprintf("input_%d.bin", i),
			Shape: in64.Shape(),
			DType: DTypeFloat32,
		}
		if err := WriteTensorFile(filepath.Join(dir, ref.File), ref.DType, toFloat64(t.Data())); err != nil {
			return err
		}
		manifest.Inputs = append(manifest.Inputs, ref)
	}

	// Forward.
	y, err := node.Forward(ctx, inputs...)
	if err != nil {
		return err
	}

	// Upstream gradient: deterministic, +-[0.25, 1.0], from the op seed.
	upstream, err := seededUpstream(y.Shape(), op.Seed)
	if err != nil {
		return err
	}
	manifest.Upstream = TensorRef{Name: "upstream", File: "upstream.bin", Shape: y.Shape(), DType: DTypeFloat32}
	if err := WriteTensorFile(filepath.Join(dir, manifest.Upstream.File), DTypeFloat32, toFloat64(upstream.Data())); err != nil {
		return err
	}

	manifest.Forward = TensorRef{Name: "forward", File: "forward.bin", Shape: y.Shape(), DType: DTypeFloat32}
	if err := WriteTensorFile(filepath.Join(dir, manifest.Forward.File), DTypeFloat32, toFloat64(y.Data())); err != nil {
		return err
	}

	// Backward.
	grads, err := node.Backward(ctx, types.FullBackprop, upstream, inputs...)
	if err != nil {
		return err
	}
	if len(grads) != len(inputs) {
		return fmt.Errorf("%s: Backward returned %d gradients for %d inputs", op.Name, len(grads), len(inputs))
	}
	for i, g := range grads {
		if g == nil {
			return fmt.Errorf("%s: nil gradient for input %d", op.Name, i)
		}
		ref := TensorRef{
			Name:  fmt.Sprintf("x%d", i),
			File:  fmt.Sprintf("grad_input_%d.bin", i),
			Shape: g.Shape(),
			DType: DTypeFloat32,
		}
		if err := WriteTensorFile(filepath.Join(dir, ref.File), ref.DType, toFloat64(g.Data())); err != nil {
			return err
		}
		manifest.InputGrads = append(manifest.InputGrads, ref)
	}

	// Parameters (values bound by name in the torch expression; gradients
	// read from Parameter.Gradient after Backward, the ztensor convention).
	for _, p := range node.Parameters() {
		if p.Gradient == nil {
			return fmt.Errorf("%s: parameter %q has nil gradient after Backward", op.Name, p.Name)
		}
		ref := ParamRef{
			TensorRef: TensorRef{
				Name:  p.Name,
				File:  fmt.Sprintf("param_%s.bin", p.Name),
				Shape: p.Value.Shape(),
				DType: DTypeFloat32,
			},
			GradFile: fmt.Sprintf("grad_param_%s.bin", p.Name),
		}
		if err := WriteTensorFile(filepath.Join(dir, ref.File), ref.DType, toFloat64(p.Value.Data())); err != nil {
			return err
		}
		if err := WriteTensorFile(filepath.Join(dir, ref.GradFile), ref.DType, toFloat64(p.Gradient.Data())); err != nil {
			return err
		}
		manifest.Params = append(manifest.Params, ref)
	}

	return WriteBundle(dir, &manifest)
}

// seededUpstream builds a deterministic upstream gradient with entries of
// magnitude in [0.25, 1.0] and pseudo-random sign (the gradcheck convention:
// a randomized upstream catches structural Jacobian errors an all-ones
// upstream can mask). The values are recorded in the bundle, so torch
// backprops exactly these bytes.
func seededUpstream(shape []int, seed int64) (*tensor.TensorNumeric[float32], error) {
	// math/rand (not crypto) is intentional: deterministic test data.
	r := rand.New(rand.NewSource(seed)) //nolint:gosec
	data := make([]float32, shapeSize(shape))
	for i := range data {
		v := 0.25 + 0.75*r.Float64()
		if r.Float64() < 0.5 {
			v = -v
		}
		data[i] = float32(v)
	}
	return tensor.New[float32](shape, data)
}

// MappedOps returns the sorted op names present in the mapping table
// (including skipped entries); used by tests to keep the table and the
// gradcheck registry in lockstep.
func MappedOps() []string {
	names := make([]string, 0, len(torchMap))
	for name := range torchMap {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

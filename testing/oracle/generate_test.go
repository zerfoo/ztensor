package oracle

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/ztensor/testing/gradcheck"
)

// TestMappingCoversRegistry keeps the op->torch mapping table in lockstep
// with the gradcheck registry: every registry op must be mapped (or
// explicitly skipped with a reason), and the table must not carry entries for
// ops that left the registry.
func TestMappingCoversRegistry(t *testing.T) {
	registry := map[string]bool{}
	for _, op := range gradcheck.Registry() {
		registry[op.Name] = true
		m, ok := torchMap[op.Name]
		if !ok {
			t.Errorf("registry op %q has no entry in the op->torch mapping table (add an Expr or a SkipReason)", op.Name)
			continue
		}
		if m.Expr == "" && m.SkipReason == "" {
			t.Errorf("mapping for %q has neither Expr nor SkipReason", op.Name)
		}
		if m.Expr != "" && m.SkipReason != "" {
			t.Errorf("mapping for %q has both Expr and SkipReason", op.Name)
		}
	}
	for _, name := range MappedOps() {
		if !registry[name] {
			t.Errorf("mapping table entry %q is not in the gradcheck registry", name)
		}
	}
}

// TestGenerateAll generates bundles for the full registry and validates the
// result end-to-end through the reader: counts, manifests, file presence,
// shape consistency, and the recorded forward output of a trivially checkable
// op (Add).
func TestGenerateAll(t *testing.T) {
	dir := t.TempDir()
	sum, err := GenerateAll(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}

	wantTotal := len(gradcheck.Registry())
	if got := len(sum.Written) + len(sum.Skipped); got != wantTotal {
		t.Fatalf("summary covers %d ops, registry has %d", got, wantTotal)
	}
	if len(sum.Skipped) != 1 || sum.Skipped[0].Op != "HadamardTransform" {
		t.Fatalf("skipped = %+v, want exactly HadamardTransform", sum.Skipped)
	}
	if _, err := os.Stat(filepath.Join(dir, "generation.json")); err != nil {
		t.Fatalf("generation.json missing: %v", err)
	}

	for _, opName := range sum.Written {
		b, err := ReadBundle(filepath.Join(dir, opName))
		if err != nil {
			t.Fatalf("%s: %v", opName, err)
		}
		m := b.Manifest
		if m.Op != opName || m.TorchExpr == "" || m.DType != DTypeFloat32 {
			t.Fatalf("%s: bad manifest: %+v", opName, m)
		}
		if len(m.InputGrads) != len(m.Inputs) {
			t.Fatalf("%s: %d input grads for %d inputs", opName, len(m.InputGrads), len(m.Inputs))
		}
		// Every referenced tensor must read back consistent with its shape.
		refs := append([]TensorRef{m.Upstream, m.Forward}, m.Inputs...)
		refs = append(refs, m.InputGrads...)
		for _, p := range m.Params {
			refs = append(refs, p.TensorRef,
				TensorRef{Name: p.Name, File: p.GradFile, Shape: p.Shape, DType: p.DType})
		}
		for _, ref := range refs {
			if _, err := b.Tensor(ref); err != nil {
				t.Fatalf("%s: %v", opName, err)
			}
		}
		// Upstream must match the forward output shape.
		if shapeSize(m.Upstream.Shape) != shapeSize(m.Forward.Shape) {
			t.Fatalf("%s: upstream shape %v vs forward shape %v", opName, m.Upstream.Shape, m.Forward.Shape)
		}
	}

	// LayerNorm is the parameterized case: gamma/beta values + grads.
	ln, err := ReadBundle(filepath.Join(dir, "LayerNorm"))
	if err != nil {
		t.Fatal(err)
	}
	if len(ln.Manifest.Params) != 2 {
		t.Fatalf("LayerNorm has %d params, want gamma+beta", len(ln.Manifest.Params))
	}

	// Spot-check semantics: Add's recorded forward equals x0+x1 exactly.
	add, err := ReadBundle(filepath.Join(dir, "Add"))
	if err != nil {
		t.Fatal(err)
	}
	x0, err := add.Tensor(add.Manifest.Inputs[0])
	if err != nil {
		t.Fatal(err)
	}
	x1, err := add.Tensor(add.Manifest.Inputs[1])
	if err != nil {
		t.Fatal(err)
	}
	fwd, err := add.Tensor(add.Manifest.Forward)
	if err != nil {
		t.Fatal(err)
	}
	for i := range fwd {
		want := float64(float32(float32(x0[i]) + float32(x1[i])))
		if fwd[i] != want {
			t.Fatalf("Add forward[%d] = %v, want %v", i, fwd[i], want)
		}
	}

	// Determinism: a second generation must be byte-identical (the DGX replay
	// compares against exactly these bytes).
	dir2 := t.TempDir()
	if _, err := GenerateAll(context.Background(), dir2); err != nil {
		t.Fatal(err)
	}
	a, err := os.ReadFile(filepath.Join(dir, "Tanh", "forward.bin"))
	if err != nil {
		t.Fatal(err)
	}
	c, err := os.ReadFile(filepath.Join(dir2, "Tanh", "forward.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if string(a) != string(c) {
		t.Fatal("two generations produced different Tanh forward bytes; generation is not deterministic")
	}
}

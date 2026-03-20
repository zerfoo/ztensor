package graph

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// paramNode is a test node that holds named parameters.
type paramNode struct {
	NoParameters[float32]
	name   string
	params []*Parameter[float32]
}

func (n *paramNode) OpType() string                     { return "ParamNode" }
func (n *paramNode) Attributes() map[string]interface{} { return nil }
func (n *paramNode) OutputShape() []int                 { return nil }
func (n *paramNode) Parameters() []*Parameter[float32]  { return n.params }

func (n *paramNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return inputs[0], nil
}

func (n *paramNode) Backward(_ context.Context, _ types.BackwardMode, grad *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return []*tensor.TensorNumeric[float32]{grad}, nil
}

// buildParamGraph creates a graph with two named parameter nodes for testing.
func buildParamGraph(t *testing.T, w1Data, w2Data []float32) *Graph[float32] {
	t.Helper()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := NewBuilder[float32](engine)
	in := b.Input([]int{2})

	v1, err := tensor.New[float32]([]int{2}, w1Data)
	if err != nil {
		t.Fatal(err)
	}
	v2, err := tensor.New[float32]([]int{3}, w2Data)
	if err != nil {
		t.Fatal(err)
	}

	n1 := &paramNode{
		name:   "layer1",
		params: []*Parameter[float32]{{Name: "weight_a", Value: v1}},
	}
	n2 := &paramNode{
		name:   "layer2",
		params: []*Parameter[float32]{{Name: "weight_b", Value: v2}},
	}
	b.AddNode(n1, in)
	b.AddNode(n2, n1)

	g, err := b.Build(n2)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestGraphSaveLoad(t *testing.T) {
	w1 := []float32{1.5, 2.5}
	w2 := []float32{3.0, 4.0, 5.0}
	g1 := buildParamGraph(t, w1, w2)

	dir := t.TempDir()
	path := filepath.Join(dir, "params.json")

	if err := g1.SaveParameters(path); err != nil {
		t.Fatalf("SaveParameters: %v", err)
	}

	// Build a new graph with different initial values, then load.
	g2 := buildParamGraph(t, []float32{0, 0}, []float32{0, 0, 0})
	if err := g2.LoadParametersFromFile(path); err != nil {
		t.Fatalf("LoadParametersFromFile: %v", err)
	}

	// Verify loaded values match original.
	for _, p := range g2.Parameters() {
		var expected []float32
		switch p.Name {
		case "weight_a":
			expected = w1
		case "weight_b":
			expected = w2
		default:
			t.Fatalf("unexpected parameter %q", p.Name)
		}
		got := p.Value.Data()
		if len(got) != len(expected) {
			t.Fatalf("%s: length mismatch: got %d, want %d", p.Name, len(got), len(expected))
		}
		for i := range got {
			if math.Abs(float64(got[i]-expected[i])) > 1e-6 {
				t.Errorf("%s[%d] = %g, want %g", p.Name, i, got[i], expected[i])
			}
		}
	}
}

func TestCheckpointRoundTrip(t *testing.T) {
	w1 := []float32{1.0, 2.0}
	w2 := []float32{3.0, 4.0, 5.0}
	g1 := buildParamGraph(t, w1, w2)

	dir := t.TempDir()
	path := filepath.Join(dir, "checkpoint.json")

	optState := map[string]interface{}{
		"learning_rate": 0.001,
		"step":          float64(42),
		"momentum":      []float64{0.1, 0.2},
	}

	if err := g1.SaveCheckpoint(path, optState); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}

	g2 := buildParamGraph(t, []float32{0, 0}, []float32{0, 0, 0})
	gotOpt, err := g2.LoadCheckpoint(path)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}

	// Verify parameters restored.
	for _, p := range g2.Parameters() {
		var expected []float32
		switch p.Name {
		case "weight_a":
			expected = w1
		case "weight_b":
			expected = w2
		default:
			t.Fatalf("unexpected parameter %q", p.Name)
		}
		got := p.Value.Data()
		for i := range got {
			if math.Abs(float64(got[i]-expected[i])) > 1e-6 {
				t.Errorf("%s[%d] = %g, want %g", p.Name, i, got[i], expected[i])
			}
		}
	}

	// Verify optimizer state restored.
	if lr, ok := gotOpt["learning_rate"].(float64); !ok || math.Abs(lr-0.001) > 1e-9 {
		t.Errorf("learning_rate = %v, want 0.001", gotOpt["learning_rate"])
	}
	if step, ok := gotOpt["step"].(float64); !ok || step != 42 {
		t.Errorf("step = %v, want 42", gotOpt["step"])
	}
	if mom, ok := gotOpt["momentum"].([]interface{}); !ok || len(mom) != 2 {
		t.Errorf("momentum = %v, want [0.1 0.2]", gotOpt["momentum"])
	}
}

func TestSaveLoadFileNotFound(t *testing.T) {
	g := buildParamGraph(t, []float32{1, 2}, []float32{3, 4, 5})

	err := g.LoadParametersFromFile("/nonexistent/path/params.json")
	if err == nil {
		t.Fatal("expected error for missing file, got nil")
	}

	_, err = g.LoadCheckpoint("/nonexistent/path/checkpoint.json")
	if err == nil {
		t.Fatal("expected error for missing checkpoint, got nil")
	}
}

func TestSaveLoadCorruptedFile(t *testing.T) {
	g := buildParamGraph(t, []float32{1, 2}, []float32{3, 4, 5})
	dir := t.TempDir()

	// Write invalid JSON.
	corruptPath := filepath.Join(dir, "corrupt.json")
	if err := os.WriteFile(corruptPath, []byte("{not valid json"), 0644); err != nil {
		t.Fatal(err)
	}

	if err := g.LoadParametersFromFile(corruptPath); err == nil {
		t.Fatal("expected error for corrupted params file, got nil")
	}

	if _, err := g.LoadCheckpoint(corruptPath); err == nil {
		t.Fatal("expected error for corrupted checkpoint file, got nil")
	}
}

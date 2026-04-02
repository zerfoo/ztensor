package stablehlo

import (
	"slices"
	"testing"
)

func TestInferShapeSameShapeArithmetic(t *testing.T) {
	for _, op := range []string{"Add", "Sub", "Mul", "Div"} {
		got, err := InferShape(op, [][]int{{2, 3}, {2, 3}}, nil)
		if err != nil {
			t.Fatalf("%s same-shape: %v", op, err)
		}
		if !slices.Equal(got, []int{2, 3}) {
			t.Errorf("%s same-shape: got %v, want [2 3]", op, got)
		}
	}
}

func TestInferShapeBroadcast2D(t *testing.T) {
	// {2,3} + {1,3} -> {2,3}
	got, err := InferShape("Add", [][]int{{2, 3}, {1, 3}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 3}) {
		t.Errorf("got %v, want [2 3]", got)
	}
}

func TestInferShapeBroadcast3D(t *testing.T) {
	// {4,1,3} + {1,5,3} -> {4,5,3}
	got, err := InferShape("Add", [][]int{{4, 1, 3}, {1, 5, 3}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{4, 5, 3}) {
		t.Errorf("got %v, want [4 5 3]", got)
	}
}

func TestInferShapeBroadcastRankMismatch(t *testing.T) {
	// {3} + {2,3} -> {2,3} (lower-rank tensor is left-padded with 1s)
	got, err := InferShape("Mul", [][]int{{3}, {2, 3}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 3}) {
		t.Errorf("got %v, want [2 3]", got)
	}
}

func TestInferShapeScalarOps(t *testing.T) {
	for _, op := range []string{"MulScalar", "DivScalar", "AddScalar"} {
		got, err := InferShape(op, [][]int{{4, 5}}, nil)
		if err != nil {
			t.Fatalf("%s: %v", op, err)
		}
		if !slices.Equal(got, []int{4, 5}) {
			t.Errorf("%s: got %v, want [4 5]", op, got)
		}
	}
}

func TestInferShapeScalarOpsTwoInputs(t *testing.T) {
	// Scalar ops with explicit scalar shape as second input.
	got, err := InferShape("MulScalar", [][]int{{3, 4}, {}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{3, 4}) {
		t.Errorf("got %v, want [3 4]", got)
	}
}

func TestInferShapeUnaryOps(t *testing.T) {
	unary := []string{"Exp", "Log", "Sin", "Cos", "Tanh", "Sqrt", "Rsqrt", "Neg", "Abs"}
	for _, op := range unary {
		got, err := InferShape(op, [][]int{{2, 3, 4}}, nil)
		if err != nil {
			t.Fatalf("%s: %v", op, err)
		}
		if !slices.Equal(got, []int{2, 3, 4}) {
			t.Errorf("%s: got %v, want [2 3 4]", op, got)
		}
	}
}

func TestInferShapePow(t *testing.T) {
	got, err := InferShape("Pow", [][]int{{2, 3}, {2, 3}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 3}) {
		t.Errorf("got %v, want [2 3]", got)
	}
}

func TestInferShapePowBroadcast(t *testing.T) {
	// Pow supports broadcasting: {2,3} ** {1,3} -> {2,3}
	got, err := InferShape("Pow", [][]int{{2, 3}, {1, 3}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{2, 3}) {
		t.Errorf("got %v, want [2 3]", got)
	}
}

func TestInferShapeIncompatibleShapes(t *testing.T) {
	_, err := InferShape("Add", [][]int{{2, 3}, {4, 3}}, nil)
	if err == nil {
		t.Fatal("expected error for incompatible shapes {2,3} and {4,3}")
	}
}

func TestInferShapeIncompatibleShapes3D(t *testing.T) {
	_, err := InferShape("Mul", [][]int{{2, 5, 3}, {2, 4, 3}}, nil)
	if err == nil {
		t.Fatal("expected error for incompatible shapes {2,5,3} and {2,4,3}")
	}
}

func TestInferShapeWrongInputCount(t *testing.T) {
	// Binary op with one input.
	_, err := InferShape("Add", [][]int{{2, 3}}, nil)
	if err == nil {
		t.Fatal("expected error for Add with 1 input")
	}

	// Unary op with two inputs.
	_, err = InferShape("Exp", [][]int{{2, 3}, {2, 3}}, nil)
	if err == nil {
		t.Fatal("expected error for Exp with 2 inputs")
	}
}

func TestInferShapeUnsupportedOp(t *testing.T) {
	_, err := InferShape("FooBarBaz", [][]int{{2, 3}}, nil)
	if err == nil {
		t.Fatal("expected error for unsupported op")
	}
}

func TestInferShapeScalarInputs(t *testing.T) {
	// Two scalar (rank-0) inputs.
	got, err := InferShape("Add", [][]int{{}, {}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("got %v, want [] (scalar)", got)
	}
}

func TestInferShapeOutputNotAliased(t *testing.T) {
	// Verify the returned slice is a copy, not the original input.
	input := [][]int{{5, 6}}
	got, err := InferShape("Neg", input, nil)
	if err != nil {
		t.Fatal(err)
	}
	got[0] = 999
	if input[0][0] != 5 {
		t.Error("InferShape returned a slice that aliases the input")
	}
}

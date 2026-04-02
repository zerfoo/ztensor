package stablehlo

import (
	"strings"
	"testing"
)

func TestEmitBinaryElementwise(t *testing.T) {
	tests := []struct {
		name   string
		op     string
		lhs    string
		rhs    string
		shape  []int
		dtype  string
		wantOp string
	}{
		{"add 2D", OpAdd, "%v0", "%v1", []int{2, 3}, DTypeF32, "stablehlo.add"},
		{"sub 1D", OpSubtract, "%a", "%b", []int{8}, DTypeF64, "stablehlo.subtract"},
		{"mul 3D", OpMultiply, "%x", "%y", []int{1, 4, 4}, DTypeBF16, "stablehlo.multiply"},
		{"div scalar", OpDivide, "%p", "%q", []int{}, DTypeF32, "stablehlo.divide"},
		{"pow 2D", OpPower, "%a", "%b", []int{3, 3}, DTypeF32, "stablehlo.power"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := NewEmitter()
			mlir, out := e.EmitBinaryElementwise(tt.op, tt.lhs, tt.rhs, tt.shape, tt.dtype)
			wantTy := FormatTensorType(tt.shape, tt.dtype)
			wantLine := out + " = " + tt.wantOp + " " + tt.lhs + ", " + tt.rhs + " : " + wantTy
			if mlir != wantLine {
				t.Errorf("got:\n  %s\nwant:\n  %s", mlir, wantLine)
			}
			if out != "%v0" {
				t.Errorf("output name = %q, want %%v0", out)
			}
		})
	}
}

func TestEmitAdd(t *testing.T) {
	e := NewEmitter()
	mlir, out := e.EmitAdd("%arg0", "%arg1", []int{2, 3}, DTypeF32)
	want := "%v0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>"
	if mlir != want {
		t.Errorf("EmitAdd:\n  got:  %s\n  want: %s", mlir, want)
	}
	if out != "%v0" {
		t.Errorf("output = %q, want %%v0", out)
	}
}

func TestEmitUnaryOps(t *testing.T) {
	tests := []struct {
		name   string
		emit   func(*Emitter, string, []int, string) (string, string)
		wantOp string
	}{
		{"Exp", (*Emitter).EmitExp, "stablehlo.exponential"},
		{"Log", (*Emitter).EmitLog, "stablehlo.log"},
		{"Sin", (*Emitter).EmitSin, "stablehlo.sine"},
		{"Cos", (*Emitter).EmitCos, "stablehlo.cosine"},
		{"Tanh", (*Emitter).EmitTanh, "stablehlo.tanh"},
		{"Sqrt", (*Emitter).EmitSqrt, "stablehlo.sqrt"},
		{"Rsqrt", (*Emitter).EmitRsqrt, "stablehlo.rsqrt"},
		{"Neg", (*Emitter).EmitNeg, "stablehlo.negate"},
	}
	shape := []int{4, 8}
	dtype := DTypeF32
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := NewEmitter()
			mlir, out := tt.emit(e, "%input", shape, dtype)
			wantTy := FormatTensorType(shape, dtype)
			want := out + " = " + tt.wantOp + " %input : " + wantTy
			if mlir != want {
				t.Errorf("got:\n  %s\nwant:\n  %s", mlir, want)
			}
			if out != "%v0" {
				t.Errorf("output = %q, want %%v0", out)
			}
		})
	}
}

func TestEmitScalarOps(t *testing.T) {
	tests := []struct {
		name   string
		emit   func(*Emitter, string, float64, []int, string) (string, string)
		wantOp string
	}{
		{"MulScalar", (*Emitter).EmitMulScalar, "stablehlo.multiply"},
		{"AddScalar", (*Emitter).EmitAddScalar, "stablehlo.add"},
		{"DivScalar", (*Emitter).EmitDivScalar, "stablehlo.divide"},
	}
	shape := []int{2, 3}
	dtype := DTypeF32
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := NewEmitter()
			mlir, out := tt.emit(e, "%x", 2.5, shape, dtype)

			lines := strings.Split(mlir, "\n")
			if len(lines) != 3 {
				t.Fatalf("expected 3 lines, got %d:\n%s", len(lines), mlir)
			}

			// Line 1: constant
			if !strings.Contains(lines[0], OpConstant) {
				t.Errorf("line 0 missing %s: %s", OpConstant, lines[0])
			}
			if !strings.Contains(lines[0], "dense<2.5>") {
				t.Errorf("line 0 missing dense<2.5>: %s", lines[0])
			}
			if !strings.Contains(lines[0], "tensor<f32>") {
				t.Errorf("line 0 missing scalar type: %s", lines[0])
			}

			// Line 2: broadcast_in_dim
			if !strings.Contains(lines[1], OpBroadcastIn) {
				t.Errorf("line 1 missing %s: %s", OpBroadcastIn, lines[1])
			}
			if !strings.Contains(lines[1], "tensor<2x3xf32>") {
				t.Errorf("line 1 missing output type: %s", lines[1])
			}

			// Line 3: element-wise op
			if !strings.Contains(lines[2], tt.wantOp) {
				t.Errorf("line 2 missing %s: %s", tt.wantOp, lines[2])
			}

			if out != "%v2" {
				t.Errorf("output = %q, want %%v2 (const=%%v0, bcast=%%v1, op=%%v2)", out)
			}
		})
	}
}

func TestEmitScalarOpFullOutput(t *testing.T) {
	e := NewEmitter()
	mlir, out := e.EmitMulScalar("%arg0", 3, []int{4}, DTypeF32)
	want := "%v0 = stablehlo.constant dense<3> : tensor<f32>\n" +
		"%v1 = stablehlo.broadcast_in_dim %v0, dims = [] : (tensor<f32>) -> tensor<4xf32>\n" +
		"%v2 = stablehlo.multiply %arg0, %v1 : tensor<4xf32>"
	if mlir != want {
		t.Errorf("EmitMulScalar full output:\n  got:\n%s\n  want:\n%s", mlir, want)
	}
	if out != "%v2" {
		t.Errorf("output = %q, want %%v2", out)
	}
}

func TestEmitOpDispatch(t *testing.T) {
	shape := []int{2, 4}
	dtype := DTypeF32

	tests := []struct {
		name    string
		opName  string
		inputs  []string
		attrs   map[string]any
		wantOp  string
		wantErr bool
	}{
		{"Add", "Add", []string{"%a", "%b"}, nil, "stablehlo.add", false},
		{"Sub", "Sub", []string{"%a", "%b"}, nil, "stablehlo.subtract", false},
		{"Mul", "Mul", []string{"%a", "%b"}, nil, "stablehlo.multiply", false},
		{"Div", "Div", []string{"%a", "%b"}, nil, "stablehlo.divide", false},
		{"Pow", "Pow", []string{"%a", "%b"}, nil, "stablehlo.power", false},
		{"Exp", "Exp", []string{"%a"}, nil, "stablehlo.exponential", false},
		{"Log", "Log", []string{"%a"}, nil, "stablehlo.log", false},
		{"Sin", "Sin", []string{"%a"}, nil, "stablehlo.sine", false},
		{"Cos", "Cos", []string{"%a"}, nil, "stablehlo.cosine", false},
		{"Tanh", "Tanh", []string{"%a"}, nil, "stablehlo.tanh", false},
		{"Sqrt", "Sqrt", []string{"%a"}, nil, "stablehlo.sqrt", false},
		{"Rsqrt", "Rsqrt", []string{"%a"}, nil, "stablehlo.rsqrt", false},
		{"Neg", "Neg", []string{"%a"}, nil, "stablehlo.negate", false},
		{"MulScalar", "MulScalar", []string{"%a"}, map[string]any{"scalar": 2.0}, "stablehlo.multiply", false},
		{"AddScalar", "AddScalar", []string{"%a"}, map[string]any{"scalar": 1.0}, "stablehlo.add", false},
		{"DivScalar", "DivScalar", []string{"%a"}, map[string]any{"scalar": 4.0}, "stablehlo.divide", false},
		{"unsupported", "Softmax", []string{"%a"}, nil, "", true},
		{"wrong inputs binary", "Add", []string{"%a"}, nil, "", true},
		{"wrong inputs unary", "Exp", []string{"%a", "%b"}, nil, "", true},
		{"missing scalar attr", "MulScalar", []string{"%a"}, nil, "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := NewEmitter()
			mlir, _, err := e.EmitOp(tt.opName, tt.inputs, shape, dtype, tt.attrs)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error, got mlir: %s", mlir)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !strings.Contains(mlir, tt.wantOp) {
				t.Errorf("EmitOp(%s) output missing %s:\n%s", tt.opName, tt.wantOp, mlir)
			}
		})
	}
}

func TestEmitSSACounterProgresses(t *testing.T) {
	e := NewEmitter()
	_, out1 := e.EmitAdd("%a", "%b", []int{2}, DTypeF32)
	_, out2 := e.EmitExp("%c", []int{2}, DTypeF32)
	_, out3 := e.EmitSub("%d", "%e", []int{2}, DTypeF32)

	if out1 != "%v0" || out2 != "%v1" || out3 != "%v2" {
		t.Errorf("SSA names = [%s, %s, %s], want [%%v0, %%v1, %%v2]", out1, out2, out3)
	}

	if e.Namer.Count() != 3 {
		t.Errorf("namer count = %d, want 3", e.Namer.Count())
	}
}

func TestEmitScalarOpSSACounterProgresses(t *testing.T) {
	e := NewEmitter()
	// MulScalar uses 3 SSA names (const, broadcast, op).
	_, out1 := e.EmitMulScalar("%x", 2.0, []int{4}, DTypeF32)
	// Next op should get %v3.
	_, out2 := e.EmitAdd("%a", "%b", []int{4}, DTypeF32)

	if out1 != "%v2" {
		t.Errorf("MulScalar output = %q, want %%v2", out1)
	}
	if out2 != "%v3" {
		t.Errorf("Add output after scalar = %q, want %%v3", out2)
	}
}

func TestEmitDifferentDtypes(t *testing.T) {
	dtypes := []string{DTypeF32, DTypeF64, DTypeF16, DTypeBF16}
	for _, dtype := range dtypes {
		t.Run(dtype, func(t *testing.T) {
			e := NewEmitter()
			mlir, _ := e.EmitAdd("%a", "%b", []int{2, 3}, dtype)
			wantTy := FormatTensorType([]int{2, 3}, dtype)
			if !strings.HasSuffix(mlir, wantTy) {
				t.Errorf("EmitAdd with dtype %s: %s does not end with %s", dtype, mlir, wantTy)
			}
		})
	}
}

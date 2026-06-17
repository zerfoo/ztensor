package oracle

// torchOp describes how to replay one gradcheck registry op in PyTorch, or
// why it cannot be replayed cleanly.
//
// The constructor arguments embedded in the expressions (leaky-relu slope
// 0.1, scalars 0.7/-1.3, reshape target (3, 2), reduction axis 1, layernorm
// width 4 and eps 1e-05) MUST match gradcheck.NewRegistryNode, which is the
// single Go-side source of truth for those arguments. The generator test
// (generate_test.go) cross-checks the mapping table against the registry so
// an op added to the registry without a mapping fails CI.
type torchOp struct {
	// Expr is a Python expression over `torch`, the inputs bound as x0..xn,
	// and the parameters bound by name, evaluating to the forward output.
	Expr string
	// SkipReason, when non-empty, marks the op as having no clean torch
	// equivalent; the generator records it in the SKIPPED list instead of
	// emitting a bundle.
	SkipReason string
}

// torchMap maps gradcheck registry op names to their torch replay.
var torchMap = map[string]torchOp{
	// Elementwise binary.
	"Add": {Expr: "x0 + x1"},
	"Sub": {Expr: "x0 - x1"},
	"Mul": {Expr: "x0 * x1"},
	"Div": {Expr: "x0 / x1"},
	"Pow": {Expr: "torch.pow(x0, x1)"},

	// Elementwise unary / activations.
	"Tanh":      {Expr: "torch.tanh(x0)"},
	"Sigmoid":   {Expr: "torch.sigmoid(x0)"},
	"ReLU":      {Expr: "torch.relu(x0)"},
	"LeakyReLU": {Expr: "torch.nn.functional.leaky_relu(x0, 0.1)"},
	"Exp":       {Expr: "torch.exp(x0)"},
	"Log":       {Expr: "torch.log(x0)"},
	"Sqrt":      {Expr: "torch.sqrt(x0)"},
	"Rsqrt":     {Expr: "torch.rsqrt(x0)"},
	"Sin":       {Expr: "torch.sin(x0)"},
	"Cos":       {Expr: "torch.cos(x0)"},
	"AddScalar": {Expr: "x0 + 0.7"},
	"MulScalar": {Expr: "x0 * -1.3"},

	// MatMul-like and shape ops.
	"MatMul":    {Expr: "x0 @ x1"},
	"Transpose": {Expr: "x0.transpose(0, 1)"},
	"Reshape":   {Expr: "x0.reshape(3, 2)"},
	"HadamardTransform": {
		SkipReason: "torch has no built-in normalized Walsh-Hadamard transform; replaying it would mean hand-building the H matrix in the runner, i.e. testing our own reimplementation rather than torch",
	},

	// Softmax and reductions. ReduceMax uses amax; torch.amax splits the
	// gradient among tied maxima while ztensor routes it to the first argmax,
	// so the registry's continuous input sampling (no ties) is load-bearing.
	"Softmax":    {Expr: "torch.softmax(x0, 1)"},
	"ReduceSum":  {Expr: "x0.sum(dim=1, keepdim=True)"},
	"ReduceMean": {Expr: "x0.mean(dim=1, keepdim=True)"},
	"ReduceMax":  {Expr: "x0.amax(dim=1, keepdim=True)"},

	// LayerNorm: ztensor's wrapper keeps gamma/beta as (1, dim); the
	// expression reshapes the leaf inside the graph so torch records the
	// gradient on the (1, dim) leaf, matching the recorded ztensor shapes.
	"LayerNorm": {Expr: "torch.nn.functional.layer_norm(x0, (4,), weight=gamma.reshape(4), bias=beta.reshape(4), eps=1e-05)"},

	// GroupNorm: 2D input [N, C] with num_groups=2 over C=4 (groups of 2),
	// per-channel affine. Matches gradcheck.newGroupNormNode(e, 4, 2); gamma/beta
	// leaves stay (1, 4) and reshape to (4,) inside the graph, like LayerNorm.
	"GroupNorm": {Expr: "torch.nn.functional.group_norm(x0, 2, weight=gamma.reshape(4), bias=beta.reshape(4), eps=1e-05)"},

	// CrossAttention: single-head scaled dot-product attention over Q=x0, K=x1,
	// V=x2. torch's default scale is 1/sqrt(E) with E the query last dim, which
	// matches gradcheck.newCrossAttentionNode's 1/sqrt(d).
	"CrossAttention": {Expr: "torch.nn.functional.scaled_dot_product_attention(x0, x1, x2)"},

	// AdaLN: out = x*(1+scale)+shift with scale = c@Ws, shift = c@Wsh. x0=x,
	// x1=c; Ws,Wsh are the named [cond,dim] projection leaves.
	"AdaLN": {Expr: "x0 * (1 + x1 @ Ws) + (x1 @ Wsh)"},

	// TimestepEmbed: sinusoidal embedding concat(sin(t@freqs), cos(t@freqs)).
	// x0 = t [N,1]; freqs is the named [1,H] leaf.
	"TimestepEmbed": {Expr: "torch.cat([torch.sin(x0 @ freqs), torch.cos(x0 @ freqs)], dim=1)"},
}

// defaultTolerance is the first-cut f32 comparison bar: ztensor CPU/GPU f32
// vs torch f32 on the same inputs. Elementwise transcendentals agree to a few
// ULP; gradients of reductions/normalizations accumulate in different orders,
// hence the looser grad rtol.
var defaultTolerance = Tolerance{
	FwdAtol:  1e-6,
	FwdRtol:  1e-5,
	GradAtol: 1e-6,
	GradRtol: 1e-4,
}

// toleranceOverrides loosens specific ops whose reference implementations
// legitimately differ in accumulation order at f32.
var toleranceOverrides = map[string]Tolerance{
	"Softmax":   {FwdAtol: 1e-6, FwdRtol: 1e-4, GradAtol: 1e-6, GradRtol: 1e-3},
	"LayerNorm": {FwdAtol: 1e-5, FwdRtol: 1e-4, GradAtol: 1e-5, GradRtol: 1e-3},
	"GroupNorm":      {FwdAtol: 1e-5, FwdRtol: 1e-4, GradAtol: 1e-5, GradRtol: 1e-3},
	"MatMul":         {FwdAtol: 1e-6, FwdRtol: 1e-4, GradAtol: 1e-6, GradRtol: 1e-3},
	"CrossAttention": {FwdAtol: 1e-5, FwdRtol: 1e-4, GradAtol: 1e-5, GradRtol: 1e-3},
}

// toleranceFor returns the per-op tolerance, falling back to the default.
func toleranceFor(op string) Tolerance {
	if t, ok := toleranceOverrides[op]; ok {
		return t
	}
	return defaultTolerance
}

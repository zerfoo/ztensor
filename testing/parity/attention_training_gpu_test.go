package parity

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// T2.4 (zerfoo docs/plan-gpu-training-hardening.md, UC-GH-004/UC-GH-007):
// the Wolf-pattern integration stress test on an ATTENTION-SHAPED synthetic
// graph. TestTrainingLoop_WolfPattern_GPU (PR #140) proved the hazard
// schedule on a linear net; this test adds the graph shape that produced
// Wolf's actual corruption class:
//
//   - a single-head attention node (Q@K^T -> scale -> softmax -> @V) whose
//     big intermediates (Q, K, V, attention weights) live in arena memory
//     and are kept alive across forward->backward via the ADR 006
//     SaveForBackward contract (multi-consumer: the attention weights are
//     read three ways in backward -- dV, dA, and the softmax denominator
//     reduction);
//   - a residual + RMS-normalization node saving its pre-norm activation
//     and normalization inverse (the Wolf QK-norm cached-inverse shape);
//   - the input feeding two consumers, so backward converges gradients
//     through the graph's accumulation path.
//
// Schedule (the Wolf gr-12 hazard, sharpened to the reset-between-fwd-bwd
// parity schedule): per-sample Forward+Backward through a graph.Graph with
// an arena reset BETWEEN forward and backward (so every saved intermediate
// is load-bearing: anything backward reads must be host-owned, persistent,
// or pinned via SaveForBackward), gradient accumulation into PERSISTENT
// (non-arena) graph.Parameters with in-place engine ops, another ResetPool
// after every sample, in-place SGD step once per batch, >=2 batches x >=2
// epochs so arena reuse crosses sample, batch, and epoch boundaries -- all
// under ZTENSOR_ARENA_POISON semantics, so any read of a recycled span is a
// deterministic NaN.
//
// PASS requires: (1) every parameter and accumulated gradient finite,
// (2) GPU result within f32 tolerance of the byte-identical CPU-engine run,
// (3) persistent parameter/gradient storage never re-homed.

const (
	attnSeq     = 6
	attnDim     = 8
	attnEpochs  = 2
	attnBatches = 2
	attnSamples = 3
	attnLR      = float32(0.05)
	attnEps     = float32(1e-5)
)

// TestAttentionTraining_WolfPattern_GPU is the GB10 leg: real CUDA arena,
// poison on, deliberately small arena. Runs on the DGX via the Spark pod in
// scripts/parity/; skips without CUDA.
func TestAttentionTraining_WolfPattern_GPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	restorePoison := cuda.SetArenaPoisonEnabledForTesting(true)
	defer restorePoison()
	restoreArena := compute.SetArenaBytesForTesting(gpuParityArenaBytes)
	defer restoreArena()

	gpuEng, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	gpu := runAttentionTrainingLoop(t, gpuTrainSide(t, gpuEng))
	cpu := runAttentionTrainingLoop(t, cpuTrainSide(t))

	for i, name := range attnParamNames {
		assertFiniteAndClose(t, name, gpu[i], cpu[i])
	}
}

// TestAttentionTraining_WolfPattern_StressCI runs the identical loop in
// ordinary CI with the host-backed-arena StressEngine as the candidate (GPU
// lifetime semantics without a GPU) under poison, against the plain CPU
// reference -- catching lifetime regressions in the attention pattern before
// a GB10 run.
func TestAttentionTraining_WolfPattern_StressCI(t *testing.T) {
	enablePoison(t)
	stress := NewStressEngine(compute.NewCPUEngine[float32](numeric.Float32Ops{}), 1<<20)
	got := runAttentionTrainingLoop(t, trainSide{
		eng:        stress,
		reset:      stress.ResetArena,
		persistent: hostPersistent(t),
	})

	cpu := runAttentionTrainingLoop(t, cpuTrainSide(t))

	for i, name := range attnParamNames {
		assertFiniteAndClose(t, name, got[i], cpu[i])
	}

	// The save-for-backward sets must all have been released: a leaked pin
	// would permanently raise the arena's rewind floor.
	if pinned := stress.Arena().PinnedBytes(); pinned != 0 {
		t.Errorf("arena PinnedBytes after training = %d, want 0 (leaked save-for-backward pin)", pinned)
	}
}

// gpuTrainSide allocates persistent tensors as raw (non-arena) GPUStorage,
// exactly like the linear-net Wolf-pattern test.
func gpuTrainSide(t *testing.T, eng *compute.GPUEngine[float32]) trainSide {
	t.Helper()
	return trainSide{
		eng:   eng,
		reset: eng.ResetPool,
		persistent: func(shape []int, data []float32) *t32 {
			st, err := tensor.NewGPUStorageFromSlice(append([]float32(nil), data...))
			if err != nil {
				t.Fatalf("NewGPUStorageFromSlice: %v", err)
			}
			tt, err := tensor.NewWithStorage[float32](shape, st)
			if err != nil {
				t.Fatalf("NewWithStorage: %v", err)
			}
			return tt
		},
	}
}

var attnParamNames = []string{"Wq", "Wk", "Wv"}

// runAttentionTrainingLoop trains the attention+residual-norm graph with the
// Wolf hazard schedule and returns host copies of the final parameter values
// in attnParamNames order.
func runAttentionTrainingLoop(t *testing.T, side trainSide) [][]float32 {
	t.Helper()
	ctx := context.Background()
	eng := side.eng

	// Persistent parameters and gradient accumulators (non-arena storage;
	// ResetPool must never recycle them).
	params := make([]*graph.Parameter[float32], len(attnParamNames))
	seeds := []float32{0.31, 0.17, 0.59}
	for i, name := range attnParamNames {
		params[i] = &graph.Parameter[float32]{
			Name:     name,
			Value:    side.persistent([]int{attnDim, attnDim}, rampData(attnDim*attnDim, seeds[i])),
			Gradient: side.persistent([]int{attnDim, attnDim}, make([]float32, attnDim*attnDim)),
		}
	}

	// Record the persistent storages: nothing in the loop may re-home them.
	type homed struct {
		name string
		tt   *t32
		st   tensor.Storage[float32]
	}
	var homes []homed
	for _, p := range params {
		homes = append(homes,
			homed{p.Name + ".Value", p.Value, p.Value.GetStorage()},
			homed{p.Name + ".Gradient", p.Gradient, p.Gradient.GetStorage()},
		)
	}

	attn := &attentionNode{eng: eng, wq: params[0], wk: params[1], wv: params[2]}
	norm := &residualNormNode{eng: eng, eps: attnEps}

	b := graph.NewBuilder[float32](eng)
	in := b.Input([]int{attnSeq, attnDim})
	b.AddNode(attn, in)
	b.AddNode(norm, in, attn) // x is multi-consumer: attention AND residual
	g, err := b.Build(norm)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	for e := 0; e < attnEpochs; e++ {
		for bt := 0; bt < attnBatches; bt++ {
			for s := 0; s < attnSamples; s++ {
				step := e*attnBatches*attnSamples + bt*attnSamples + s
				x := mustNew(t, []int{attnSeq, attnDim}, rampData(attnSeq*attnDim, 0.41+float32(step)*0.07))
				target := rowOneHotData(attnSeq, attnDim, e+bt+s)

				y, err := g.Forward(ctx, x)
				if err != nil {
					t.Fatalf("epoch %d batch %d sample %d: Forward: %v", e, bt, s, err)
				}

				// Softmax cross-entropy head, materialized on the HOST
				// (caller-owned: the mid-step reset below must not be able
				// to poison the initial gradient): dy = softmax(y) - target.
				p := mustOp(t, "Softmax(y)")(eng.Softmax(ctx, y, 1))
				dyData := append([]float32(nil), p.Data()...)
				for i := range dyData {
					dyData[i] -= target[i]
				}
				dy := mustNew(t, []int{attnSeq, attnDim}, dyData)

				// The HARD hazard: reset (and poison) the arena BETWEEN
				// forward and backward. Backward may now only read
				// host-owned tensors, persistent parameters, and the
				// intermediates pinned via SaveForBackward -- any other
				// forward intermediate is poison.
				if side.reset != nil {
					side.reset()
				}

				if err := g.Backward(ctx, types.FullBackprop, dy); err != nil {
					t.Fatalf("epoch %d batch %d sample %d: Backward: %v", e, bt, s, err)
				}

				// Wolf's per-sample ResetPool: with the saved sets released
				// by Backward, this recycles (and poisons) everything.
				if side.reset != nil {
					side.reset()
				}
			}

			// Accumulated gradients must be finite BEFORE the step: a stale
			// arena read in backward lands here as poison NaN.
			for _, p := range params {
				assertAllFinite(t, fmt.Sprintf("epoch %d batch %d %s.Gradient", e, bt, p.Name), p.Gradient.Data())
			}

			// Optimizer step, once per batch: W -= lr * G, in place.
			for _, p := range params {
				sgdStepInPlace(t, ctx, eng, p.Value, p.Gradient, attnLR, p.Name)
				if err := eng.Zero(ctx, p.Gradient); err != nil {
					t.Fatalf("Zero(%s.Gradient): %v", p.Name, err)
				}
			}
			// Recycle the step's arena temporaries before the next batch.
			if side.reset != nil {
				side.reset()
			}
		}
	}

	// Persistent storage must never have been re-homed (zerfoo#850/#855).
	for _, h := range homes {
		if h.tt.GetStorage() != h.st {
			t.Errorf("persistent storage of %s was re-homed (%p -> %p)", h.name, h.st, h.tt.GetStorage())
		}
	}

	out := make([][]float32, len(params))
	for i, p := range params {
		out[i] = append([]float32(nil), p.Value.Data()...)
	}
	return out
}

// assertAllFinite fails if any value is NaN/Inf (the poison signature).
func assertAllFinite(t *testing.T, name string, vals []float32) {
	t.Helper()
	for i, v := range vals {
		if f := float64(v); math.IsNaN(f) || math.IsInf(f, 0) {
			t.Fatalf("%s[%d] = %v: non-finite under poison (stale arena read)", name, i, v)
		}
	}
}

// rowOneHotData builds [rows, n] one-hot target data with row r hot at
// (hotBase+r) % n.
func rowOneHotData(rows, n, hotBase int) []float32 {
	data := make([]float32, rows*n)
	for r := 0; r < rows; r++ {
		data[r*n+(hotBase+r)%n] = 1
	}
	return data
}

// ---------------------------------------------------------------------------
// Graph nodes
// ---------------------------------------------------------------------------

// errCollect makes engine-op chains readable inside node code: it records
// the first error and lets the math read sequentially. Callers must check
// err() between dependent stages.
type errCollect struct{ e error }

func (c *errCollect) t(t *t32, err error) *t32 {
	if c.e == nil && err != nil {
		c.e = err
	}
	return t
}
func (c *errCollect) err() error { return c.e }

// attentionNode is single-head self-attention with trainable projections:
//
//	Q = x@Wq;  K = x@Wk;  V = x@Wv
//	A = softmax(Q@K^T / sqrt(d));  out = A@V
//
// Forward keeps Q, K, V and the attention weights A in struct fields (the
// node's own references) and registers them via SaveForBackward (ADR 006),
// which pins their arena-backed storage until this node's Backward returns.
// Backward reads all four (A is multi-consumer: dV, dA, and the softmax
// reduction) and accumulates dWq/dWk/dWv IN PLACE into the persistent
// parameter gradients, verifying the engine honored the dst contract.
type attentionNode struct {
	graph.NoParameters[float32]
	eng        compute.Engine[float32]
	saver      graph.Saver[float32]
	wq, wk, wv *graph.Parameter[float32]

	// Saved forward intermediates (kept alive by the contract).
	q, k, v, attn *t32
}

func (n *attentionNode) OpType() string                     { return "WolfAttention" }
func (n *attentionNode) Attributes() map[string]interface{} { return nil }
func (n *attentionNode) OutputShape() []int                 { return []int{attnSeq, attnDim} }
func (n *attentionNode) SetSaver(s graph.Saver[float32])    { n.saver = s }

func (n *attentionNode) Parameters() []*graph.Parameter[float32] {
	return []*graph.Parameter[float32]{n.wq, n.wk, n.wv}
}

func (n *attentionNode) scale() float32 {
	return float32(1 / math.Sqrt(float64(attnDim)))
}

func (n *attentionNode) Forward(ctx context.Context, inputs ...*t32) (*t32, error) {
	x := inputs[0]
	c := &errCollect{}
	n.q = c.t(n.eng.MatMul(ctx, x, n.wq.Value))
	n.k = c.t(n.eng.MatMul(ctx, x, n.wk.Value))
	n.v = c.t(n.eng.MatMul(ctx, x, n.wv.Value))
	if err := c.err(); err != nil {
		return nil, err
	}
	kT := c.t(n.eng.Transpose(ctx, n.k, []int{1, 0}))
	scores := c.t(n.eng.MatMul(ctx, n.q, kT))
	scaled := c.t(n.eng.MulScalar(ctx, scores, n.scale()))
	if err := c.err(); err != nil {
		return nil, err
	}
	n.attn = c.t(n.eng.Softmax(ctx, scaled, 1))
	out := c.t(n.eng.MatMul(ctx, n.attn, n.v))
	if err := c.err(); err != nil {
		return nil, err
	}
	// The contract: these arena-backed intermediates must survive until this
	// node's Backward returns, across any intra-pass reuse.
	if n.saver != nil {
		n.saver.SaveForBackward(n.q, n.k, n.v, n.attn)
	}
	return out, nil
}

func (n *attentionNode) Backward(ctx context.Context, _ types.BackwardMode, dOut *t32, inputs ...*t32) ([]*t32, error) {
	x := inputs[0]
	c := &errCollect{}

	// out = A@V
	vT := c.t(n.eng.Transpose(ctx, n.v, []int{1, 0}))
	dAttn := c.t(n.eng.MatMul(ctx, dOut, vT))
	attnT := c.t(n.eng.Transpose(ctx, n.attn, []int{1, 0}))
	dV := c.t(n.eng.MatMul(ctx, attnT, dOut))
	if err := c.err(); err != nil {
		return nil, err
	}

	// Softmax backward (reads the saved weights A, including the
	// denominator-style row reduction): dS = A * (dA - rowsum(dA*A)).
	prod := c.t(n.eng.Mul(ctx, dAttn, n.attn))
	rowSum := c.t(n.eng.ReduceSum(ctx, prod, 1, true))
	if err := c.err(); err != nil {
		return nil, err
	}
	rowSumR := c.t(n.eng.Repeat(ctx, rowSum, 1, attnSeq))
	diff := c.t(n.eng.Sub(ctx, dAttn, rowSumR))
	dScaled := c.t(n.eng.Mul(ctx, n.attn, diff))
	dScores := c.t(n.eng.MulScalar(ctx, dScaled, n.scale()))
	if err := c.err(); err != nil {
		return nil, err
	}

	// scores = Q@K^T
	dQ := c.t(n.eng.MatMul(ctx, dScores, n.k))
	dScoresT := c.t(n.eng.Transpose(ctx, dScores, []int{1, 0}))
	dK := c.t(n.eng.MatMul(ctx, dScoresT, n.q))
	if err := c.err(); err != nil {
		return nil, err
	}

	// Projections: dW = x^T @ dProj, accumulated into the PERSISTENT
	// parameter gradients in place (the zerfoo#855 dst contract).
	xT := c.t(n.eng.Transpose(ctx, x, []int{1, 0}))
	dWq := c.t(n.eng.MatMul(ctx, xT, dQ))
	dWk := c.t(n.eng.MatMul(ctx, xT, dK))
	dWv := c.t(n.eng.MatMul(ctx, xT, dV))
	if err := c.err(); err != nil {
		return nil, err
	}
	for _, acc := range []struct {
		p  *graph.Parameter[float32]
		dW *t32
	}{{n.wq, dWq}, {n.wk, dWk}, {n.wv, dWv}} {
		before := acc.p.Gradient.GetStorage()
		res, err := n.eng.Add(ctx, acc.p.Gradient, acc.dW, acc.p.Gradient)
		if err != nil {
			return nil, fmt.Errorf("accumulate %s gradient: %w", acc.p.Name, err)
		}
		if res != acc.p.Gradient || res.GetStorage() != before {
			return nil, fmt.Errorf("engine.Add relocated persistent gradient %s (storage %p -> %p); dst must be written in place", acc.p.Name, before, res.GetStorage())
		}
	}

	// dx = dQ@Wq^T + dK@Wk^T + dV@Wv^T
	wqT := c.t(n.eng.Transpose(ctx, n.wq.Value, []int{1, 0}))
	wkT := c.t(n.eng.Transpose(ctx, n.wk.Value, []int{1, 0}))
	wvT := c.t(n.eng.Transpose(ctx, n.wv.Value, []int{1, 0}))
	if err := c.err(); err != nil {
		return nil, err
	}
	dxQ := c.t(n.eng.MatMul(ctx, dQ, wqT))
	dxK := c.t(n.eng.MatMul(ctx, dK, wkT))
	dxV := c.t(n.eng.MatMul(ctx, dV, wvT))
	if err := c.err(); err != nil {
		return nil, err
	}
	dx := c.t(n.eng.Add(ctx, dxQ, dxK))
	dx = c.t(n.eng.Add(ctx, dx, dxV))
	if err := c.err(); err != nil {
		return nil, err
	}
	return []*t32{dx}, nil
}

// residualNormNode computes y = (x + a) * rsqrt(mean((x+a)^2) + eps)
// row-wise (residual + RMS normalization). Forward saves the pre-norm
// activation r and the normalization inverse invCol -- the Wolf QK-norm
// cached-inverse shape that corrupted under per-sample ResetPool -- via
// SaveForBackward; Backward reads both.
type residualNormNode struct {
	graph.NoParameters[float32]
	eng   compute.Engine[float32]
	saver graph.Saver[float32]
	eps   float32

	// Saved forward intermediates (kept alive by the contract).
	r, invCol *t32
}

func (n *residualNormNode) OpType() string                     { return "WolfResidualRMSNorm" }
func (n *residualNormNode) Attributes() map[string]interface{} { return nil }
func (n *residualNormNode) OutputShape() []int                 { return []int{attnSeq, attnDim} }
func (n *residualNormNode) SetSaver(s graph.Saver[float32])    { n.saver = s }

func (n *residualNormNode) Forward(ctx context.Context, inputs ...*t32) (*t32, error) {
	x, a := inputs[0], inputs[1]
	c := &errCollect{}
	n.r = c.t(n.eng.Add(ctx, x, a))
	if err := c.err(); err != nil {
		return nil, err
	}
	sq := c.t(n.eng.Mul(ctx, n.r, n.r))
	mean := c.t(n.eng.ReduceMean(ctx, sq, 1, true))
	if err := c.err(); err != nil {
		return nil, err
	}
	meanEps := c.t(n.eng.AddScalar(ctx, mean, n.eps))
	n.invCol = c.t(n.eng.Rsqrt(ctx, meanEps))
	if err := c.err(); err != nil {
		return nil, err
	}
	invR := c.t(n.eng.Repeat(ctx, n.invCol, 1, attnDim))
	y := c.t(n.eng.Mul(ctx, n.r, invR))
	if err := c.err(); err != nil {
		return nil, err
	}
	if n.saver != nil {
		n.saver.SaveForBackward(n.r, n.invCol)
	}
	return y, nil
}

// Backward: with inv = (mean(r^2)+eps)^(-1/2) per row and D = attnDim,
//
//	dr_j = dy_j*inv - r_j * inv^3 * rowsum(dy*r) / D
//
// and the residual fans dr out to both inputs.
func (n *residualNormNode) Backward(ctx context.Context, _ types.BackwardMode, dy *t32, _ ...*t32) ([]*t32, error) {
	c := &errCollect{}
	invR := c.t(n.eng.Repeat(ctx, n.invCol, 1, attnDim))
	term1 := c.t(n.eng.Mul(ctx, dy, invR))
	prod := c.t(n.eng.Mul(ctx, dy, n.r))
	if err := c.err(); err != nil {
		return nil, err
	}
	rowSum := c.t(n.eng.ReduceSum(ctx, prod, 1, true))
	inv2 := c.t(n.eng.Mul(ctx, n.invCol, n.invCol))
	inv3 := c.t(n.eng.Mul(ctx, inv2, n.invCol))
	if err := c.err(); err != nil {
		return nil, err
	}
	coef := c.t(n.eng.Mul(ctx, rowSum, inv3))
	coef = c.t(n.eng.MulScalar(ctx, coef, 1/float32(attnDim)))
	if err := c.err(); err != nil {
		return nil, err
	}
	coefR := c.t(n.eng.Repeat(ctx, coef, 1, attnDim))
	term2 := c.t(n.eng.Mul(ctx, n.r, coefR))
	dr := c.t(n.eng.Sub(ctx, term1, term2))
	if err := c.err(); err != nil {
		return nil, err
	}
	return []*t32{dr, dr}, nil
}

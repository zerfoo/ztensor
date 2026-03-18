package graph

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

// makeInstruction creates a test instruction with the given opName, input
// slot indices, and output slot index. The Forward function performs a
// simple elementwise operation on the first input.
func makeInstruction(opName string, inputIdx []int, outputIdx int) Instruction[float32] {
	fwd := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		// Identity pass-through for test purposes.
		return inputs[0], nil
	}
	return Instruction[float32]{
		Forward:   fwd,
		InputIdx:  inputIdx,
		OutputIdx: outputIdx,
		OpName:    opName,
	}
}

// buildTestPlan creates an ExecutionPlan with the given instructions and slot
// count. inputIdx specifies which slot receives the graph input; outputIdx
// specifies which slot holds the final output.
func buildTestPlan(instructions []Instruction[float32], slotCount, inputSlot, outputSlot int) *ExecutionPlan[float32] {
	return &ExecutionPlan[float32]{
		instructions: instructions,
		slots:        make([]*tensor.TensorNumeric[float32], slotCount),
		inputIdx:     []int{inputSlot},
		outputIdx:    outputSlot,
	}
}

// TestScheduleBatches_EmptyPlan verifies that an empty plan produces no groups.
func TestScheduleBatches_EmptyPlan(t *testing.T) {
	plan := buildTestPlan(nil, 0, 0, 0)
	sched := ScheduleBatches(plan)
	if sched.GroupCount() != 0 {
		t.Errorf("empty plan: expected 0 groups, got %d", sched.GroupCount())
	}
	if sched.TotalInstructions != 0 {
		t.Errorf("expected TotalInstructions=0, got %d", sched.TotalInstructions)
	}
}

// TestScheduleBatches_AllBatchable verifies a plan whose every instruction is
// batchable produces a single group covering the full instruction list.
func TestScheduleBatches_AllBatchable(t *testing.T) {
	instrs := []Instruction[float32]{
		makeInstruction("Add", []int{0}, 1),
		makeInstruction("Mul", []int{1}, 2),
		makeInstruction("MulScalar", []int{2}, 3),
		makeInstruction("Exp", []int{3}, 4),
	}
	plan := buildTestPlan(instrs, 5, 0, 4)
	sched := ScheduleBatches(plan)

	if sched.GroupCount() != 1 {
		t.Fatalf("expected 1 group, got %d", sched.GroupCount())
	}
	g := sched.Groups[0]
	if g.Start != 0 || g.End != 4 {
		t.Errorf("expected group [0,4), got [%d,%d)", g.Start, g.End)
	}
	if sched.BatchedCount != 4 {
		t.Errorf("expected BatchedCount=4, got %d", sched.BatchedCount)
	}
}

// TestScheduleBatches_NoBatchable verifies a plan with no batchable ops
// produces no groups.
func TestScheduleBatches_NoBatchable(t *testing.T) {
	instrs := []Instruction[float32]{
		makeInstruction("EmbeddingLookup", []int{0}, 1),
		makeInstruction("MatMul", []int{1}, 2),
		makeInstruction("Softmax", []int{2}, 3),
	}
	plan := buildTestPlan(instrs, 4, 0, 3)
	sched := ScheduleBatches(plan)

	if sched.GroupCount() != 0 {
		t.Errorf("expected 0 groups, got %d", sched.GroupCount())
	}
	if sched.BatchedCount != 0 {
		t.Errorf("expected BatchedCount=0, got %d", sched.BatchedCount)
	}
}

// TestScheduleBatches_MixedOps verifies that batchable runs are correctly
// identified within a mixed instruction sequence.
func TestScheduleBatches_MixedOps(t *testing.T) {
	// Sequence: MatMul | Add Mul MulScalar | Softmax | Exp Log Tanh Sqrt | MatMul
	// Batchable runs: [1,4) and [5,9)
	instrs := []Instruction[float32]{
		makeInstruction("MatMul", []int{0}, 1),      // 0 — non-batchable
		makeInstruction("Add", []int{1}, 2),          // 1
		makeInstruction("Mul", []int{2}, 3),          // 2
		makeInstruction("MulScalar", []int{3}, 4),   // 3
		makeInstruction("Softmax", []int{4}, 5),      // 4 — non-batchable
		makeInstruction("Exp", []int{5}, 6),          // 5
		makeInstruction("Log", []int{6}, 7),          // 6
		makeInstruction("Tanh", []int{7}, 8),         // 7
		makeInstruction("Sqrt", []int{8}, 9),         // 8
		makeInstruction("MatMul", []int{9}, 10),      // 9 — non-batchable
	}
	plan := buildTestPlan(instrs, 11, 0, 10)
	sched := ScheduleBatches(plan)

	if sched.TotalInstructions != 10 {
		t.Errorf("expected TotalInstructions=10, got %d", sched.TotalInstructions)
	}
	if sched.GroupCount() != 2 {
		t.Fatalf("expected 2 groups, got %d", sched.GroupCount())
	}
	cases := []struct {
		wantStart, wantEnd int
	}{
		{1, 4},
		{5, 9},
	}
	for i, tc := range cases {
		g := sched.Groups[i]
		if g.Start != tc.wantStart || g.End != tc.wantEnd {
			t.Errorf("group[%d]: expected [%d,%d), got [%d,%d)", i, tc.wantStart, tc.wantEnd, g.Start, g.End)
		}
	}
	if sched.BatchedCount != 7 {
		t.Errorf("expected BatchedCount=7, got %d", sched.BatchedCount)
	}
}

// TestScheduleBatches_SingleBatchable verifies that a single batchable op
// (below minBatchSize) does NOT form a group.
func TestScheduleBatches_SingleBatchable(t *testing.T) {
	instrs := []Instruction[float32]{
		makeInstruction("MatMul", []int{0}, 1),
		makeInstruction("Add", []int{1}, 2),   // single batchable — too short
		makeInstruction("Softmax", []int{2}, 3),
	}
	plan := buildTestPlan(instrs, 4, 0, 3)
	sched := ScheduleBatches(plan)

	if sched.GroupCount() != 0 {
		t.Errorf("expected 0 groups (single batchable below minBatchSize), got %d", sched.GroupCount())
	}
}

// TestScheduleBatches_Coverage verifies the Coverage helper.
func TestScheduleBatches_Coverage(t *testing.T) {
	// 4 batchable out of 6 total = 66.6...%
	instrs := []Instruction[float32]{
		makeInstruction("MatMul", []int{0}, 1),
		makeInstruction("Add", []int{1}, 2),
		makeInstruction("Mul", []int{2}, 3),
		makeInstruction("MulScalar", []int{3}, 4),
		makeInstruction("Exp", []int{4}, 5),
		makeInstruction("Softmax", []int{5}, 6),
	}
	plan := buildTestPlan(instrs, 7, 0, 6)
	sched := ScheduleBatches(plan)

	want := 4.0 / 6.0
	got := sched.Coverage()
	if got < want-0.001 || got > want+0.001 {
		t.Errorf("Coverage: expected ~%.4f, got %.4f", want, got)
	}
}

// TestRunBatched_ProducesCorrectOutput verifies that RunBatched produces
// the same output as RunInstructions for a simple linear chain.
func TestRunBatched_ProducesCorrectOutput(t *testing.T) {
	// Build a plan with two batchable instructions: double, then add 10.
	doubleInstr := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		for i, v := range in {
			out[i] = v * 2
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}
	add10Instr := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		for i, v := range in {
			out[i] = v + 10
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}

	instrs := []Instruction[float32]{
		{Forward: doubleInstr, InputIdx: []int{0}, OutputIdx: 1, OpName: "Mul"},
		{Forward: add10Instr, InputIdx: []int{1}, OutputIdx: 2, OpName: "AddScalar"},
	}
	plan := buildTestPlan(instrs, 3, 0, 2)
	sched := ScheduleBatches(plan)

	if sched.GroupCount() != 1 {
		t.Fatalf("expected 1 batch group, got %d", sched.GroupCount())
	}

	input, err := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}

	result, err := plan.RunBatched(context.Background(), sched, input)
	if err != nil {
		t.Fatalf("RunBatched: %v", err)
	}

	want := []float32{12, 14, 16, 18}
	got := result.Data()
	for i, w := range want {
		if got[i] != w {
			t.Errorf("result[%d] = %v, want %v", i, got[i], w)
		}
	}
}

// TestRunBatched_MixedPlan verifies correct output when the plan mixes
// batched and non-batched instructions.
func TestRunBatched_MixedPlan(t *testing.T) {
	// Plan: triple (non-batchable) | double, add10 (batched)
	tripleInstr := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		for i, v := range in {
			out[i] = v * 3
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}
	doubleInstr := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		for i, v := range in {
			out[i] = v * 2
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}
	add5Instr := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		for i, v := range in {
			out[i] = v + 5
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}

	instrs := []Instruction[float32]{
		{Forward: tripleInstr, InputIdx: []int{0}, OutputIdx: 1, OpName: "MatMul"}, // non-batchable
		{Forward: doubleInstr, InputIdx: []int{1}, OutputIdx: 2, OpName: "Mul"},    // batchable
		{Forward: add5Instr, InputIdx: []int{2}, OutputIdx: 3, OpName: "AddScalar"}, // batchable
	}
	plan := buildTestPlan(instrs, 4, 0, 3)
	sched := ScheduleBatches(plan)

	input, err := tensor.New[float32]([]int{1, 3}, []float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}

	result, err := plan.RunBatched(context.Background(), sched, input)
	if err != nil {
		t.Fatalf("RunBatched: %v", err)
	}

	// Expected: ((x*3)*2)+5 = x*6+5
	want := []float32{11, 17, 23}
	got := result.Data()
	for i, w := range want {
		if got[i] != w {
			t.Errorf("result[%d] = %v, want %v", i, got[i], w)
		}
	}
}

// TestRunBatched_MatchesRunInstructions verifies that RunBatched output
// matches RunInstructions output for an identical plan (regression guard).
func TestRunBatched_MatchesRunInstructions(t *testing.T) {
	makeLinear := func(factor float32) func(context.Context, []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
			in := inputs[0].Data()
			out := make([]float32, len(in))
			for i, v := range in {
				out[i] = v * factor
			}
			return tensor.New[float32](inputs[0].Shape(), out)
		}
	}

	instrs := []Instruction[float32]{
		{Forward: makeLinear(2), InputIdx: []int{0}, OutputIdx: 1, OpName: "Mul"},
		{Forward: makeLinear(3), InputIdx: []int{1}, OutputIdx: 2, OpName: "Mul"},
		{Forward: makeLinear(0.5), InputIdx: []int{2}, OutputIdx: 3, OpName: "MulScalar"},
	}
	plan := buildTestPlan(instrs, 4, 0, 3)
	sched := ScheduleBatches(plan)

	input, err := tensor.New[float32]([]int{1, 4}, []float32{2, 4, 6, 8})
	if err != nil {
		t.Fatal(err)
	}

	batchedResult, err := plan.RunBatched(context.Background(), sched, input)
	if err != nil {
		t.Fatalf("RunBatched: %v", err)
	}

	// Reset scratch slots by clearing them so RunInstructions reinitialises.
	plan.scratchSlots = nil

	normalResult, err := plan.RunInstructions(context.Background(), input)
	if err != nil {
		t.Fatalf("RunInstructions: %v", err)
	}

	batchedData := batchedResult.Data()
	normalData := normalResult.Data()
	if len(batchedData) != len(normalData) {
		t.Fatalf("length mismatch: batched=%d normal=%d", len(batchedData), len(normalData))
	}
	for i := range batchedData {
		if batchedData[i] != normalData[i] {
			t.Errorf("data[%d]: batched=%v normal=%v", i, batchedData[i], normalData[i])
		}
	}
}

// TestApplyBatchSchedule_ExpandsCapture verifies that ApplyBatchSchedule
// correctly expands the CUDA graph capture region to absorb adjacent batchable
// groups that are also GPU-capturable.
func TestApplyBatchSchedule_ExpandsCapture(t *testing.T) {
	// Plan layout:
	// 0: EmbeddingLookup (pre-capture, non-capturable)
	// 1: Add   \
	// 2: Mul    > batch group [1,4) — capturable
	// 3: Exp   /
	// 4: MatMul  (original capture start)
	// ...
	// 8: MatMul  (original capture end)
	// 9:  Sub  \
	// 10: Mul   > batch group [9,11) — capturable
	// 11: Softmax (non-capturable, breaks expansion)

	instrs := []Instruction[float32]{
		makeInstruction("EmbeddingLookup", []int{0}, 1), // 0
		makeInstruction("Add", []int{1}, 2),              // 1
		makeInstruction("Mul", []int{2}, 3),              // 2
		makeInstruction("Exp", []int{3}, 4),              // 3
		makeInstruction("MatMul", []int{4}, 5),           // 4
		makeInstruction("MatMul", []int{5}, 6),           // 5
		makeInstruction("MatMul", []int{6}, 7),           // 6
		makeInstruction("MatMul", []int{7}, 8),           // 7
		makeInstruction("MatMul", []int{8}, 9),           // 8
		makeInstruction("Sub", []int{9}, 10),             // 9
		makeInstruction("Mul", []int{10}, 11),            // 10
	}
	plan := buildTestPlan(instrs, 12, 0, 11)
	sched := ScheduleBatches(plan)

	// Original capture region covers just the MatMul block.
	captureStart, captureEnd := 4, 9
	newStart, newEnd := ApplyBatchSchedule(plan, captureStart, captureEnd, sched)

	// [1,4) should be absorbed since it's immediately before captureStart=4
	// and all of Add, Mul, Exp are capturable (not in nonCapturableOps).
	if newStart != 1 {
		t.Errorf("expected newStart=1 (absorbed batch group [1,4)), got %d", newStart)
	}
	// [9,11) should be absorbed since it immediately follows captureEnd=9.
	if newEnd != 11 {
		t.Errorf("expected newEnd=11 (absorbed batch group [9,11)), got %d", newEnd)
	}
}

// TestApplyBatchSchedule_NoExpansionWhenNonCapturable verifies that a batch
// group containing non-capturable ops is NOT absorbed into the capture region.
func TestApplyBatchSchedule_NoExpansionWhenNonCapturable(t *testing.T) {
	// Plan:
	// 0: EmbeddingLookup  \  batch group [0,2) but non-capturable
	// 1: Gather           /
	// 2: MatMul (capture start)
	// 3: MatMul (capture end)

	instrs := []Instruction[float32]{
		makeInstruction("EmbeddingLookup", []int{0}, 1), // 0 — non-capturable
		makeInstruction("Gather", []int{1}, 2),           // 1 — non-capturable
		makeInstruction("MatMul", []int{2}, 3),           // 2
		makeInstruction("MatMul", []int{3}, 4),           // 3
	}
	plan := buildTestPlan(instrs, 5, 0, 4)

	// Manually create a schedule with a group [0,2) to test the guard.
	sched := &BatchSchedule{
		Groups:            []BatchGroup{{Start: 0, End: 2}},
		TotalInstructions: 4,
		BatchedCount:      2,
	}

	captureStart, captureEnd := 2, 4
	newStart, newEnd := ApplyBatchSchedule(plan, captureStart, captureEnd, sched)

	// Group [0,2) should NOT be absorbed because EmbeddingLookup and Gather
	// are non-capturable.
	if newStart != 2 {
		t.Errorf("expected newStart=2 (non-capturable group not absorbed), got %d", newStart)
	}
	if newEnd != 4 {
		t.Errorf("expected newEnd=4 (unchanged), got %d", newEnd)
	}
}

// TestBatchGroup_Size verifies the Size() helper.
func TestBatchGroup_Size(t *testing.T) {
	g := BatchGroup{Start: 3, End: 7}
	if g.Size() != 4 {
		t.Errorf("expected Size()=4, got %d", g.Size())
	}
}

// TestScheduleBatches_AllOpsListed verifies that all ops in batchableOps
// are recognised by the scheduler.
func TestScheduleBatches_AllOpsListed(t *testing.T) {
	ops := []string{"Add", "Sub", "Mul", "Div", "Pow", "Exp", "Log", "Tanh", "Sqrt", "Rsqrt", "MulScalar", "AddScalar", "DivScalar"}
	instrs := make([]Instruction[float32], len(ops))
	for i, op := range ops {
		inIdx := i
		outIdx := i + 1
		instrs[i] = makeInstruction(op, []int{inIdx}, outIdx)
	}
	plan := buildTestPlan(instrs, len(ops)+1, 0, len(ops))
	sched := ScheduleBatches(plan)

	if sched.GroupCount() != 1 {
		t.Fatalf("expected 1 group for all batchable ops, got %d", sched.GroupCount())
	}
	g := sched.Groups[0]
	if g.Start != 0 || g.End != len(ops) {
		t.Errorf("expected group [0,%d), got [%d,%d)", len(ops), g.Start, g.End)
	}
}

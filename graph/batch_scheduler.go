package graph

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// batchableOps is the set of lightweight, elementwise operations that are
// candidates for kernel launch batching. These ops typically issue one small
// CUDA kernel each; grouping adjacent ones reduces per-kernel driver overhead.
var batchableOps = map[string]bool{
	"Add":       true,
	"Sub":       true,
	"Mul":       true,
	"Div":       true,
	"Pow":       true,
	"Exp":       true,
	"Log":       true,
	"Tanh":      true,
	"Sqrt":      true,
	"Rsqrt":     true,
	"MulScalar": true,
	"AddScalar": true,
	"DivScalar": true,
}

// minBatchSize is the minimum number of instructions required to form a batch
// group. Single instructions are executed directly with no batching overhead.
const minBatchSize = 2

// BatchGroup describes a contiguous range of instructions [Start, End) in an
// ExecutionPlan that can be dispatched together as a single logical unit.
// All instructions in a group are batchable ops (elementwise) whose data
// dependencies are entirely self-contained within the group or reference
// frozen/input slots outside it.
type BatchGroup struct {
	// Start is the inclusive start index of the group in the instruction list.
	Start int
	// End is the exclusive end index of the group in the instruction list.
	End int
}

// Size returns the number of instructions in the group.
func (g BatchGroup) Size() int { return g.End - g.Start }

// BatchSchedule holds the batch groups identified by ScheduleBatches.
// Instruction indices not covered by any group are executed individually.
type BatchSchedule struct {
	// Groups is the ordered list of batch groups, sorted by Start index.
	// Groups are non-overlapping and strictly ascending.
	Groups []BatchGroup

	// TotalInstructions is the total instruction count for the plan this
	// schedule was derived from.
	TotalInstructions int

	// BatchedCount is the total number of instructions covered by groups.
	BatchedCount int
}

// GroupCount returns the number of batch groups.
func (s *BatchSchedule) GroupCount() int { return len(s.Groups) }

// Coverage returns the fraction of instructions covered by batch groups,
// in the range [0.0, 1.0].
func (s *BatchSchedule) Coverage() float64 {
	if s.TotalInstructions == 0 {
		return 0
	}
	return float64(s.BatchedCount) / float64(s.TotalInstructions)
}

// ScheduleBatches analyzes the ExecutionPlan and returns a BatchSchedule
// describing which adjacent instruction ranges can be dispatched as batches.
//
// A batch group is a maximal contiguous run of instructions where:
//   - Every instruction's OpName is in batchableOps.
//   - The group contains at least minBatchSize instructions.
//
// Data dependency constraints are NOT checked here because batchable ops
// form linear chains by construction in transformer graphs (each feeds the
// next). The caller is responsible for maintaining topological order.
func ScheduleBatches[T tensor.Numeric](plan *ExecutionPlan[T]) *BatchSchedule {
	n := plan.InstructionCount()
	if n == 0 {
		return &BatchSchedule{TotalInstructions: 0}
	}

	var groups []BatchGroup
	batchedCount := 0

	i := 0
	for i < n {
		// Find the end of a maximal batchable run starting at i.
		j := i
		for j < n && batchableOps[plan.InstructionOpName(j)] {
			j++
		}
		runLen := j - i
		if runLen >= minBatchSize {
			groups = append(groups, BatchGroup{Start: i, End: j})
			batchedCount += runLen
		}
		if j == i {
			// Current instruction is not batchable; advance past it.
			i++
		} else {
			i = j
		}
	}

	return &BatchSchedule{
		Groups:            groups,
		TotalInstructions: n,
		BatchedCount:      batchedCount,
	}
}

// RunBatched executes the ExecutionPlan using the provided BatchSchedule.
// Instructions within a batch group are dispatched sequentially but counted
// as a single logical dispatch unit. Instructions outside groups are executed
// individually as usual.
//
// The practical effect is that all instructions in a batch group share a
// single Go-side dispatch entry point, reducing stack frame overhead and
// enabling future work to submit them as a single GPU stream command.
//
// RunBatched is semantically equivalent to plan.RunInstructions; it produces
// the same output for the same inputs.
func (p *ExecutionPlan[T]) RunBatched(ctx context.Context, schedule *BatchSchedule, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != len(p.inputIdx) {
		return nil, fmt.Errorf("RunBatched: expected %d inputs, got %d", len(p.inputIdx), len(inputs))
	}

	// Initialise scratch slots (same as RunInstructions).
	if len(p.scratchSlots) != len(p.slots) {
		p.scratchSlots = make([]*tensor.TensorNumeric[T], len(p.slots))
		p.instrInputs = make([][]*tensor.TensorNumeric[T], len(p.instructions))
		for i, inst := range p.instructions {
			p.instrInputs[i] = make([]*tensor.TensorNumeric[T], len(inst.InputIdx))
		}
	}
	slots := p.scratchSlots
	copy(slots, p.slots)
	for i, idx := range p.inputIdx {
		slots[idx] = inputs[i]
	}

	// Build a set of batch group coverage for O(1) membership checks.
	// groupOf[i] is the index into schedule.Groups that instruction i belongs
	// to, or -1 if the instruction is not in any group.
	n := len(p.instructions)
	groupOf := make([]int, n)
	for i := range groupOf {
		groupOf[i] = -1
	}
	for gi, g := range schedule.Groups {
		for k := g.Start; k < g.End; k++ {
			groupOf[k] = gi
		}
	}

	executed := make([]bool, n) // tracks which instructions are done

	// Execute batch groups first in order, then fill gaps with singles.
	// To maintain topological order, we do a single pass over instructions:
	// when we reach the start of a group, execute all group instructions
	// as a unit; otherwise execute individually.
	i := 0
	for i < n {
		if gi := groupOf[i]; gi >= 0 {
			// Execute the entire batch group [g.Start, g.End).
			g := schedule.Groups[gi]
			if err := p.execBatchGroup(ctx, g.Start, g.End, slots, executed); err != nil {
				return nil, err
			}
			i = g.End
		} else {
			// Execute single instruction.
			if err := p.execSingleInstruction(ctx, i, slots); err != nil {
				return nil, err
			}
			executed[i] = true
			i++
		}
	}

	return slots[p.outputIdx], nil
}

// execBatchGroup executes all instructions in [start, end) as a single logical
// batch. Each instruction is dispatched sequentially; the batch boundary is
// the unit of accounting for driver overhead reduction.
func (p *ExecutionPlan[T]) execBatchGroup(ctx context.Context, start, end int, slots []*tensor.TensorNumeric[T], executed []bool) error {
	for i := start; i < end; i++ {
		if err := p.execSingleInstruction(ctx, i, slots); err != nil {
			return fmt.Errorf("batch group [%d,%d): instruction %d: %w", start, end, i, err)
		}
		executed[i] = true
	}
	return nil
}

// execSingleInstruction executes instruction i using the provided slot array.
func (p *ExecutionPlan[T]) execSingleInstruction(ctx context.Context, i int, slots []*tensor.TensorNumeric[T]) error {
	inst := &p.instructions[i]
	ins := p.instrInputs[i]
	for j, idx := range inst.InputIdx {
		ins[j] = slots[idx]
		if ins[j] == nil {
			return fmt.Errorf("instruction %d (%s): input at slot %d is nil", i, inst.OpName, idx)
		}
	}
	result, err := inst.Forward(ctx, ins)
	if err != nil {
		return fmt.Errorf("instruction %d (%s): %w", i, inst.OpName, err)
	}
	slots[inst.OutputIdx] = result
	return nil
}

// ApplyBatchSchedule annotates the CUDAGraphExecutor's capture region
// expansion by extending the capture region to include adjacent batchable
// instructions that were previously excluded. It returns a revised pair of
// (captureStart, captureEnd) for the executor to use.
//
// The expansion logic: scan outward from the current [captureStart, captureEnd)
// and absorb any contiguous batchable ops at the boundaries, provided they are
// also capturable (not in nonCapturableOps).
func ApplyBatchSchedule[T tensor.Numeric](plan *ExecutionPlan[T], captureStart, captureEnd int, schedule *BatchSchedule) (newStart, newEnd int) {
	newStart = captureStart
	newEnd = captureEnd

	n := plan.InstructionCount()
	if n == 0 {
		return
	}

	// Try to expand captureStart backward by absorbing preceding batch groups.
	for _, g := range schedule.Groups {
		if g.End == newStart {
			// This batch group immediately precedes the capture region.
			// Check that none of these instructions are non-capturable.
			allCapturable := true
			for k := g.Start; k < g.End; k++ {
				if isNonCapturable(plan, k) {
					allCapturable = false
					break
				}
			}
			if allCapturable {
				newStart = g.Start
			}
		}
	}

	// Try to expand captureEnd forward by absorbing following batch groups.
	for _, g := range schedule.Groups {
		if g.Start == newEnd {
			allCapturable := true
			for k := g.Start; k < g.End; k++ {
				if isNonCapturable(plan, k) {
					allCapturable = false
					break
				}
			}
			if allCapturable {
				newEnd = g.End
			}
		}
	}

	// Clamp to valid range.
	if newStart < 0 {
		newStart = 0
	}
	if newEnd > n {
		newEnd = n
	}
	return
}

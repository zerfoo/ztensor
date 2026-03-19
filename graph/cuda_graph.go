package graph

import (
	"context"
	"fmt"
	"log"
	"os"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/tensor"
)

// debugGraphCapture enables verbose capture debug logging when ZERFOO_DEBUG_GPU=1.
var debugGraphCapture = os.Getenv("ZERFOO_DEBUG_GPU") == "1"

// nonCapturableOps lists instruction op names that must always run outside
// CUDA graph capture. These ops perform CPU work or D2H copies that are
// incompatible with stream capture.
//
// EmbeddingLookup: reads token IDs from GPU via .Data() (D2H), does CPU
// float→int conversion.
//
// Gather: uses CPU index tensors to select rows; triggers H2D copies
// incompatible with stream capture.
//
// AutoAttentionMask / AutoPositionIds: allocate CPU tensors (make([]T, ...))
// and return them via tensor.New. Downstream ops (e.g. Mul) would call
// getDevicePtr triggering cudaMemcpy H2D on the capturing stream.
//
// Slice: reads start/end/axes indices from input tensors via Data() which
// triggers D2H cudaMemcpy for GPU-resident index tensors.
//
// ConstantOfShape: allocates a CPU tensor filled with a constant value.
// Downstream ops (e.g. Mul) would trigger cudaMemcpy H2D on the
// capturing stream.
//
// Shape: produces a 1-D CPU tensor containing the shape of its input.
// Downstream consumers trigger cudaMemcpy H2D during stream capture.
//
// GroupedQueryAttention was previously non-capturable because it read
// cache.SeqLen() on the CPU for RoPE positions and used CPU-computed offsets
// for KV cache appends. Now that TensorCache uses a GPU-resident counter
// (offset_memcpy kernel) and GQA uses GPU RoPE selection (rope_select kernel),
// all position-dependent state is read from GPU memory at replay time, making
// GQA fully capturable.
var nonCapturableOps = map[string]bool{
	"EmbeddingLookup":   true,
	"Gather":            true,
	"AutoAttentionMask": true,
	"AutoPositionIds":   true,
	"Slice":             true,
	"ConstantOfShape":   true,
	"Shape":             true,
}

// isNonCapturable returns true if the instruction at index i in the plan
// cannot be captured in a CUDA graph. Most ops are checked via the
// nonCapturableOps map, but some ops like Reshape are conditionally
// capturable depending on their input count.
//
// Reshape with 1 input (static targetShape from attributes) is capture-safe:
// it only reads input.Shape() (metadata) and calls engine.Reshape which is a
// zero-copy GPU view operation. Reshape with 2 inputs (dynamic shape tensor)
// calls .Data() on the shape tensor, triggering a D2H copy.
func isNonCapturable[T tensor.Numeric](plan *ExecutionPlan[T], i int) bool {
	inst := plan.instructions[i]
	if nonCapturableOps[inst.OpName] {
		return true
	}
	if inst.OpName == "Reshape" && len(inst.InputIdx) > 1 {
		return true
	}
	return false
}

// CaptureStats reports CUDA graph capture coverage.
type CaptureStats struct {
	// TotalInstructions is the total number of instructions in the plan.
	TotalInstructions int
	// CapturedInstructions is the number of instructions inside the capture region.
	CapturedInstructions int
	// BypassedOps lists the op names that were absorbed into the graph via
	// bypass (pre-executed on CPU, outputs cached on GPU, identity copy captured).
	BypassedOps []string
	// CoveragePercent is the capture coverage as a percentage (0-100).
	CoveragePercent float64
}

// CUDAGraphExecutor captures and replays a CUDA graph for an ExecutionPlan.
// All instructions are captured into the graph (100% coverage). Ops that
// cannot run natively during stream capture (EmbeddingLookup, Gather, etc.)
// are pre-executed on the CPU, their outputs uploaded to fixed GPU buffers,
// and an identity (no-op) forward is captured for those instructions. During
// replay, the original ops run first to update the GPU buffers, then the
// full graph is replayed.
type CUDAGraphExecutor[T tensor.Numeric] struct {
	plan      *ExecutionPlan[T]
	stream    *cuda.Stream
	graphExec *cuda.GraphExec
	graph     *cuda.Graph
	warmups   int // number of warmup runs before capture
	calls     int // total calls so far
	failed    bool

	// Capture region boundaries: instructions [captureStart, captureEnd)
	// are captured into the CUDA graph. With full capture, these span
	// [0, InstructionCount).
	captureStart int
	captureEnd   int

	// replayReady is set after the first successful replay. When true,
	// replay() uses a fast path that skips PrepareSlots and EnsureSlotsGPU.
	replayReady bool

	// inputSlotIdx is the scratch slot index for the first graph input,
	// cached from plan.inputIdx[0] to avoid per-replay slice access.
	inputSlotIdx int

	// hasPostCapture is true when there are instructions after captureEnd.
	hasPostCapture bool

	// Fixed device buffer for the input token.
	inputDevPtr unsafe.Pointer

	// Cache of GPU tensors for slots that arrive as CPU from pre-capture
	// (e.g. EmbeddingLookup with Q4K). Device addresses are reused across
	// replays so the captured graph stays valid.
	gpuSlotCache map[int]*tensor.TensorNumeric[T]

	// capturedSlots holds the tensors from scratchSlots that were written
	// during the capture run. These tensors' GPU buffers are the destinations
	// of the captured graph's operations. During replay, these must be
	// restored into scratchSlots after PrepareSlots (which resets them)
	// so that GraphLaunch writes to the same buffers and OutputTensor()
	// returns the correct result.
	capturedSlots []*tensor.TensorNumeric[T] // indexed by slot number, nil for non-captured slots

	// Embedding replay optimization: after the first replay, cache the GPU
	// tensor for the EmbeddingLookup output slot. On subsequent replays
	// (replayFast), copy just the embedding vector into the cached GPU
	// buffer via CopyFromHost and set the scratch slot directly, skipping
	// the full EnsureSlotsGPU scan over all slots.
	embeddingCached  bool
	embeddingSlotIdx int                      // output slot of EmbeddingLookup
	embeddingGPUPtr  *tensor.TensorNumeric[T] // cached GPU tensor from first replay

	// onCaptured is called after successful capture, allowing the caller
	// to protect arena allocations from being reclaimed by Reset.
	onCaptured func()

	// snapshotCache is called before the capture region to snapshot KV cache
	// state. It returns a restore function that is called if capture fails,
	// allowing the caller to roll back cache mutations (e.g. Truncate seqLen
	// and reset GPU counters) before the RunInstructions fallback.
	snapshotCache func(ctx context.Context) func()

	// bypassIndices stores instruction indices of non-capturable ops that
	// were absorbed into the graph via the bypass mechanism. During capture,
	// these ops use an identity forward that returns their pre-computed GPU
	// output. During replay, the original forward runs first (outside the
	// graph) to update the GPU buffers.
	bypassIndices []int

	// bypassOrigFwd stores the original Forward functions for bypassed
	// instructions, keyed by instruction index. These are restored after
	// capture and used during replay to compute fresh results.
	bypassOrigFwd map[int]func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
}

// NewCUDAGraphExecutor creates a graph executor for the given plan.
// The optional onCaptured callback is invoked after a successful capture,
// allowing the caller to protect arena allocations from being reclaimed.
// The optional snapshotCache callback is called before the capture region to
// snapshot KV cache state. It returns a restore function invoked on capture
// failure, preventing double KV cache updates in the RunInstructions fallback.
func NewCUDAGraphExecutor[T tensor.Numeric](plan *ExecutionPlan[T], streamPtr unsafe.Pointer, warmups int, onCaptured func(), snapshotCache func(ctx context.Context) func()) *CUDAGraphExecutor[T] {
	if warmups < 1 {
		warmups = 1
	}

	n := plan.InstructionCount()
	if n == 0 {
		log.Printf("cuda graph: no instructions found, graph disabled")
		return &CUDAGraphExecutor[T]{plan: plan, failed: true}
	}

	// Full capture: the capture region spans [0, n). Non-capturable ops are
	// absorbed via the bypass mechanism — they are pre-executed on CPU, their
	// outputs uploaded to GPU, and an identity forward is captured. This
	// achieves 100% instruction coverage.
	captureStart := 0
	captureEnd := n

	// Identify non-capturable instructions that need the bypass mechanism.
	var bypassIndices []int
	for i := 0; i < n; i++ {
		if isNonCapturable(plan, i) {
			bypassIndices = append(bypassIndices, i)
		}
	}

	// Verify there is at least one capturable instruction.
	if len(bypassIndices) == n {
		log.Printf("cuda graph: all %d instructions are non-capturable, graph disabled", n)
		return &CUDAGraphExecutor[T]{plan: plan, failed: true}
	}

	if len(bypassIndices) > 0 {
		log.Printf("cuda graph: full capture [0, %d) with %d bypassed ops", n, len(bypassIndices))
	} else {
		log.Printf("cuda graph: full capture [0, %d), all ops natively capturable", n)
	}

	// Upload frozen weights to GPU before any warmup runs or graph capture.
	// This prevents getDevicePtr from issuing synchronous H2D copies on the
	// capturing stream, which would cause cuda error 901.
	if err := plan.PreUploadFrozenWeights(); err != nil {
		log.Printf("cuda graph: frozen weight pre-upload failed: %v", err)
		return &CUDAGraphExecutor[T]{plan: plan, failed: true}
	}

	inputSlotIdx := 0
	if len(plan.inputIdx) > 0 {
		inputSlotIdx = plan.inputIdx[0]
	}

	// Compute tensor lifetime analysis for intra-pass buffer reuse.
	// This annotates each slot with the last instruction that reads from it,
	// enabling the arena free-list to reclaim intermediates mid-pass.
	plan.ComputeLastUse()
	if arena := cuda.DefaultArenaPool(); arena != nil {
		plan.SetArenaFreeFn(func(ptr unsafe.Pointer, byteSize int) {
			arena.FreeArena(ptr, byteSize)
		})
	}

	return &CUDAGraphExecutor[T]{
		plan:            plan,
		stream:          cuda.StreamFromPtr(streamPtr),
		warmups:         warmups,
		captureStart:    captureStart,
		captureEnd:      captureEnd,
		inputSlotIdx:    inputSlotIdx,
		hasPostCapture:  false, // full capture: no post-capture region
		gpuSlotCache:    make(map[int]*tensor.TensorNumeric[T]),
		onCaptured:      onCaptured,
		snapshotCache:   snapshotCache,
		bypassIndices:   bypassIndices,
		bypassOrigFwd:   make(map[int]func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)),
	}
}

// Run executes the plan, using graph capture/replay when available.
func (g *CUDAGraphExecutor[T]) Run(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	g.calls++

	if g.failed {
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Phase 1: Warmup runs.
	if g.calls <= g.warmups {
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Phase 2: Capture on first post-warmup call.
	// Only capture during decode (seqLen=1). Prefill inputs have larger
	// sequence lengths and take different code paths inside composite nodes
	// (e.g. GQA uses SDPA instead of FlashAttentionDecode), which may
	// trigger allocations incompatible with CUDA stream capture.
	if g.graphExec == nil {
		if len(inputs) > 0 && inputs[0] != nil {
			shape := inputs[0].Shape()
			if len(shape) >= 2 && shape[len(shape)-1] > 1 {
				// Prefill input (seqLen > 1): skip capture, run normally.
				return g.plan.RunInstructions(ctx, inputs...)
			}
		}
		return g.captureAndRun(ctx, inputs...)
	}

	// Phase 3: Replay.
	return g.replay(ctx, inputs...)
}

// captureAndRun records the full instruction range as a CUDA graph.
// Non-capturable ops are pre-executed on CPU, their outputs uploaded to GPU,
// and identity forwards are installed for the capture pass. After capture,
// original forwards are restored for use during replay.
func (g *CUDAGraphExecutor[T]) captureAndRun(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Prepare slots with the real inputs.
	if err := g.plan.PrepareSlots(inputs...); err != nil {
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Pre-execute non-capturable ops on CPU to populate their output slots.
	// Run ALL instructions up to the first capturable instruction, then
	// selectively run remaining non-capturable ones. For simplicity, run
	// the full instruction range normally first to get all slot values.
	if len(g.bypassIndices) > 0 {
		if err := g.plan.RunInstructionRange(ctx, 0, g.captureEnd); err != nil {
			g.failed = true
			log.Printf("cuda graph: pre-execute run failed: %v", err)
			return g.plan.RunInstructions(ctx, inputs...)
		}
	}

	// Ensure all slot data is GPU-resident before capture. Non-capturable
	// instructions (e.g. EmbeddingLookup with Q4K embedding tables) may
	// produce CPU tensors. Upload them now so the capture region sees only
	// GPU-resident data and avoids sync D2H copies that break capture.
	g.plan.EnsureSlotsGPU(g.gpuSlotCache)

	// Also upload frozen scalar constants that are inputs to capture-region
	// instructions. PreUploadFrozenWeights keeps scalars on CPU for ops like
	// Range/Pow that read host values, but capture-region ops (Mul, Add, etc.)
	// need all inputs on GPU to avoid cudaMemcpy during stream capture.
	g.plan.EnsureCaptureInputsGPU(g.captureStart, g.captureEnd, g.gpuSlotCache)

	// Install bypass forwards for non-capturable instructions. During
	// capture, these return the pre-computed GPU output already in the
	// scratch slot, contributing a no-op to the graph. The original forward
	// functions are saved and restored after capture for use during replay.
	g.installBypassForwards()

	// Snapshot KV cache state before the capture region runs. If capture
	// fails, restoreCache rolls back cache mutations (seqLen, GPU counters)
	// so the RunInstructions fallback doesn't double-update the cache.
	var restoreCache func()
	if g.snapshotCache != nil {
		restoreCache = g.snapshotCache(ctx)
	}

	// Re-prepare slots for the capture pass. The pre-execute run consumed
	// the slots; we need fresh slot state with GPU-resident data.
	if err := g.plan.PrepareSlots(inputs...); err != nil {
		g.restoreBypassForwards()
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}
	g.plan.EnsureSlotsGPU(g.gpuSlotCache)
	g.plan.EnsureCaptureInputsGPU(g.captureStart, g.captureEnd, g.gpuSlotCache)

	// Begin capture for the full instruction range.
	log.Printf("CUDA GRAPH: about to begin full capture, instructions [%d, %d)", g.captureStart, g.captureEnd)
	if err := cuda.StreamBeginCapture(g.stream); err != nil {
		log.Printf("cuda graph: begin capture failed: %v", err)
		g.restoreBypassForwards()
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}
	log.Printf("CUDA GRAPH: capture started, running instructions [%d, %d)", g.captureStart, g.captureEnd)

	// Run all instructions — GPU operations are recorded. Bypassed ops
	// contribute identity forwards that just return the cached GPU tensor.
	var captureErr error
	if debugGraphCapture {
		for i := g.captureStart; i < g.captureEnd; i++ {
			opName := g.plan.InstructionOpName(i)
			log.Printf("CUDA GRAPH capture: running instruction %d/%d op=%s", i, g.captureEnd-1, opName)
			err := g.plan.RunInstructionRange(ctx, i, i+1)
			if err != nil {
				log.Printf("CUDA GRAPH capture: FAILED at instruction %d op=%s error=%v", i, opName, err)
				captureErr = err
				break
			}
			log.Printf("CUDA GRAPH capture: instruction %d op=%s OK", i, opName)
		}
	} else {
		captureErr = g.plan.RunInstructionRange(ctx, g.captureStart, g.captureEnd)
	}

	// End capture.
	capturedGraph, endErr := cuda.StreamEndCapture(g.stream)

	// Restore original forwards immediately after capture ends.
	g.restoreBypassForwards()

	if endErr != nil || captureErr != nil {
		if endErr != nil {
			log.Printf("cuda graph: end capture failed: %v", endErr)
		}
		if captureErr != nil {
			log.Printf("CUDA GRAPH: capture failed: %v", captureErr)
		}
		g.failed = true
		if capturedGraph != nil {
			_ = cuda.GraphDestroy(capturedGraph)
		}
		if restoreCache != nil {
			restoreCache()
		}
		return g.plan.RunInstructions(ctx, inputs...)
	}
	g.graph = capturedGraph

	// Instantiate executable graph.
	exec, err := cuda.GraphInstantiate(capturedGraph)
	if err != nil {
		log.Printf("cuda graph: instantiate failed: %v", err)
		_ = cuda.GraphDestroy(capturedGraph)
		g.graph = nil
		g.failed = true
		if restoreCache != nil {
			restoreCache()
		}
		return g.plan.RunInstructions(ctx, inputs...)
	}
	g.graphExec = exec
	log.Printf("cuda graph: captured and instantiated successfully (instructions %d-%d, 100%% coverage)", g.captureStart, g.captureEnd-1)

	// Save all scratch slots written by captured instructions.
	g.capturedSlots = make([]*tensor.TensorNumeric[T], g.plan.SlotCount())
	for i := g.captureStart; i < g.captureEnd; i++ {
		outSlot := g.plan.InstructionOutputIdx(i)
		if t := g.plan.ScratchSlot(outSlot); t != nil {
			g.capturedSlots[outSlot] = t
		}
	}

	// Notify the caller to protect arena allocations from reset.
	if g.onCaptured != nil {
		g.onCaptured()
	}

	// Launch the graph once to actually compute the results.
	if err := cuda.GraphLaunch(g.graphExec, g.stream); err != nil {
		return nil, fmt.Errorf("cuda graph: first launch failed: %w", err)
	}
	if err := g.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("cuda graph: sync after first launch: %w", err)
	}

	return g.plan.OutputTensor(), nil
}

// installBypassForwards replaces the Forward functions of non-capturable
// instructions with identity functions that return the pre-computed GPU
// tensor already in the scratch slot. The original forwards are saved
// in bypassOrigFwd for restoration after capture.
func (g *CUDAGraphExecutor[T]) installBypassForwards() {
	for _, idx := range g.bypassIndices {
		inst := &g.plan.instructions[idx]
		g.bypassOrigFwd[idx] = inst.Forward

		// Capture the output slot index for the closure.
		outSlot := inst.OutputIdx
		inst.Forward = func(_ context.Context, _ []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			// Return the GPU tensor already in the scratch slot from pre-execution.
			t := g.plan.ScratchSlot(outSlot)
			if t == nil {
				return nil, fmt.Errorf("bypass: no cached GPU tensor for slot %d", outSlot)
			}
			return t, nil
		}
	}
}

// restoreBypassForwards restores the original Forward functions for all
// bypassed instructions after capture completes.
func (g *CUDAGraphExecutor[T]) restoreBypassForwards() {
	for _, idx := range g.bypassIndices {
		if origFwd, ok := g.bypassOrigFwd[idx]; ok {
			g.plan.instructions[idx].Forward = origFwd
		}
	}
}

// replay launches the pre-captured graph with updated input.
// For bypassed (non-capturable) ops, the original forwards are executed
// outside the graph to compute fresh results, which are then uploaded to
// the fixed GPU buffers. The graph then replays reading the updated data.
func (g *CUDAGraphExecutor[T]) replay(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if err := g.plan.PrepareSlots(inputs...); err != nil {
		return nil, fmt.Errorf("cuda graph replay: prepare slots: %w", err)
	}

	// Run bypassed (non-capturable) instructions using their original
	// forwards to produce fresh CPU results, then upload to GPU.
	if len(g.bypassIndices) > 0 {
		for _, idx := range g.bypassIndices {
			inst := &g.plan.instructions[idx]
			ins := make([]*tensor.TensorNumeric[T], len(inst.InputIdx))
			for j, slotIdx := range inst.InputIdx {
				ins[j] = g.plan.ScratchSlot(slotIdx)
				if ins[j] == nil {
					return nil, fmt.Errorf("cuda graph replay: bypass instruction %d (%s): nil input at slot %d", idx, inst.OpName, slotIdx)
				}
			}
			result, err := inst.Forward(ctx, ins)
			if err != nil {
				return nil, fmt.Errorf("cuda graph replay: bypass instruction %d (%s): %w", idx, inst.OpName, err)
			}
			g.plan.SetScratchSlot(inst.OutputIdx, result)
		}
	}

	// Ensure all bypass outputs and other CPU-resident slots are on GPU.
	g.plan.EnsureSlotsGPU(g.gpuSlotCache)

	// Restore captured slots. PrepareSlots resets scratchSlots from p.slots,
	// which clears the tensors allocated during capture. The captured CUDA
	// graph writes to those GPU buffers, so we must restore the tensor
	// pointers so that OutputTensor() returns the correct result.
	for idx, t := range g.capturedSlots {
		if t != nil {
			g.plan.SetScratchSlot(idx, t)
		}
	}

	// Replay the captured graph.
	if err := cuda.GraphLaunch(g.graphExec, g.stream); err != nil {
		return nil, fmt.Errorf("cuda graph: launch failed: %w", err)
	}
	if err := g.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("cuda graph: sync failed: %w", err)
	}

	// Cache the EmbeddingLookup output GPU tensor for fast replay.
	g.initEmbeddingCache()

	g.replayReady = true

	return g.plan.OutputTensor(), nil
}

// replayFast is the O(1) Go-work replay path used after the first successful
// replay. It skips PrepareSlots (which copies ~185 slot pointers),
// EnsureSlotsGPU (which iterates all slots checking residency), and
// capturedSlots restoration (already in place from previous replay).
// Only the input slot is updated and bypassed instructions re-run.
func (g *CUDAGraphExecutor[T]) replayFast(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Set only the input slot — all other scratch slots are already correct
	// from the previous replay.
	g.plan.SetScratchSlot(g.inputSlotIdx, inputs[0])

	// Re-run bypassed (non-capturable) instructions because the input token
	// changes each step. Their outputs are uploaded to cached GPU buffers.
	for _, idx := range g.bypassIndices {
		inst := &g.plan.instructions[idx]
		ins := make([]*tensor.TensorNumeric[T], len(inst.InputIdx))
		for j, slotIdx := range inst.InputIdx {
			ins[j] = g.plan.ScratchSlot(slotIdx)
		}
		result, err := inst.Forward(ctx, ins)
		if err != nil {
			return nil, fmt.Errorf("cuda graph replay: bypass instruction %d (%s): %w", idx, inst.OpName, err)
		}
		g.plan.SetScratchSlot(inst.OutputIdx, result)
	}

	// Targeted GPU upload for the EmbeddingLookup output slot.
	if g.embeddingCached {
		g.uploadEmbeddingToGPU()
	}

	// Launch the captured graph — slots are already GPU-resident and
	// capturedSlots tensors are still in place from the previous replay.
	if err := cuda.GraphLaunch(g.graphExec, g.stream); err != nil {
		return nil, fmt.Errorf("cuda graph: launch failed: %w", err)
	}
	if err := g.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("cuda graph: sync failed: %w", err)
	}

	return g.plan.OutputTensor(), nil
}

// initEmbeddingCache identifies the EmbeddingLookup bypassed instruction
// and caches its GPU tensor from gpuSlotCache. Called once after the first
// replay when EnsureSlotsGPU has populated the cache.
func (g *CUDAGraphExecutor[T]) initEmbeddingCache() {
	for _, idx := range g.bypassIndices {
		if g.plan.InstructionOpName(idx) == "EmbeddingLookup" {
			slotIdx := g.plan.InstructionOutputIdx(idx)
			if cached, ok := g.gpuSlotCache[slotIdx]; ok {
				g.embeddingCached = true
				g.embeddingSlotIdx = slotIdx
				g.embeddingGPUPtr = cached
			}
			break
		}
	}
}

// uploadEmbeddingToGPU copies the CPU embedding result from the pre-capture
// EmbeddingLookup instruction into the cached GPU tensor and sets the scratch
// slot. This replaces the full EnsureSlotsGPU scan with a single CopyFromHost.
func (g *CUDAGraphExecutor[T]) uploadEmbeddingToGPU() {
	cpuResult := g.plan.ScratchSlot(g.embeddingSlotIdx)
	if cpuResult == nil {
		return
	}
	// If already GPU-resident (shouldn't happen, but safe), skip.
	if _, ok := cpuResult.GetStorage().(*tensor.GPUStorage[T]); ok {
		return
	}
	gs, ok := g.embeddingGPUPtr.GetStorage().(*tensor.GPUStorage[T])
	if !ok {
		return
	}
	if err := gs.CopyFromHost(cpuResult.Data(), 0); err == nil {
		g.plan.SetScratchSlot(g.embeddingSlotIdx, g.embeddingGPUPtr)
	}
}

// CaptureStats returns statistics about the CUDA graph capture coverage.
func (g *CUDAGraphExecutor[T]) CaptureStats() CaptureStats {
	total := g.plan.InstructionCount()
	captured := g.captureEnd - g.captureStart
	var bypassed []string
	for _, idx := range g.bypassIndices {
		bypassed = append(bypassed, g.plan.InstructionOpName(idx))
	}
	pct := 0.0
	if total > 0 {
		pct = float64(captured) / float64(total) * 100.0
	}
	return CaptureStats{
		TotalInstructions:    total,
		CapturedInstructions: captured,
		BypassedOps:          bypassed,
		CoveragePercent:      pct,
	}
}

// Destroy releases the CUDA graph resources.
func (g *CUDAGraphExecutor[T]) Destroy() {
	if g.graphExec != nil {
		_ = cuda.GraphExecDestroy(g.graphExec)
		g.graphExec = nil
	}
	if g.graph != nil {
		_ = cuda.GraphDestroy(g.graph)
		g.graph = nil
	}
	if g.inputDevPtr != nil {
		_ = cuda.Free(g.inputDevPtr)
		g.inputDevPtr = nil
	}
}

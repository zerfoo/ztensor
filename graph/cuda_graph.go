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

// CUDAGraphExecutor captures and replays a CUDA graph for an ExecutionPlan.
// It splits the plan into three regions:
//  1. Pre-capture: instructions that trigger D2H copies or have dynamic state
//  2. Capture region: GPU-only, position-independent instructions
//  3. Post-capture: any trailing non-capturable instructions
//
// During replay, regions 1 and 3 run normally while region 2 is replayed
// from the captured graph with near-zero launch overhead.
type CUDAGraphExecutor[T tensor.Numeric] struct {
	plan      *ExecutionPlan[T]
	stream    *cuda.Stream
	graphExec *cuda.GraphExec
	graph     *cuda.Graph
	warmups   int // number of warmup runs before capture
	calls     int // total calls so far
	failed    bool

	// Capture region boundaries: instructions [captureStart, captureEnd)
	// are captured into the CUDA graph. Instructions outside this range
	// run normally every call.
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

	// Determine capture region: find the LONGEST contiguous run of capturable
	// instructions. Non-capturable ops (EmbeddingLookup, Gather, Slice, etc.)
	// may appear at the start, end, or scattered throughout the instruction
	// list. The longest run is typically the transformer layers (attention +
	// FFN) which form the bulk of GPU compute.
	n := plan.InstructionCount()
	captureStart, captureEnd := 0, 0
	runStart := 0
	for i := 0; i <= n; i++ {
		if i == n || isNonCapturable(plan, i) {
			if i-runStart > captureEnd-captureStart {
				captureStart = runStart
				captureEnd = i
			}
			runStart = i + 1
		}
	}

	if captureStart >= captureEnd {
		log.Printf("cuda graph: no capturable instructions found, graph disabled")
		return &CUDAGraphExecutor[T]{plan: plan, failed: true}
	}
	log.Printf("cuda graph: capture region is instructions [%d, %d) of %d total", captureStart, captureEnd, n)

	// NOTE: PreUploadFrozenWeights was removed. Phase 6 achieved 234 tok/s
	// with CUDA graph capture WITHOUT pre-uploading frozen weights. Adding
	// PreUploadFrozenWeights converted Q4Storage to float32 (8x bandwidth),
	// regressing throughput from 234 to 188. The Q4 GEMV path with
	// arena-based GPU allocation is capture-safe and faster.
	// Weights are pre-uploaded by UploadWeights in inference/load_gguf.go.

	inputSlotIdx := 0
	if len(plan.inputIdx) > 0 {
		inputSlotIdx = plan.inputIdx[0]
	}

	return &CUDAGraphExecutor[T]{
		plan:            plan,
		stream:          cuda.StreamFromPtr(streamPtr),
		warmups:         warmups,
		captureStart:    captureStart,
		captureEnd:      captureEnd,
		inputSlotIdx:    inputSlotIdx,
		hasPostCapture:  captureEnd < plan.InstructionCount(),
		gpuSlotCache:    make(map[int]*tensor.TensorNumeric[T]),
		onCaptured:    onCaptured,
		snapshotCache: snapshotCache,
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
	if g.graphExec == nil {
		return g.captureAndRun(ctx, inputs...)
	}

	// Phase 3: Replay.
	return g.replay(ctx, inputs...)
}

// captureAndRun records the capturable region as a CUDA graph.
func (g *CUDAGraphExecutor[T]) captureAndRun(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Prepare slots with the real inputs.
	if err := g.plan.PrepareSlots(inputs...); err != nil {
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Run pre-capture instructions normally (e.g. EmbeddingLookup).
	if g.captureStart > 0 {
		if err := g.plan.RunInstructionRange(ctx, 0, g.captureStart); err != nil {
			g.failed = true
			log.Printf("cuda graph: pre-capture run failed: %v", err)
			return g.plan.RunInstructions(ctx, inputs...)
		}
	}

	// Ensure all slot data is GPU-resident before capture. Pre-capture
	// instructions (e.g. EmbeddingLookup with Q4K embedding tables) may
	// produce CPU tensors. Upload them now so the capture region sees only
	// GPU-resident data and avoids sync D2H copies that break capture.
	g.plan.EnsureSlotsGPU(g.gpuSlotCache)

	// Also upload frozen scalar constants that are inputs to capture-region
	// instructions. PreUploadFrozenWeights keeps scalars on CPU for ops like
	// Range/Pow that read host values, but capture-region ops (Mul, Add, etc.)
	// need all inputs on GPU to avoid cudaMemcpy during stream capture.
	g.plan.EnsureCaptureInputsGPU(g.captureStart, g.captureEnd, g.gpuSlotCache)

	// Snapshot KV cache state before the capture region runs. If capture
	// fails, restoreCache rolls back cache mutations (seqLen, GPU counters)
	// so the RunInstructions fallback doesn't double-update the cache.
	var restoreCache func()
	if g.snapshotCache != nil {
		restoreCache = g.snapshotCache(ctx)
	}

	// Begin capture for the GPU-heavy region.
	log.Printf("CUDA GRAPH: about to begin capture, instructions [%d, %d)", g.captureStart, g.captureEnd)
	if err := cuda.StreamBeginCapture(g.stream); err != nil {
		log.Printf("cuda graph: begin capture failed: %v", err)
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}
	log.Printf("CUDA GRAPH: capture started, running instructions [%d, %d)", g.captureStart, g.captureEnd)

	// Run capturable instructions — GPU operations are recorded.
	var captureErr error
	if debugGraphCapture {
		// Run instructions one at a time with logging to identify the exact failure point.
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
		// Restore KV cache state before fallback: the capture region ran
		// GQA layers that called cache.Update(), incrementing seqLen and
		// GPU counters. Without this restore, the fallback RunInstructions
		// would double-update the cache for the same token.
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
		// Restore KV cache state: capture region already ran cache.Update().
		if restoreCache != nil {
			restoreCache()
		}
		return g.plan.RunInstructions(ctx, inputs...)
	}
	g.graphExec = exec
	log.Printf("cuda graph: captured and instantiated successfully (instructions %d-%d)", g.captureStart, g.captureEnd-1)

	// Save all scratch slots written by captured instructions. These tensors
	// hold the GPU buffers that the captured graph writes to. During replay,
	// we must restore them after PrepareSlots (which resets scratchSlots).
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

	// Run post-capture instructions if any.
	if g.captureEnd < g.plan.InstructionCount() {
		if err := g.plan.RunInstructionRange(ctx, g.captureEnd, g.plan.InstructionCount()); err != nil {
			return nil, fmt.Errorf("cuda graph: post-capture run failed: %w", err)
		}
	}

	return g.plan.OutputTensor(), nil
}

// replay launches the pre-captured graph with updated input.
func (g *CUDAGraphExecutor[T]) replay(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if g.replayReady {
		return g.replayFast(ctx, inputs...)
	}

	// First replay: full path that sets up slot state.
	if err := g.plan.PrepareSlots(inputs...); err != nil {
		return nil, fmt.Errorf("cuda graph replay: prepare slots: %w", err)
	}

	// Run pre-capture instructions normally.
	if g.captureStart > 0 {
		if err := g.plan.RunInstructionRange(ctx, 0, g.captureStart); err != nil {
			return nil, fmt.Errorf("cuda graph replay: pre-capture: %w", err)
		}
	}

	// Ensure pre-capture outputs are GPU-resident before replay.
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

	// Run post-capture instructions if any.
	if g.hasPostCapture {
		if err := g.plan.RunInstructionRange(ctx, g.captureEnd, g.plan.InstructionCount()); err != nil {
			return nil, fmt.Errorf("cuda graph replay: post-capture: %w", err)
		}
	}

	// Cache the EmbeddingLookup output GPU tensor for fast replay.
	// After EnsureSlotsGPU, the embedding slot has a GPU tensor that we can
	// reuse on subsequent replays via targeted CopyFromHost.
	g.initEmbeddingCache()

	// All slots are now GPU-resident and capturedSlots are in place.
	// Subsequent replays can use the fast path.
	g.replayReady = true

	return g.plan.OutputTensor(), nil
}

// replayFast is the O(1) Go-work replay path used after the first successful
// replay. It skips PrepareSlots (which copies ~185 slot pointers),
// EnsureSlotsGPU (which iterates all slots checking residency), and
// capturedSlots restoration (already in place from previous replay).
// Only the input slot is updated and pre-capture instructions re-run.
func (g *CUDAGraphExecutor[T]) replayFast(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Set only the input slot — all other scratch slots are already correct
	// from the previous replay.
	g.plan.SetScratchSlot(g.inputSlotIdx, inputs[0])

	// Pre-capture instructions (e.g. EmbeddingLookup) must still run because
	// the input token changes each step.
	if g.captureStart > 0 {
		if err := g.plan.RunInstructionRange(ctx, 0, g.captureStart); err != nil {
			return nil, fmt.Errorf("cuda graph replay: pre-capture: %w", err)
		}
	}

	// Targeted GPU upload for the EmbeddingLookup output slot. Instead of
	// scanning all slots via EnsureSlotsGPU, copy just the embedding vector
	// into the cached GPU buffer. This avoids iterating ~185 slots.
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

	// Run post-capture instructions if any.
	if g.hasPostCapture {
		if err := g.plan.RunInstructionRange(ctx, g.captureEnd, g.plan.InstructionCount()); err != nil {
			return nil, fmt.Errorf("cuda graph replay: post-capture: %w", err)
		}
	}

	return g.plan.OutputTensor(), nil
}

// initEmbeddingCache identifies the EmbeddingLookup pre-capture instruction
// and caches its GPU tensor from gpuSlotCache. Called once after the first
// replay when EnsureSlotsGPU has populated the cache.
func (g *CUDAGraphExecutor[T]) initEmbeddingCache() {
	for i := 0; i < g.captureStart; i++ {
		if g.plan.InstructionOpName(i) == "EmbeddingLookup" {
			slotIdx := g.plan.InstructionOutputIdx(i)
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

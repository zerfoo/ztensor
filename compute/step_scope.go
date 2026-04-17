package compute

// StepScope marks the boundary of a single training step or inference
// pass. On Close the engine's arena pool is reset, reclaiming all
// intermediate allocations made inside the scope. Long-lived
// allocations made before the scope opened are preserved via the
// arena reset floor (see GPUEngine.MarkStepBoundary).
//
// A StepScope is safe to Close multiple times; the second and later
// calls are no-ops. This lets defer compose with explicit early-exit
// cleanup without double-reset.
//
// Typical training loop:
//
//	// After weight upload + optimizer state materialization:
//	engine.MarkStepBoundary()
//
//	for _, batch := range batches {
//	    scope := engine.BeginStep()
//	    // forward, backward, optimizer.Step ...
//	    scope.Close()
//	}
//
// Or with defer:
//
//	for _, batch := range batches {
//	    func() {
//	        scope := engine.BeginStep()
//	        defer scope.Close()
//	        // forward, backward, optimizer.Step ...
//	    }()
//	}
//
// BeginStep and MarkStepBoundary are no-ops on engines whose pool is
// not arena-backed (e.g. CPUEngine), so callers can use this pattern
// uniformly across engine types.
type StepScope struct {
	reset  func()
	closed bool
}

// Close resets the arena pool for the engine this scope was opened on.
// Idempotent: repeated calls after the first have no effect.
func (s *StepScope) Close() {
	if s == nil || s.closed {
		return
	}
	s.closed = true
	if s.reset != nil {
		s.reset()
	}
}

// BeginStep returns a StepScope bound to this engine. Close the scope
// (typically via defer) at the end of the training step to release
// intermediate tensors.
//
// Long-lived allocations (weights, optimizer state) are preserved if
// the caller invoked MarkStepBoundary after those allocations
// completed but before the first BeginStep.
func (e *GPUEngine[T]) BeginStep() *StepScope {
	return &StepScope{reset: e.ResetPool}
}

// MarkStepBoundary records the current arena offset as the reset
// floor. All subsequent StepScope.Close / ResetPool calls rewind only
// to this offset, preserving allocations made up to this point.
//
// Call once after:
//   - UploadWeights has completed, and
//   - any permanent per-training-run state (optimizer m/v tensors,
//     persistent scratch buffers) has been materialized on the engine.
//
// Safe to call on non-arena pools; it is a no-op in that case.
func (e *GPUEngine[T]) MarkStepBoundary() {
	e.SetArenaResetFloor(e.ArenaUsedBytes())
}

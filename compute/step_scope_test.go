package compute

import (
	"sync/atomic"
	"testing"
)

// TestStepScope_CloseInvokesReset verifies that closing a scope runs
// the captured reset closure exactly once.
func TestStepScope_CloseInvokesReset(t *testing.T) {
	var calls atomic.Int32
	s := &StepScope{reset: func() { calls.Add(1) }}

	s.Close()

	if got := calls.Load(); got != 1 {
		t.Fatalf("reset called %d times, want 1", got)
	}
}

// TestStepScope_CloseIsIdempotent verifies that calling Close twice
// still only resets once. This lets callers defer Close and also
// Close explicitly at a control-flow exit without double-reset.
func TestStepScope_CloseIsIdempotent(t *testing.T) {
	var calls atomic.Int32
	s := &StepScope{reset: func() { calls.Add(1) }}

	s.Close()
	s.Close()
	s.Close()

	if got := calls.Load(); got != 1 {
		t.Fatalf("reset called %d times, want 1", got)
	}
}

// TestStepScope_NilCloseNoPanic verifies that a nil scope's Close is a
// safe no-op. This guards against panics in code paths that short-
// circuit before BeginStep returns a non-nil scope.
func TestStepScope_NilCloseNoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("nil scope Close panicked: %v", r)
		}
	}()
	var s *StepScope
	s.Close()
}

// TestStepScope_ZeroValueCloseNoPanic verifies that a zero-value scope
// (no reset closure attached) closes without panicking.
func TestStepScope_ZeroValueCloseNoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("zero-value scope Close panicked: %v", r)
		}
	}()
	var s StepScope
	s.Close()
}

// TestStepScope_TrainingLoopPattern exercises the documented idiom
// across many iterations to confirm that per-step resets stay bounded
// while intermediate allocations would otherwise grow unboundedly.
//
// This is the regression test for the arena-exhaustion hang observed
// when Wolf's CrossAsset training ran ~132 batches on GB10 without
// resetting the GPU engine's arena pool between steps. The symptom on
// hardware was the CUDA driver's async stream saturating and new
// allocations blocking in cudaMalloc after the 2GB arena filled.
//
// The test does not depend on real CUDA: it models the arena with a
// counter that Reset zeroes. Without Reset the counter grows linearly
// in the number of allocations; with per-step Reset it stays bounded
// by the per-step allocation count regardless of step count.
func TestStepScope_TrainingLoopPattern(t *testing.T) {
	const (
		steps         = 200
		allocsPerStep = 500
	)

	// offset simulates the arena bump-allocator offset. Reset rewinds to
	// zero; Alloc advances the offset. peakWithoutReset captures how
	// large the offset would grow across the whole training loop if
	// nobody reset between steps.
	var (
		offset            int
		peakWithoutReset  int
		peakWithReset     int
		resetCalls        int
	)

	reset := func() {
		resetCalls++
		offset = 0
	}
	alloc := func() {
		offset++ // each alloc advances the bump pointer
		if offset > peakWithReset {
			peakWithReset = offset
		}
	}

	// With per-step StepScope: arena offset is bounded by allocsPerStep.
	for step := 0; step < steps; step++ {
		func() {
			scope := &StepScope{reset: reset}
			defer scope.Close()
			for i := 0; i < allocsPerStep; i++ {
				alloc()
			}
		}()
	}

	if peakWithReset != allocsPerStep {
		t.Errorf("with StepScope: peak offset = %d, want %d (bounded by allocsPerStep)",
			peakWithReset, allocsPerStep)
	}
	if resetCalls != steps {
		t.Errorf("with StepScope: reset called %d times, want %d", resetCalls, steps)
	}

	// Now the contrast: without any StepScope the offset grows unbounded.
	offset = 0
	for step := 0; step < steps; step++ {
		for i := 0; i < allocsPerStep; i++ {
			offset++
			if offset > peakWithoutReset {
				peakWithoutReset = offset
			}
		}
	}

	if peakWithoutReset != steps*allocsPerStep {
		t.Errorf("without StepScope: peak offset = %d, want %d",
			peakWithoutReset, steps*allocsPerStep)
	}

	// Sanity: the ratio is what matters for the bug. Without reset the
	// arena would need to be steps-times larger to hold all per-step
	// intermediates. On GB10 with a 2GB arena, this is what caused the
	// cudaMalloc fallback stall around batch ~130.
	if peakWithoutReset <= peakWithReset {
		t.Errorf("without-reset peak (%d) must exceed with-reset peak (%d) "+
			"for the regression pattern to be reproduced",
			peakWithoutReset, peakWithReset)
	}
}

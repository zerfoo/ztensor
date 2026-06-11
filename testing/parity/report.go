package parity

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/zerfoo/ztensor/testing/oracle"
)

// OpResult is one op's verdict under one schedule, in the report style of
// the oracle harness (per-tensor max_abs/max_rel + pass).
type OpResult struct {
	Op       string `json:"op"`
	Schedule string `json:"schedule"`
	// Tolerance is the per-op bar the diffs were judged against.
	Tolerance oracle.Tolerance `json:"tolerance"`
	// Forward diffs the candidate forward output against the reference.
	Forward *oracle.DiffStats `json:"forward,omitempty"`
	// InputGrads diff the candidate input gradients, aligned with the op's
	// inputs.
	InputGrads []oracle.DiffStats `json:"input_grads,omitempty"`
	// ParamGrads diff the candidate parameter gradients by parameter name.
	ParamGrads map[string]oracle.DiffStats `json:"param_grads,omitempty"`
	// Error records a side failing Forward/Backward (or shape disagreement)
	// instead of producing comparable values. A NaN read from a recycled
	// arena buffer is NOT an error: it surfaces as a failing diff with
	// max_abs=+Inf, attributed to this op and schedule.
	Error string `json:"error,omitempty"`
	Pass  bool   `json:"pass"`
}

// String renders one PASS/FAIL line for log scraping (the Spark pod greps
// these, mirroring run_oracle.py's per-op lines).
func (r OpResult) String() string {
	status := "PASS"
	if !r.Pass {
		status = "FAIL"
	}
	if r.Error != "" {
		return fmt.Sprintf("parity %-5s %-22s schedule=%s error=%s", status, r.Op, r.Schedule, r.Error)
	}
	maxGrad := 0.0
	for _, d := range r.InputGrads {
		if d.MaxAbs > maxGrad {
			maxGrad = d.MaxAbs
		}
	}
	for _, d := range r.ParamGrads {
		if d.MaxAbs > maxGrad {
			maxGrad = d.MaxAbs
		}
	}
	return fmt.Sprintf("parity %-5s %-22s schedule=%s fwd max_abs=%.3e max_rel=%.3e bwd max_abs=%.3e",
		status, r.Op, r.Schedule, r.Forward.MaxAbs, r.Forward.MaxRel, maxGrad)
}

// Report is one (reference, candidate, schedule) run: the JSON artifact for
// the devlog, mirroring run_oracle.py's report.json shape
// (results + passed/failed/errored totals).
type Report struct {
	Reference string     `json:"reference"`
	Candidate string     `json:"candidate"`
	Schedule  string     `json:"schedule"`
	Results   []OpResult `json:"results"`
	Passed    int        `json:"passed"`
	Failed    int        `json:"failed"`
	Errored   int        `json:"errored"`
	Pass      bool       `json:"pass"`
}

// Result returns the result for the named op, or nil.
func (r *Report) Result(op string) *OpResult {
	for i := range r.Results {
		if r.Results[i].Op == op {
			return &r.Results[i]
		}
	}
	return nil
}

// WriteJSON writes the report to path with a trailing newline.
func (r *Report) WriteJSON(path string) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("parity: marshaling report: %w", err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
		return fmt.Errorf("parity: writing report: %w", err)
	}
	return nil
}

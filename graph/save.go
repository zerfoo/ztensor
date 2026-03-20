package graph

import (
	"encoding/json"
	"fmt"
	"os"
)

// SaveParameters serializes all graph parameters to a JSON file.
// Parameter values are converted to float64 for JSON compatibility.
func (g *Graph[T]) SaveParameters(path string) error {
	params := g.Parameters()
	data := make(map[string][]float64, len(params))
	for _, p := range params {
		vals := p.Value.Data()
		f64 := make([]float64, len(vals))
		for i, v := range vals {
			f64[i] = float64(v)
		}
		data[p.Name] = f64
	}

	b, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("marshal parameters: %w", err)
	}
	if err := os.WriteFile(path, b, 0644); err != nil {
		return fmt.Errorf("write parameters: %w", err)
	}
	return nil
}

// LoadParametersFromFile reads parameter values from a JSON file and loads
// them into the graph. The file must have been created by SaveParameters.
func (g *Graph[T]) LoadParametersFromFile(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read parameters: %w", err)
	}

	var data map[string][]float64
	if err := json.Unmarshal(b, &data); err != nil {
		return fmt.Errorf("unmarshal parameters: %w", err)
	}

	params := make(map[string][]T, len(data))
	for name, f64 := range data {
		vals := make([]T, len(f64))
		for i, v := range f64 {
			vals[i] = T(v)
		}
		params[name] = vals
	}
	return g.LoadParameters(params)
}

// checkpointData is the on-disk format for SaveCheckpoint / LoadCheckpoint.
type checkpointData struct {
	Parameters map[string][]float64       `json:"parameters"`
	Optimizer  map[string]json.RawMessage `json:"optimizer"`
	Version    int                        `json:"version"`
}

// SaveCheckpoint serializes graph parameters and optimizer state to a JSON file.
func (g *Graph[T]) SaveCheckpoint(path string, optimizerState map[string]interface{}) error {
	params := g.Parameters()
	paramData := make(map[string][]float64, len(params))
	for _, p := range params {
		vals := p.Value.Data()
		f64 := make([]float64, len(vals))
		for i, v := range vals {
			f64[i] = float64(v)
		}
		paramData[p.Name] = f64
	}

	optData := make(map[string]json.RawMessage, len(optimizerState))
	for k, v := range optimizerState {
		raw, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("marshal optimizer state %q: %w", k, err)
		}
		optData[k] = raw
	}

	cp := checkpointData{
		Parameters: paramData,
		Optimizer:  optData,
		Version:    1,
	}

	b, err := json.Marshal(cp)
	if err != nil {
		return fmt.Errorf("marshal checkpoint: %w", err)
	}
	if err := os.WriteFile(path, b, 0644); err != nil {
		return fmt.Errorf("write checkpoint: %w", err)
	}
	return nil
}

// LoadCheckpoint restores graph parameters from a checkpoint file and returns
// the optimizer state. The file must have been created by SaveCheckpoint.
func (g *Graph[T]) LoadCheckpoint(path string) (map[string]interface{}, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read checkpoint: %w", err)
	}

	var cp checkpointData
	if err := json.Unmarshal(b, &cp); err != nil {
		return nil, fmt.Errorf("unmarshal checkpoint: %w", err)
	}

	params := make(map[string][]T, len(cp.Parameters))
	for name, f64 := range cp.Parameters {
		vals := make([]T, len(f64))
		for i, v := range f64 {
			vals[i] = T(v)
		}
		params[name] = vals
	}
	if err := g.LoadParameters(params); err != nil {
		return nil, err
	}

	optState := make(map[string]interface{}, len(cp.Optimizer))
	for k, raw := range cp.Optimizer {
		var v interface{}
		if err := json.Unmarshal(raw, &v); err != nil {
			return nil, fmt.Errorf("unmarshal optimizer state %q: %w", k, err)
		}
		optState[k] = v
	}
	return optState, nil
}

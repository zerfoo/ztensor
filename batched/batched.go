// Package batched provides batched multi-model inference, enabling 1000+
// per-source models that share the same architecture to run in a single
// batched GEMM call rather than N sequential matrix multiplications.
package batched

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// ActivationType identifies a supported activation function.
type ActivationType int

const (
	// ActivationNone applies no activation (identity).
	ActivationNone ActivationType = iota
	// ActivationReLU applies max(0, x).
	ActivationReLU
	// ActivationTanh applies the hyperbolic tangent function.
	ActivationTanh
)

// LayerSpec describes one fully-connected layer in the shared architecture.
type LayerSpec struct {
	InputSize  int
	OutputSize int
	Activation ActivationType
}

// Architecture describes the shared model architecture. Every model in a
// batch must conform to the same architecture (layer sizes and activations).
type Architecture struct {
	Layers []LayerSpec
}

// Validate checks that the architecture is well-formed.
func (a Architecture) Validate() error {
	if len(a.Layers) == 0 {
		return errors.New("architecture must have at least one layer")
	}
	for i, l := range a.Layers {
		if l.InputSize <= 0 || l.OutputSize <= 0 {
			return fmt.Errorf("layer %d: input and output sizes must be positive", i)
		}
		if i > 0 && a.Layers[i-1].OutputSize != l.InputSize {
			return fmt.Errorf("layer %d: input size %d does not match previous output size %d",
				i, l.InputSize, a.Layers[i-1].OutputSize)
		}
	}
	return nil
}

// totalWeights returns the total number of float32 parameters across all layers.
func (a Architecture) totalWeights() int {
	n := 0
	for _, l := range a.Layers {
		n += l.InputSize * l.OutputSize
	}
	return n
}

// BatchedWeights holds weight tensors for N models that share the same
// architecture but have different learned parameters. Weights are stored
// contiguously with a batch dimension so that a single batched GEMM can
// process all models at once.
type BatchedWeights struct {
	numModels    int
	architecture Architecture
	// weights[layerIdx] is a tensor of shape [numModels, inputSize, outputSize].
	weights []*tensor.TensorNumeric[float32]
	loaded  []bool
}

// NewBatchedWeights allocates contiguous weight storage for numModels models.
func NewBatchedWeights(numModels int, arch Architecture) (*BatchedWeights, error) {
	if numModels <= 0 {
		return nil, errors.New("numModels must be positive")
	}
	if err := arch.Validate(); err != nil {
		return nil, fmt.Errorf("invalid architecture: %w", err)
	}
	bw := &BatchedWeights{
		numModels:    numModels,
		architecture: arch,
		weights:      make([]*tensor.TensorNumeric[float32], len(arch.Layers)),
		loaded:       make([]bool, numModels),
	}
	for i, l := range arch.Layers {
		data := make([]float32, numModels*l.InputSize*l.OutputSize)
		t, err := tensor.New[float32]([]int{numModels, l.InputSize, l.OutputSize}, data)
		if err != nil {
			return nil, fmt.Errorf("allocating layer %d weights: %w", i, err)
		}
		bw.weights[i] = t
	}
	return bw, nil
}

// BatchedInference runs forward passes for all models in parallel using
// batched GEMM operations through the compute.Engine[float32] interface.
type BatchedInference struct {
	mu      sync.RWMutex
	weights *BatchedWeights
	engine  compute.Engine[float32]
}

// NewBatchedInference creates a new BatchedInference for numModels models
// sharing the given architecture. All tensor arithmetic flows through engine.
func NewBatchedInference(numModels int, arch Architecture, engine compute.Engine[float32]) (*BatchedInference, error) {
	bw, err := NewBatchedWeights(numModels, arch)
	if err != nil {
		return nil, err
	}
	return &BatchedInference{
		weights: bw,
		engine:  engine,
	}, nil
}

// LoadWeights loads weight parameters for a single model identified by
// modelIdx. The weights map is keyed by "layer_<N>" and each value must
// be a row-major float32 slice of size InputSize*OutputSize.
func (bi *BatchedInference) LoadWeights(modelIdx int, weights map[string][]float32) error {
	bi.mu.Lock()
	defer bi.mu.Unlock()

	if modelIdx < 0 || modelIdx >= bi.weights.numModels {
		return fmt.Errorf("modelIdx %d out of range [0, %d)", modelIdx, bi.weights.numModels)
	}
	arch := bi.weights.architecture
	for i, l := range arch.Layers {
		key := fmt.Sprintf("layer_%d", i)
		w, ok := weights[key]
		if !ok {
			return fmt.Errorf("missing weights for %s", key)
		}
		expected := l.InputSize * l.OutputSize
		if len(w) != expected {
			return fmt.Errorf("%s: expected %d weights, got %d", key, expected, len(w))
		}
		// Copy into the contiguous batch tensor at offset modelIdx.
		data := bi.weights.weights[i].Data()
		offset := modelIdx * expected
		copy(data[offset:offset+expected], w)
	}
	bi.weights.loaded[modelIdx] = true
	return nil
}

// Forward runs all models in parallel via batched GEMM. inputs[i] is the
// input vector for model i. All inputs must have length equal to the first
// layer's InputSize. Returns one output vector per model.
func (bi *BatchedInference) Forward(inputs [][]float32) ([][]float32, error) {
	bi.mu.RLock()
	defer bi.mu.RUnlock()

	numModels := bi.weights.numModels
	if len(inputs) != numModels {
		return nil, fmt.Errorf("expected %d inputs, got %d", numModels, len(inputs))
	}

	arch := bi.weights.architecture
	inputSize := arch.Layers[0].InputSize

	// Validate and build batched input tensor [numModels, 1, inputSize].
	flatInput := make([]float32, 0, numModels*inputSize)
	for i, inp := range inputs {
		if len(inp) != inputSize {
			return nil, fmt.Errorf("input %d: expected length %d, got %d", i, inputSize, len(inp))
		}
		flatInput = append(flatInput, inp...)
	}

	// Shape: [numModels, 1, inputSize] — each model's input is a 1-row matrix.
	activations, err := tensor.New[float32]([]int{numModels, 1, inputSize}, flatInput)
	if err != nil {
		return nil, fmt.Errorf("creating input tensor: %w", err)
	}

	ctx := context.Background()

	// Forward through each layer: activations = activations @ weights[layer]
	for i, l := range arch.Layers {
		// activations: [numModels, 1, l.InputSize]
		// weights[i]:  [numModels, l.InputSize, l.OutputSize]
		// result:      [numModels, 1, l.OutputSize]
		result, err := bi.engine.MatMul(ctx, activations, bi.weights.weights[i])
		if err != nil {
			return nil, fmt.Errorf("layer %d matmul: %w", i, err)
		}

		// Apply activation function.
		result, err = bi.applyActivation(ctx, result, l.Activation)
		if err != nil {
			return nil, fmt.Errorf("layer %d activation: %w", i, err)
		}

		activations = result
	}

	// Extract per-model outputs.
	lastOutputSize := arch.Layers[len(arch.Layers)-1].OutputSize
	data := activations.Data()
	outputs := make([][]float32, numModels)
	for i := range numModels {
		out := make([]float32, lastOutputSize)
		copy(out, data[i*lastOutputSize:(i+1)*lastOutputSize])
		outputs[i] = out
	}

	return outputs, nil
}

// NumModels returns the number of models in the batch.
func (bi *BatchedInference) NumModels() int {
	return bi.weights.numModels
}

// Architecture returns the shared architecture.
func (bi *BatchedInference) Architecture() Architecture {
	return bi.weights.architecture
}

// applyActivation applies the given activation function element-wise.
func (bi *BatchedInference) applyActivation(ctx context.Context, t *tensor.TensorNumeric[float32], act ActivationType) (*tensor.TensorNumeric[float32], error) {
	switch act {
	case ActivationNone:
		return t, nil
	case ActivationReLU:
		ops := bi.engine.Ops()
		return bi.engine.UnaryOp(ctx, t, func(v float32) float32 {
			return ops.ReLU(v)
		})
	case ActivationTanh:
		return bi.engine.Tanh(ctx, t)
	default:
		return nil, fmt.Errorf("unsupported activation type: %d", act)
	}
}

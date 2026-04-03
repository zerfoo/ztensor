package graph

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/internal/pjrt"
	"github.com/zerfoo/ztensor/internal/stablehlo"
	"github.com/zerfoo/ztensor/tensor"
)

// PJRTPlan holds a compiled PJRT program ready for execution. It wraps the
// compiled executable(s), device-resident weight buffers, and KV cache
// metadata needed to run inference through the PJRT backend.
//
// KV cache buffers are nil initially and populated after the first RunPrefill
// call. Each RunDecode call donates the current KV buffers and replaces them
// with the updated outputs from the executable.
type PJRTPlan[T tensor.Numeric] struct {
	// PrefillExec is the compiled executable for the prefill phase.
	// For graphs without KV cache, this is the only executable.
	PrefillExec *pjrt.LoadedExecutable

	// DecodeExec is the compiled executable for the decode phase.
	// Nil for graphs without KV cache.
	DecodeExec *pjrt.LoadedExecutable

	// Client is the PJRT client used for buffer management and execution.
	Client *pjrt.Client

	// WeightBuffers holds device-resident weight tensors, ordered to match
	// the frozen slot positions in the compiled function signature.
	WeightBuffers []*pjrt.Buffer

	// KVSlots describes the KV cache slots for stateful execution.
	// Empty for graphs without KV cache.
	KVSlots []stablehlo.KVCacheSlot

	// kvBuffers holds persistent KV cache buffers between execution calls.
	// Nil until the first RunPrefill populates them. Each RunDecode donates
	// these buffers and replaces them with updated outputs.
	kvBuffers []*pjrt.Buffer

	// InputSlots are the slot indices that receive graph inputs (non-frozen,
	// non-KV-cache). Ordered to match the function signature.
	InputSlots []int

	// OutputSlot is the slot index that produces the primary output.
	OutputSlot int

	// SlotShapes maps slot index to tensor shape.
	SlotShapes map[int][]int

	// Dtype is the MLIR dtype string (e.g. "f32") for this plan.
	Dtype string

	// FrozenSlots are the slot indices holding frozen (weight) tensors,
	// ordered to match the compiled function signature.
	FrozenSlots []int
}

// HasKVCache returns true if this plan has KV cache state.
func (p *PJRTPlan[T]) HasKVCache() bool {
	return len(p.KVSlots) > 0
}

// RunPrefill executes the prefill program with the given input tensors,
// stores the resulting KV cache buffers, and returns the logits tensor.
//
// inputs maps graph slot indices to their corresponding Go tensors. The
// typical prefill input is the token ID tensor for the full prompt.
//
// After RunPrefill, the plan holds KV cache buffers for subsequent decode
// steps. Call Reset() before starting a new sequence.
func (p *PJRTPlan[T]) RunPrefill(ctx context.Context, inputs map[int]*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return p.execute(p.PrefillExec, inputs)
}

// RunDecode executes the decode program with the given single-token input,
// donating the current KV cache buffers and updating them with the new
// outputs. Returns the logits tensor for the decoded token.
//
// RunDecode must be called after RunPrefill. The KV buffers from the
// previous step are donated (the runtime may consume them to avoid copies)
// and replaced with the updated KV outputs.
func (p *PJRTPlan[T]) RunDecode(ctx context.Context, inputs map[int]*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return p.execute(p.DecodeExec, inputs)
}

// execute is the shared execution path for both prefill and decode.
// It assembles the full input buffer list (weights + user inputs + KV caches),
// calls the PJRT executable, extracts the logits output, and updates KV
// cache buffers from the remaining outputs.
func (p *PJRTPlan[T]) execute(exec *pjrt.LoadedExecutable, inputs map[int]*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if exec == nil {
		return nil, fmt.Errorf("pjrt_plan: executable is nil")
	}

	device, err := p.firstDevice()
	if err != nil {
		return nil, err
	}

	// Convert user input tensors to PJRT buffers.
	userBuffers := make([]*pjrt.Buffer, len(p.InputSlots))
	for i, slot := range p.InputSlots {
		t, ok := inputs[slot]
		if !ok {
			return nil, fmt.Errorf("pjrt_plan: missing input for slot %d", slot)
		}
		buf, err := pjrt.BufferFromHost(p.Client, t.Data(), t.Shape(), device)
		if err != nil {
			for j := 0; j < i; j++ {
				userBuffers[j].Close()
			}
			return nil, fmt.Errorf("pjrt_plan: buffer from host for slot %d: %w", slot, err)
		}
		userBuffers[i] = buf
	}

	// Assemble the full argument list: weights + user inputs + KV caches.
	// The compiled program expects arguments in this order (set by CompilePJRT).
	allInputs := make([]*pjrt.Buffer, 0, len(p.WeightBuffers)+len(userBuffers)+len(p.kvBuffers))
	allInputs = append(allInputs, p.WeightBuffers...)
	allInputs = append(allInputs, userBuffers...)

	// Mark KV cache buffers for donation so the runtime can reuse the memory.
	var execOpts []pjrt.ExecOption
	if len(p.kvBuffers) > 0 {
		allInputs = append(allInputs, p.kvBuffers...)
		donated := make([]bool, len(allInputs))
		kvStart := len(p.WeightBuffers) + len(userBuffers)
		for i := kvStart; i < len(donated); i++ {
			donated[i] = true
		}
		execOpts = append(execOpts, pjrt.WithInputDonation(donated))
	}

	outputs, err := exec.Execute(allInputs, execOpts...)
	if err != nil {
		for _, buf := range userBuffers {
			buf.Close()
		}
		return nil, fmt.Errorf("pjrt_plan: execute: %w", err)
	}

	// Clean up user input buffers (they were copied, not donated).
	for _, buf := range userBuffers {
		buf.Close()
	}

	if len(outputs) == 0 {
		return nil, fmt.Errorf("pjrt_plan: execution produced no outputs")
	}

	// The program returns: [logits, kv_cache_0, kv_cache_1, ...].
	// First output is always the logits tensor.
	logitsBuf := outputs[0]

	// Update KV cache buffers from the remaining outputs.
	// After donation, the old kvBuffers are invalid — replace with new ones.
	if len(p.KVSlots) > 0 {
		newKV := make([]*pjrt.Buffer, len(p.KVSlots))
		for i := range p.KVSlots {
			outputIdx := 1 + i
			if outputIdx >= len(outputs) {
				return nil, fmt.Errorf("pjrt_plan: expected KV output at index %d, got %d outputs", outputIdx, len(outputs))
			}
			newKV[i] = outputs[outputIdx]
		}
		p.kvBuffers = newKV
	}

	// Read logits from device to host.
	logitsShape, err := logitsBuf.Shape()
	if err != nil {
		return nil, fmt.Errorf("pjrt_plan: read logits shape: %w", err)
	}

	numElements := 1
	for _, d := range logitsShape {
		numElements *= d
	}

	data := make([]T, numElements)
	if err := pjrt.ToHostSlice(logitsBuf, data); err != nil {
		return nil, fmt.Errorf("pjrt_plan: read logits to host: %w", err)
	}
	logitsBuf.Close()

	result, err := tensor.New[T](logitsShape, data)
	if err != nil {
		return nil, fmt.Errorf("pjrt_plan: create logits tensor: %w", err)
	}
	return result, nil
}

// Reset destroys KV cache buffers, allowing a new generation sequence.
// Weight buffers and compiled executables are preserved.
func (p *PJRTPlan[T]) Reset() {
	for _, buf := range p.kvBuffers {
		if buf != nil {
			buf.Close()
		}
	}
	p.kvBuffers = nil
}

// Close releases all PJRT resources held by this plan: executables,
// weight buffers, and KV cache buffers. Safe to call multiple times.
func (p *PJRTPlan[T]) Close() error {
	p.Reset()

	var firstErr error

	if p.PrefillExec != nil {
		if err := p.PrefillExec.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		p.PrefillExec = nil
	}

	if p.DecodeExec != nil {
		if err := p.DecodeExec.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		p.DecodeExec = nil
	}

	for _, buf := range p.WeightBuffers {
		if buf != nil {
			if err := buf.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	p.WeightBuffers = nil

	return firstErr
}

// firstDevice returns the first addressable device from the client.
func (p *PJRTPlan[T]) firstDevice() (*pjrt.Device, error) {
	devices, err := p.Client.AddressableDevices()
	if err != nil {
		return nil, fmt.Errorf("pjrt_plan: list devices: %w", err)
	}
	if len(devices) == 0 {
		return nil, fmt.Errorf("pjrt_plan: no addressable devices")
	}
	return devices[0], nil
}


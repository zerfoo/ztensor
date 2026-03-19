package gpuapi

import (
	"fmt"
	"unsafe"
)

// MetalDNN implements the DNN interface for Metal.
// Metal Performance Shaders provides some DNN primitives, but comprehensive
// DNN support is not yet implemented. All operations return ErrNotSupported.
type MetalDNN struct{}

// NewMetalDNN returns a new Metal DNN stub.
func NewMetalDNN() *MetalDNN {
	return &MetalDNN{}
}

var errMetalDNNNotSupported = fmt.Errorf("DNN operations not yet implemented on Metal backend")

func (d *MetalDNN) SetStream(_ Stream) error { return nil }
func (d *MetalDNN) Destroy() error            { return nil }

func (d *MetalDNN) ConvForward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

func (d *MetalDNN) ConvBackwardData(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

func (d *MetalDNN) ConvBackwardFilter(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

func (d *MetalDNN) BatchNormForwardInference(
	_ unsafe.Pointer, _ [4]int,
	_, _, _, _ unsafe.Pointer,
	_ int,
	_ float64,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

func (d *MetalDNN) BatchNormForwardTraining(
	_ unsafe.Pointer, _ [4]int,
	_, _ unsafe.Pointer,
	_ int,
	_, _ float64,
	_, _ unsafe.Pointer,
	_, _ unsafe.Pointer,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

func (d *MetalDNN) BatchNormBackward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ unsafe.Pointer,
	_ int,
	_, _ unsafe.Pointer,
	_, _, _ unsafe.Pointer,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

func (d *MetalDNN) ActivationForward(
	_ ActivationMode,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

func (d *MetalDNN) ActivationBackward(
	_ ActivationMode,
	_ unsafe.Pointer, _ unsafe.Pointer,
	_ unsafe.Pointer, _ unsafe.Pointer,
	_ [4]int,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

func (d *MetalDNN) PoolingForward(
	_ PoolingMode,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _, _, _, _ int,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

func (d *MetalDNN) PoolingBackward(
	_ PoolingMode,
	_ unsafe.Pointer, _ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ unsafe.Pointer, _ [4]int,
	_, _, _, _, _, _ int,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

func (d *MetalDNN) SoftmaxForward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

func (d *MetalDNN) AddTensor(
	_ float32,
	_ unsafe.Pointer, _ [4]int,
	_ float32,
	_ unsafe.Pointer, _ [4]int,
	_ Stream,
) error {
	return errMetalDNNNotSupported
}

// Compile-time interface assertion.
var _ DNN = (*MetalDNN)(nil)

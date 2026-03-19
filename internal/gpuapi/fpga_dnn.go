package gpuapi

import (
	"fmt"
	"unsafe"
)

// FPGADnn implements the DNN interface for FPGA accelerators.
// DNN operations are not yet implemented on FPGA; all return ErrNotSupported.
type FPGADnn struct{}

// NewFPGADnn returns a new FPGA DNN stub.
func NewFPGADnn() *FPGADnn {
	return &FPGADnn{}
}

var errFPGADNNNotSupported = fmt.Errorf("DNN operations not yet implemented on FPGA backend")

func (d *FPGADnn) SetStream(_ Stream) error { return nil }
func (d *FPGADnn) Destroy() error            { return nil }

func (d *FPGADnn) ConvForward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

func (d *FPGADnn) ConvBackwardData(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

func (d *FPGADnn) ConvBackwardFilter(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

func (d *FPGADnn) BatchNormForwardInference(
	_ unsafe.Pointer, _ [4]int,
	_, _, _, _ unsafe.Pointer,
	_ int,
	_ float64,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

func (d *FPGADnn) BatchNormForwardTraining(
	_ unsafe.Pointer, _ [4]int,
	_, _ unsafe.Pointer,
	_ int,
	_, _ float64,
	_, _ unsafe.Pointer,
	_, _ unsafe.Pointer,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

func (d *FPGADnn) BatchNormBackward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ unsafe.Pointer,
	_ int,
	_, _ unsafe.Pointer,
	_, _, _ unsafe.Pointer,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

func (d *FPGADnn) ActivationForward(
	_ ActivationMode,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

func (d *FPGADnn) ActivationBackward(
	_ ActivationMode,
	_ unsafe.Pointer, _ unsafe.Pointer,
	_ unsafe.Pointer, _ unsafe.Pointer,
	_ [4]int,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

func (d *FPGADnn) PoolingForward(
	_ PoolingMode,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _, _, _, _ int,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

func (d *FPGADnn) PoolingBackward(
	_ PoolingMode,
	_ unsafe.Pointer, _ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ unsafe.Pointer, _ [4]int,
	_, _, _, _, _, _ int,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

func (d *FPGADnn) SoftmaxForward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

func (d *FPGADnn) AddTensor(
	_ float32,
	_ unsafe.Pointer, _ [4]int,
	_ float32,
	_ unsafe.Pointer, _ [4]int,
	_ Stream,
) error {
	return errFPGADNNNotSupported
}

// Compile-time interface assertion.
var _ DNN = (*FPGADnn)(nil)

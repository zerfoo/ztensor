package gpuapi

import (
	"fmt"
	"unsafe"
)

// SYCLDnn implements the DNN interface for SYCL devices.
// DNN operations are delegated to oneDNN when available; all return
// not-supported until oneDNN bindings are added.
type SYCLDnn struct{}

// NewSYCLDnn returns a new SYCL DNN stub.
func NewSYCLDnn() *SYCLDnn {
	return &SYCLDnn{}
}

var errSYCLDNNNotSupported = fmt.Errorf("DNN operations not yet implemented on SYCL backend")

func (d *SYCLDnn) SetStream(_ Stream) error { return nil }
func (d *SYCLDnn) Destroy() error            { return nil }

func (d *SYCLDnn) ConvForward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

func (d *SYCLDnn) ConvBackwardData(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

func (d *SYCLDnn) ConvBackwardFilter(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

func (d *SYCLDnn) BatchNormForwardInference(
	_ unsafe.Pointer, _ [4]int,
	_, _, _, _ unsafe.Pointer,
	_ int,
	_ float64,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

func (d *SYCLDnn) BatchNormForwardTraining(
	_ unsafe.Pointer, _ [4]int,
	_, _ unsafe.Pointer,
	_ int,
	_, _ float64,
	_, _ unsafe.Pointer,
	_, _ unsafe.Pointer,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

func (d *SYCLDnn) BatchNormBackward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ unsafe.Pointer,
	_ int,
	_, _ unsafe.Pointer,
	_, _, _ unsafe.Pointer,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

func (d *SYCLDnn) ActivationForward(
	_ ActivationMode,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

func (d *SYCLDnn) ActivationBackward(
	_ ActivationMode,
	_ unsafe.Pointer, _ unsafe.Pointer,
	_ unsafe.Pointer, _ unsafe.Pointer,
	_ [4]int,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

func (d *SYCLDnn) PoolingForward(
	_ PoolingMode,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _, _, _, _ int,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

func (d *SYCLDnn) PoolingBackward(
	_ PoolingMode,
	_ unsafe.Pointer, _ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ unsafe.Pointer, _ [4]int,
	_, _, _, _, _, _ int,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

func (d *SYCLDnn) SoftmaxForward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

func (d *SYCLDnn) AddTensor(
	_ float32,
	_ unsafe.Pointer, _ [4]int,
	_ float32,
	_ unsafe.Pointer, _ [4]int,
	_ Stream,
) error {
	return errSYCLDNNNotSupported
}

// Compile-time interface assertion.
var _ DNN = (*SYCLDnn)(nil)

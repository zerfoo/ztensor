package compute

import (
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
	"github.com/zerfoo/ztensor/internal/cuda/kernels"
	"github.com/zerfoo/ztensor/tensor"
)

// TernaryGEMVGPU performs ternary GEMV on GPU if CUDA is available,
// falling back to the CPU implementation otherwise.
//
// When GPU dispatch succeeds, packed ternary weights and the input vector
// are uploaded to the device, the kernel runs, and results are copied back.
// On any failure (CUDA unavailable, allocation error, kernel error), the
// CPU path is used transparently.
func TernaryGEMVGPU(weights *tensor.TernaryStorage, x []float32, rows, cols int) []float32 {
	if cuda.Available() {
		if y, err := ternaryGEMVCUDA(weights, x, rows, cols); err == nil {
			return y
		}
	}
	return TernaryGEMV(weights, x, rows, cols)
}

// ternaryGEMVCUDA runs the ternary GEMV kernel on GPU.
func ternaryGEMVCUDA(weights *tensor.TernaryStorage, x []float32, rows, cols int) ([]float32, error) {
	rawBytes := weights.RawBytes()
	bytesPerRow := (cols + 3) / 4
	totalWeightBytes := rows * bytesPerRow

	// Allocate device memory.
	dW, err := cuda.Malloc(totalWeightBytes)
	if err != nil {
		return nil, err
	}
	defer cuda.Free(dW) //nolint:errcheck

	dX, err := cuda.Malloc(cols * 4) // float32
	if err != nil {
		return nil, err
	}
	defer cuda.Free(dX) //nolint:errcheck

	dY, err := cuda.Malloc(rows * 4) // float32
	if err != nil {
		return nil, err
	}
	defer cuda.Free(dY) //nolint:errcheck

	// Upload weights. The CPU packing stores values contiguously across rows
	// (element i*cols+j at bit offset (i*cols+j)*2), but the kernel expects
	// each row to start at a fresh byte boundary. Repack into per-row layout.
	packed := make([]byte, totalWeightBytes)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			srcIdx := i*cols + j
			srcByte := srcIdx / 4
			srcShift := uint(srcIdx%4) * 2
			bits := (rawBytes[srcByte] >> srcShift) & 0x03

			dstIdx := i*bytesPerRow + j/4
			dstShift := uint(j%4) * 2
			packed[dstIdx] |= bits << dstShift
		}
	}

	if err := cuda.Memcpy(dW, unsafe.Pointer(&packed[0]), totalWeightBytes, cuda.MemcpyHostToDevice); err != nil {
		return nil, err
	}
	if err := cuda.Memcpy(dX, unsafe.Pointer(&x[0]), cols*4, cuda.MemcpyHostToDevice); err != nil {
		return nil, err
	}

	// Launch kernel.
	if err := kernels.TernaryGemvF32(
		dW, dX, dY,
		rows, cols, nil, // nil stream = default
	); err != nil {
		return nil, err
	}

	// Copy result back.
	y := make([]float32, rows)
	if err := cuda.Memcpy(unsafe.Pointer(&y[0]), dY, rows*4, cuda.MemcpyDeviceToHost); err != nil {
		return nil, err
	}

	return y, nil
}

package compute

import "github.com/zerfoo/ztensor/tensor"

// TernaryGEMV performs matrix-vector multiplication where the weight matrix
// is stored in packed ternary format {-1, 0, 1}. For each output element:
//
//	y[i] = sum of x[j] where w[i,j]=1, minus sum of x[j] where w[i,j]=-1
//
// No floating-point multiply is needed — only additions and subtractions.
// weights is a row-major matrix of shape [rows, cols], x has length cols.
func TernaryGEMV(weights *tensor.TernaryStorage, x []float32, rows, cols int) []float32 {
	y := make([]float32, rows)
	rawBytes := weights.RawBytes()

	for i := 0; i < rows; i++ {
		var sum float32
		base := i * cols
		j := 0

		byteStart := base / 4
		startShift := uint(base%4) * 2

		if startShift == 0 {
			// Row is byte-aligned: fast path processing 4 elements per byte
			// using a precomputed lookup table.
			fullBytes := cols / 4
			for bi := 0; bi < fullBytes; bi++ {
				b := rawBytes[byteStart+bi]
				lut := &ternaryLUT[b]
				sum += lut[0] * x[j]
				sum += lut[1] * x[j+1]
				sum += lut[2] * x[j+2]
				sum += lut[3] * x[j+3]
				j += 4
			}
			// Remaining elements.
			for j < cols {
				idx := base + j
				bits := (rawBytes[idx/4] >> (uint(idx%4) * 2)) & 0x03
				if bits == 0 {
					sum -= x[j]
				} else if bits == 2 {
					sum += x[j]
				}
				j++
			}
		} else {
			// Row is not byte-aligned: per-element extraction.
			for j < cols {
				idx := base + j
				bits := (rawBytes[idx/4] >> (uint(idx%4) * 2)) & 0x03
				if bits == 0 {
					sum -= x[j]
				} else if bits == 2 {
					sum += x[j]
				}
				j++
			}
		}
		y[i] = sum
	}
	return y
}

// ternaryLUT maps each packed byte to its 4 decoded float32 values (-1, 0, 1).
// Encoding: 00=-1, 01=0, 10=1.
var ternaryLUT [256][4]float32

func init() {
	for b := 0; b < 256; b++ {
		for k := 0; k < 4; k++ {
			bits := (b >> (uint(k) * 2)) & 0x03
			ternaryLUT[b][k] = float32(int8(bits) - 1)
		}
	}
}

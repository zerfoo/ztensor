package tensor

// iq3sGrid is the IQ3_S codebook: 512 entries mapping 9-bit grid indices
// to 4 unsigned magnitude values from {1, 3, 5, 7}.
//
// Each 8-bit qs byte encodes 4 values using 2 bits each:
//   bits [1:0] -> val0, [3:2] -> val1, [5:4] -> val2, [7:6] -> val3
//   2-bit code mapping: 0->1, 1->3, 2->5, 3->7
//
// The 9th bit (from qh) selects between two halves of the grid.
// Both halves use the same byte-to-values mapping.
// Signs are applied separately via the sign bytes in the block.
//
// Reference: llama.cpp iq3s_grid in ggml-quants.c
var iq3sGrid [512][4]uint8

func init() {
	for i := range 256 {
		iq3sGrid[i] = [4]uint8{
			uint8(1 + 2*((i>>0)&3)),
			uint8(1 + 2*((i>>2)&3)),
			uint8(1 + 2*((i>>4)&3)),
			uint8(1 + 2*((i>>6)&3)),
		}
		iq3sGrid[256+i] = iq3sGrid[i]
	}
}

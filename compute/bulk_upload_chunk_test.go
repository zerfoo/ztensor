package compute

import (
	"reflect"
	"testing"
)

// TestBulkUploadChunkRanges_Tiling verifies the chunk splitter exactly tiles
// the input (no gaps, no overlaps) and respects both the byte and tensor caps,
// which is the correctness-critical part of the GB10 wedge fix (ztensor#106).
func TestBulkUploadChunkRanges_Tiling(t *testing.T) {
	const elemSize = 4

	cases := []struct {
		name       string
		nelems     []int
		maxBytes   int
		maxTensors int
		want       [][2]int
	}{
		{"empty", nil, 64, 8, [][2]int{}},
		{"single", []int{10}, 64, 8, [][2]int{{0, 1}}},
		{"all-fit-one-chunk", []int{1, 1, 1, 1}, 1 << 20, 1024, [][2]int{{0, 4}}},
		{
			// 4 tensors x 4 elems x 4 bytes = 16 bytes each; cap 32 bytes -> 2 per chunk.
			"byte-cap-splits", []int{4, 4, 4, 4}, 32, 1024,
			[][2]int{{0, 2}, {2, 4}},
		},
		{
			"tensor-cap-splits", []int{1, 1, 1, 1, 1}, 1 << 20, 2,
			[][2]int{{0, 2}, {2, 4}, {4, 5}},
		},
		{
			// Middle tensor alone exceeds the byte cap: it must still get its
			// own range, and the split must not stall or drop tensors.
			"lone-oversized-gets-own-range", []int{1, 100, 1}, 32, 1024,
			[][2]int{{0, 1}, {1, 2}, {2, 3}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := bulkUploadChunkRanges(tc.nelems, elemSize, tc.maxBytes, tc.maxTensors)
			if len(tc.nelems) == 0 {
				if len(got) != 0 {
					t.Fatalf("empty input: got %v, want no ranges", got)
				}
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("ranges = %v, want %v", got, tc.want)
			}

			// Invariants: contiguous tiling of [0,len) and caps respected
			// (except a single tensor that alone exceeds maxBytes).
			prev := 0
			for _, r := range got {
				if r[0] != prev {
					t.Fatalf("gap/overlap: range %v does not start at %d", r, prev)
				}
				if r[1] <= r[0] {
					t.Fatalf("empty/inverted range %v", r)
				}
				if r[1]-r[0] > tc.maxTensors {
					t.Fatalf("range %v exceeds maxTensors=%d", r, tc.maxTensors)
				}
				bytes := 0
				for i := r[0]; i < r[1]; i++ {
					bytes += tc.nelems[i] * elemSize
				}
				if bytes > tc.maxBytes && r[1]-r[0] > 1 {
					t.Fatalf("range %v (%d bytes) exceeds maxBytes=%d with >1 tensor", r, bytes, tc.maxBytes)
				}
				prev = r[1]
			}
			if prev != len(tc.nelems) {
				t.Fatalf("ranges cover [0,%d), want [0,%d)", prev, len(tc.nelems))
			}
		})
	}
}

// TestBulkUploadChunkRanges_LargeCountIsBounded mirrors the production failure:
// a very large tensor count must split into many bounded chunks rather than one
// giant range (which wedged the GB10 driver).
func TestBulkUploadChunkRanges_LargeCountIsBounded(t *testing.T) {
	const elemSize = 4
	const n = 213304 // the observed hang count
	nelems := make([]int, n)
	for i := range nelems {
		nelems[i] = 193 // one feature row
	}
	ranges := bulkUploadChunkRanges(nelems, elemSize, bulkUploadF32MaxChunkBytes, bulkUploadF32MaxChunkTensors)

	if len(ranges) < 2 {
		t.Fatalf("expected many chunks for %d tensors, got %d", n, len(ranges))
	}
	covered := 0
	for _, r := range ranges {
		if r[1]-r[0] > bulkUploadF32MaxChunkTensors {
			t.Fatalf("chunk %v exceeds tensor cap %d", r, bulkUploadF32MaxChunkTensors)
		}
		bytes := 0
		for i := r[0]; i < r[1]; i++ {
			bytes += nelems[i] * elemSize
		}
		if bytes > bulkUploadF32MaxChunkBytes {
			t.Fatalf("chunk %v (%d bytes) exceeds byte cap %d", r, bytes, bulkUploadF32MaxChunkBytes)
		}
		covered += r[1] - r[0]
	}
	if covered != n {
		t.Fatalf("chunks cover %d tensors, want %d", covered, n)
	}
}

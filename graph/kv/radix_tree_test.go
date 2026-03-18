package kv

import (
	"sync"
	"testing"
)

func makeBlocks(n int) []*Block[float32] {
	blocks := make([]*Block[float32], n)
	for i := range blocks {
		blocks[i] = &Block[float32]{K: []float32{float32(i)}, V: []float32{float32(i)}}
	}
	return blocks
}

func TestRadixTree_Insert(t *testing.T) {
	rt := NewRadixTree[float32](100)

	// Insert "hello world" = tokens [1, 2, 3, 4, 5]
	tokens1 := []int32{1, 2, 3, 4, 5}
	blocks1 := makeBlocks(5)
	rt.Insert(tokens1, blocks1)

	if got := rt.Size(); got != 5 {
		t.Fatalf("Size after first insert: got %d, want 5", got)
	}

	// Insert "hello there" = tokens [1, 2, 3, 6, 7] — shares prefix [1,2,3]
	tokens2 := []int32{1, 2, 3, 6, 7}
	blocks2 := makeBlocks(5)
	rt.Insert(tokens2, blocks2)

	// The shared prefix [1,2,3] should be stored once. The tree should have:
	//   root -> [1,2,3] -> [4,5] (from tokens1)
	//                   -> [6,7] (from tokens2)
	// Total blocks: 3 (shared) + 2 (tokens1 suffix) + 2 (tokens2 suffix) = 7
	if got := rt.Size(); got != 7 {
		t.Fatalf("Size after second insert: got %d, want 7", got)
	}

	// Verify the tree structure: root should have one child keyed on token 1.
	rt.mu.Lock()
	if len(rt.root.children) != 1 {
		t.Fatalf("root children: got %d, want 1", len(rt.root.children))
	}
	child := rt.root.children[1]
	if len(child.tokenIDs) != 3 {
		t.Fatalf("shared edge length: got %d, want 3", len(child.tokenIDs))
	}
	if len(child.children) != 2 {
		t.Fatalf("shared node children: got %d, want 2", len(child.children))
	}
	rt.mu.Unlock()
}

func TestRadixTree_Match(t *testing.T) {
	rt := NewRadixTree[float32](100)

	tokens1 := []int32{1, 2, 3, 4, 5}
	blocks1 := makeBlocks(5)
	rt.Insert(tokens1, blocks1)

	tokens2 := []int32{1, 2, 3, 6, 7}
	blocks2 := makeBlocks(5)
	rt.Insert(tokens2, blocks2)

	tests := []struct {
		name      string
		prefix    []int32
		wantLen   int
		wantCount int // number of blocks returned
	}{
		{
			name:      "exact match first sequence",
			prefix:    []int32{1, 2, 3, 4, 5},
			wantLen:   5,
			wantCount: 5,
		},
		{
			name:      "exact match second sequence",
			prefix:    []int32{1, 2, 3, 6, 7},
			wantLen:   5,
			wantCount: 5,
		},
		{
			name:      "shared prefix only",
			prefix:    []int32{1, 2, 3},
			wantLen:   3,
			wantCount: 3,
		},
		{
			name:      "partial match into shared prefix",
			prefix:    []int32{1, 2},
			wantLen:   2,
			wantCount: 2,
		},
		{
			name:      "100% hit rate: same prefix",
			prefix:    tokens1,
			wantLen:   5,
			wantCount: 5,
		},
		{
			name:      "no match: different start token",
			prefix:    []int32{9, 9, 9},
			wantLen:   0,
			wantCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blocks, matchedLen := rt.Match(tt.prefix)
			if matchedLen != tt.wantLen {
				t.Errorf("matchedLen: got %d, want %d", matchedLen, tt.wantLen)
			}
			if len(blocks) != tt.wantCount {
				t.Errorf("block count: got %d, want %d", len(blocks), tt.wantCount)
			}
		})
	}
}

func TestRadixTree_LRUEviction(t *testing.T) {
	rt := NewRadixTree[float32](10)

	// Insert first sequence: 5 blocks.
	tokens1 := []int32{1, 2, 3, 4, 5}
	rt.Insert(tokens1, makeBlocks(5))

	// Insert second sequence (no overlap): 5 blocks. Total = 10 (at capacity).
	tokens2 := []int32{10, 20, 30, 40, 50}
	rt.Insert(tokens2, makeBlocks(5))

	if got := rt.Size(); got != 10 {
		t.Fatalf("Size at capacity: got %d, want 10", got)
	}

	// Insert third sequence: 5 blocks. Should evict oldest to stay <= 10.
	tokens3 := []int32{100, 200, 300, 400, 500}
	rt.Insert(tokens3, makeBlocks(5))

	if got := rt.Size(); got > 10 {
		t.Fatalf("Size after eviction: got %d, want <= 10", got)
	}

	// The first inserted sequence should have been evicted (LRU).
	_, matchedLen := rt.Match(tokens1)
	if matchedLen != 0 {
		t.Errorf("expected tokens1 to be evicted, but matched %d tokens", matchedLen)
	}

	// The third sequence should still be present.
	_, matchedLen = rt.Match(tokens3)
	if matchedLen != 5 {
		t.Errorf("expected tokens3 to match 5, got %d", matchedLen)
	}
}

func TestRadixTree_NoMatch(t *testing.T) {
	rt := NewRadixTree[float32](100)

	blocks, matchedLen := rt.Match([]int32{1, 2, 3})
	if matchedLen != 0 {
		t.Errorf("empty tree matchedLen: got %d, want 0", matchedLen)
	}
	if blocks != nil {
		t.Errorf("empty tree blocks: got %v, want nil", blocks)
	}

	// nil prefix
	blocks, matchedLen = rt.Match(nil)
	if matchedLen != 0 || blocks != nil {
		t.Errorf("nil prefix: got matchedLen=%d blocks=%v, want 0 and nil", matchedLen, blocks)
	}
}

func TestRadixTree_Concurrent(t *testing.T) {
	rt := NewRadixTree[float32](1000)

	var wg sync.WaitGroup
	const goroutines = 10
	const iterations = 100

	// Concurrent inserts.
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(base int32) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				tokens := []int32{base, int32(i), int32(i + 1)}
				rt.Insert(tokens, makeBlocks(3))
			}
		}(int32(g))
	}

	// Concurrent matches while inserts are running.
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(base int32) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				tokens := []int32{base, int32(i), int32(i + 1)}
				rt.Match(tokens)
			}
		}(int32(g))
	}

	wg.Wait()

	// Just verify no panics and size is reasonable.
	if got := rt.Size(); got < 0 {
		t.Fatalf("negative size: %d", got)
	}
}

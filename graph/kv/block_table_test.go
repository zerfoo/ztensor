package kv

import "testing"

func TestBlockTable_Append(t *testing.T) {
	tests := []struct {
		name       string
		blockSize  int
		appends    []int // successive Append calls
		wantBlocks int
		wantTokens int
	}{
		{"single-partial", 16, []int{5}, 1, 5},
		{"exact-one-block", 16, []int{16}, 1, 16},
		{"spill-two-blocks", 16, []int{17}, 2, 17},
		{"two-appends-fill", 16, []int{10, 6}, 1, 16},
		{"two-appends-spill", 16, []int{10, 10}, 2, 20},
		{"three-blocks", 16, []int{48}, 3, 48},
		{"incremental-growth", 4, []int{3, 2, 5}, 3, 10},
		{"zero-append", 16, []int{0, 5}, 1, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pool, err := NewBlockPool[float32](16, 2, tt.blockSize, 64)
			if err != nil {
				t.Fatalf("NewBlockPool: %v", err)
			}
			bt := NewBlockTable[float32](pool)

			for _, n := range tt.appends {
				if err := bt.Append(n); err != nil {
					t.Fatalf("Append(%d): %v", n, err)
				}
			}

			if got := bt.BlockCount(); got != tt.wantBlocks {
				t.Errorf("BlockCount() = %d, want %d", got, tt.wantBlocks)
			}
			if got := bt.TokenCount(); got != tt.wantTokens {
				t.Errorf("TokenCount() = %d, want %d", got, tt.wantTokens)
			}
		})
	}
}

func TestBlockTable_Lookup(t *testing.T) {
	pool, err := NewBlockPool[float32](8, 2, 4, 64)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	bt := NewBlockTable[float32](pool)

	if err := bt.Append(10); err != nil {
		t.Fatalf("Append: %v", err)
	}
	// 10 tokens, blockSize=4 → blocks: [0-3], [4-7], [8-9]

	tests := []struct {
		pos       int
		wantBlock int // index into bt.blocks
		wantOff   int
	}{
		{0, 0, 0},
		{3, 0, 3},
		{4, 1, 0},
		{7, 1, 3},
		{8, 2, 0},
		{9, 2, 1},
	}

	for _, tt := range tests {
		b, off, err := bt.Lookup(tt.pos)
		if err != nil {
			t.Errorf("Lookup(%d): %v", tt.pos, err)
			continue
		}
		if b != bt.blocks[tt.wantBlock] {
			t.Errorf("Lookup(%d): got block %p, want block[%d] %p", tt.pos, b, tt.wantBlock, bt.blocks[tt.wantBlock])
		}
		if off != tt.wantOff {
			t.Errorf("Lookup(%d): offset = %d, want %d", tt.pos, off, tt.wantOff)
		}
	}

	// Out-of-range positions.
	for _, pos := range []int{-1, 10, 100} {
		if _, _, err := bt.Lookup(pos); err == nil {
			t.Errorf("Lookup(%d) should return error for out-of-range", pos)
		}
	}
}

func TestBlockTable_Free(t *testing.T) {
	pool, err := NewBlockPool[float32](8, 2, 16, 64)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	bt := NewBlockTable[float32](pool)
	if err := bt.Append(40); err != nil {
		t.Fatalf("Append: %v", err)
	}
	// 40 tokens, blockSize=16 → 3 blocks (16+16+8)
	if got := bt.BlockCount(); got != 3 {
		t.Fatalf("BlockCount() = %d, want 3", got)
	}

	beforeFree := pool.Available()
	bt.Free()

	if got := bt.BlockCount(); got != 0 {
		t.Errorf("BlockCount() after Free = %d, want 0", got)
	}
	if got := bt.TokenCount(); got != 0 {
		t.Errorf("TokenCount() after Free = %d, want 0", got)
	}
	if got := pool.Available(); got != beforeFree+3 {
		t.Errorf("pool.Available() after Free = %d, want %d", got, beforeFree+3)
	}

	// Reuse after free.
	if err := bt.Append(5); err != nil {
		t.Errorf("Append after Free: %v", err)
	}
	if got := bt.BlockCount(); got != 1 {
		t.Errorf("BlockCount() after reuse = %d, want 1", got)
	}
}

func TestBlockTable_MultiSeq(t *testing.T) {
	pool, err := NewBlockPool[float32](8, 2, 4, 64)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	seq1 := NewBlockTable[float32](pool)
	seq2 := NewBlockTable[float32](pool)

	if err := seq1.Append(6); err != nil {
		t.Fatalf("seq1.Append: %v", err)
	}
	if err := seq2.Append(10); err != nil {
		t.Fatalf("seq2.Append: %v", err)
	}

	// seq1: 2 blocks (4+2), seq2: 3 blocks (4+4+2) → 5 total from 8
	if got := pool.Available(); got != 3 {
		t.Errorf("pool.Available() = %d, want 3", got)
	}

	// Verify independent lookups.
	b1, off1, err := seq1.Lookup(5)
	if err != nil {
		t.Fatalf("seq1.Lookup(5): %v", err)
	}
	b2, off2, err := seq2.Lookup(5)
	if err != nil {
		t.Fatalf("seq2.Lookup(5): %v", err)
	}

	// Position 5 → block index 1, offset 1 for both (blockSize=4).
	if off1 != 1 || off2 != 1 {
		t.Errorf("offsets: seq1=%d, seq2=%d, want 1,1", off1, off2)
	}
	// But the physical blocks must differ.
	if b1 == b2 {
		t.Error("seq1 and seq2 should have different physical blocks for the same logical position")
	}

	// Free seq1 and verify pool recovers.
	seq1.Free()
	if got := pool.Available(); got != 5 {
		t.Errorf("pool.Available() after seq1.Free = %d, want 5", got)
	}

	seq2.Free()
	if got := pool.Available(); got != 8 {
		t.Errorf("pool.Available() after seq2.Free = %d, want 8", got)
	}
}

func TestBlockTable_AppendExhaustsPool(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 1, 4, 32)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	bt := NewBlockTable[float32](pool)
	// 2 blocks × 4 tokens = 8 max. Appending 9 should fail.
	if err := bt.Append(9); err == nil {
		t.Fatal("Append beyond pool capacity should return error")
	}
}

package kv

import (
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// BlockTable maps a single sequence's logical token positions to physical
// blocks allocated from a BlockPool. It grows automatically as tokens are
// appended and returns all blocks to the pool on Free.
type BlockTable[T tensor.Numeric] struct {
	pool       *BlockPool[T]
	blocks     []*Block[T]
	tokenCount int
}

// NewBlockTable creates a BlockTable backed by the given pool.
func NewBlockTable[T tensor.Numeric](pool *BlockPool[T]) *BlockTable[T] {
	return &BlockTable[T]{pool: pool}
}

// Append adds tokenCount new token positions to the table, allocating
// additional blocks from the pool as needed.
func (bt *BlockTable[T]) Append(tokenCount int) error {
	if tokenCount <= 0 {
		return nil
	}

	bs := bt.pool.BlockSize()

	// Fill the current last block first.
	remaining := tokenCount
	if len(bt.blocks) > 0 {
		last := bt.blocks[len(bt.blocks)-1]
		avail := bs - last.Used
		if avail > 0 {
			fill := avail
			if remaining < fill {
				fill = remaining
			}
			last.Used += fill
			remaining -= fill
		}
	}

	// Allocate new blocks for the rest.
	for remaining > 0 {
		b, err := bt.pool.Alloc()
		if err != nil {
			return fmt.Errorf("kv.BlockTable.Append: %w", err)
		}
		fill := bs
		if remaining < fill {
			fill = remaining
		}
		b.Used = fill
		bt.blocks = append(bt.blocks, b)
		remaining -= fill
	}

	bt.tokenCount += tokenCount
	return nil
}

// Lookup returns the physical block and offset within that block for a
// logical token position.
func (bt *BlockTable[T]) Lookup(logicalPos int) (*Block[T], int, error) {
	if logicalPos < 0 || logicalPos >= bt.tokenCount {
		return nil, 0, fmt.Errorf("kv.BlockTable.Lookup: position %d out of range [0, %d)", logicalPos, bt.tokenCount)
	}

	bs := bt.pool.BlockSize()
	blockIdx := logicalPos / bs
	offset := logicalPos % bs
	return bt.blocks[blockIdx], offset, nil
}

// Free returns all blocks to the pool and resets the table.
func (bt *BlockTable[T]) Free() {
	for _, b := range bt.blocks {
		bt.pool.Free(b)
	}
	bt.blocks = bt.blocks[:0]
	bt.tokenCount = 0
}

// BlockCount returns the number of blocks currently held.
func (bt *BlockTable[T]) BlockCount() int {
	return len(bt.blocks)
}

// TokenCount returns the total number of token positions tracked.
func (bt *BlockTable[T]) TokenCount() int {
	return bt.tokenCount
}

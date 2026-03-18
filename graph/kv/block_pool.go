// Package kv provides paged-attention KV cache primitives for transformer
// inference. The BlockPool pre-allocates fixed-size blocks of key/value
// storage so that the hot path (Alloc / Free) is allocation-free.
package kv

import (
	"fmt"
	"sync"

	"github.com/zerfoo/ztensor/tensor"
)

// DefaultBlockSize is the number of token positions per block.
const DefaultBlockSize = 16

// Block holds pre-allocated key and value data for a fixed number of token
// positions across all layers. K and V each have
// numLayers * blockSize * headDim elements laid out as
// [layer][position][headDim] in row-major order.
type Block[T tensor.Numeric] struct {
	K    []T
	V    []T
	Used int // number of token positions written (0..blockSize)
}

// BlockPool manages a fixed-size pool of pre-allocated KV cache blocks.
// Blocks are allocated at startup and recycled via Alloc/Free.
// All methods are safe for concurrent use.
type BlockPool[T tensor.Numeric] struct {
	blocks    []Block[T]
	free      []*Block[T] // stack of available block pointers
	numLayers int
	blockSize int // tokens per block
	headDim   int
	mu        sync.Mutex
}

// NewBlockPool creates a pool with numBlocks pre-allocated blocks. Each block
// holds K and V data for blockSize token positions across numLayers, with
// headDim elements per position per layer. blockSize defaults to
// DefaultBlockSize (16) when zero.
func NewBlockPool[T tensor.Numeric](numBlocks, numLayers, blockSize, headDim int) (*BlockPool[T], error) {
	if blockSize == 0 {
		blockSize = DefaultBlockSize
	}
	if numBlocks <= 0 || numLayers <= 0 || blockSize <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("kv.NewBlockPool: all parameters must be positive: blocks=%d layers=%d blockSize=%d headDim=%d",
			numBlocks, numLayers, blockSize, headDim)
	}

	elemsPerSide := numLayers * blockSize * headDim
	blocks := make([]Block[T], numBlocks)
	free := make([]*Block[T], numBlocks)
	for i := range blocks {
		blocks[i].K = make([]T, elemsPerSide)
		blocks[i].V = make([]T, elemsPerSide)
		free[i] = &blocks[numBlocks-1-i] // stack order: first Alloc returns blocks[0]
	}

	return &BlockPool[T]{
		blocks:    blocks,
		free:      free,
		numLayers: numLayers,
		blockSize: blockSize,
		headDim:   headDim,
	}, nil
}

// Alloc returns a free block from the pool. Returns an error if the pool is
// exhausted. The returned block has Used reset to 0.
func (p *BlockPool[T]) Alloc() (*Block[T], error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.free) == 0 {
		return nil, fmt.Errorf("kv.BlockPool: pool exhausted, all %d blocks allocated", len(p.blocks))
	}

	b := p.free[len(p.free)-1]
	p.free = p.free[:len(p.free)-1]
	b.Used = 0
	return b, nil
}

// Free returns a block to the pool. The block's Used counter is reset to 0.
func (p *BlockPool[T]) Free(b *Block[T]) {
	b.Used = 0
	p.mu.Lock()
	p.free = append(p.free, b)
	p.mu.Unlock()
}

// Cap returns the total number of blocks in the pool.
func (p *BlockPool[T]) Cap() int {
	return len(p.blocks)
}

// Available returns the number of free blocks.
func (p *BlockPool[T]) Available() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.free)
}

// BlockSize returns the number of token positions per block.
func (p *BlockPool[T]) BlockSize() int {
	return p.blockSize
}

// NumLayers returns the number of layers per block.
func (p *BlockPool[T]) NumLayers() int {
	return p.numLayers
}

// HeadDim returns the head dimension per position per layer.
func (p *BlockPool[T]) HeadDim() int {
	return p.headDim
}

// FragmentationRatio returns the fraction of allocated blocks that are
// partially used (0 < Used < blockSize). A ratio of 0.0 means all allocated
// blocks are either empty or full; 1.0 means every allocated block is
// partially filled. Returns 0.0 when no blocks are allocated.
func (p *BlockPool[T]) FragmentationRatio() float64 {
	p.mu.Lock()
	defer p.mu.Unlock()

	allocated := len(p.blocks) - len(p.free)
	if allocated == 0 {
		return 0.0
	}

	// Build a set of free block pointers for O(1) lookup.
	freeSet := make(map[*Block[T]]struct{}, len(p.free))
	for _, b := range p.free {
		freeSet[b] = struct{}{}
	}

	fragmented := 0
	for i := range p.blocks {
		b := &p.blocks[i]
		if _, ok := freeSet[b]; ok {
			continue // free block, skip
		}
		if b.Used > 0 && b.Used < p.blockSize {
			fragmented++
		}
	}

	return float64(fragmented) / float64(allocated)
}

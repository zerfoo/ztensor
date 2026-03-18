package kv

import "github.com/zerfoo/ztensor/tensor"

// PagedKVCache bundles a BlockTable and its backing BlockPool into a single
// value that can be passed to the compute engine's paged GQA method.
type PagedKVCache[T tensor.Numeric] struct {
	Table *BlockTable[T]
}

// NewPagedKVCache creates a PagedKVCache wrapping the given BlockTable.
func NewPagedKVCache[T tensor.Numeric](table *BlockTable[T]) *PagedKVCache[T] {
	return &PagedKVCache[T]{Table: table}
}

// SeqLen returns the number of token positions in the cache.
func (c *PagedKVCache[T]) SeqLen() int {
	return c.Table.TokenCount()
}

// BlockSize returns the number of token positions per block.
func (c *PagedKVCache[T]) BlockSize() int {
	return c.Table.Pool().BlockSize()
}

// NumKVHeads returns the number of KV heads. The block pool stores
// numLayers * blockSize * headDim elements per block, but for a single-layer
// paged attention call the "numKVHeads" is derived from the caller.
// This is intentionally not stored here — the caller provides it.

// HeadDim returns the head dimension from the backing pool.
func (c *PagedKVCache[T]) HeadDim() int {
	return c.Table.Pool().HeadDim()
}

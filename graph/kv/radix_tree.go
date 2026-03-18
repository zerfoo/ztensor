package kv

import (
	"sync"
	"time"

	"github.com/zerfoo/ztensor/tensor"
)

// RadixNode is a node in the radix tree. Each edge stores a slice of token
// IDs and a corresponding slice of KV blocks. Children are keyed by the
// first token ID of their edge.
type RadixNode[T tensor.Numeric] struct {
	children map[int32]*RadixNode[T]
	tokenIDs []int32     // token IDs on the edge leading to this node
	blocks   []*Block[T] // KV blocks corresponding to tokenIDs
	lastUsed time.Time
}

// RadixTree stores shared prompt prefixes so that KV blocks can be reused
// across sequences that share a common prefix. It is safe for concurrent use.
type RadixTree[T tensor.Numeric] struct {
	root     *RadixNode[T]
	capacity int // max total blocks cached in tree
	size     int // current total blocks in tree
	mu       sync.Mutex
}

// NewRadixTree creates a radix tree that caches up to capacity KV blocks.
func NewRadixTree[T tensor.Numeric](capacity int) *RadixTree[T] {
	return &RadixTree[T]{
		root:     &RadixNode[T]{children: make(map[int32]*RadixNode[T])},
		capacity: capacity,
	}
}

// Insert adds a token prefix and its associated KV blocks into the tree.
// If the prefix already exists, the blocks are updated. When the tree
// exceeds capacity, LRU leaf nodes are evicted.
func (rt *RadixTree[T]) Insert(tokenIDs []int32, blocks []*Block[T]) {
	if len(tokenIDs) == 0 || len(blocks) == 0 {
		return
	}

	rt.mu.Lock()
	defer rt.mu.Unlock()

	now := time.Now()
	rt.insert(rt.root, tokenIDs, blocks, now)

	for rt.size > rt.capacity {
		if !rt.evictOne() {
			break
		}
	}
}

func (rt *RadixTree[T]) insert(node *RadixNode[T], tokenIDs []int32, blocks []*Block[T], now time.Time) {
	if len(tokenIDs) == 0 {
		return
	}

	key := tokenIDs[0]
	child, exists := node.children[key]

	if !exists {
		// No matching child — create a new leaf.
		newNode := &RadixNode[T]{
			children: make(map[int32]*RadixNode[T]),
			tokenIDs: make([]int32, len(tokenIDs)),
			blocks:   make([]*Block[T], len(blocks)),
			lastUsed: now,
		}
		copy(newNode.tokenIDs, tokenIDs)
		copy(newNode.blocks, blocks)
		node.children[key] = newNode
		rt.size += len(blocks)
		return
	}

	// Find the common prefix length between the child edge and tokenIDs.
	commonLen := commonPrefix(child.tokenIDs, tokenIDs)

	if commonLen == len(child.tokenIDs) {
		// child edge is fully matched — descend into child.
		child.lastUsed = now
		if commonLen < len(tokenIDs) {
			rt.insert(child, tokenIDs[commonLen:], blocks[commonLen:], now)
		}
		return
	}

	// Split the child edge at commonLen.
	// Create an intermediate node for the shared prefix.
	mid := &RadixNode[T]{
		children: make(map[int32]*RadixNode[T]),
		tokenIDs: make([]int32, commonLen),
		blocks:   make([]*Block[T], commonLen),
		lastUsed: now,
	}
	copy(mid.tokenIDs, child.tokenIDs[:commonLen])
	copy(mid.blocks, child.blocks[:commonLen])
	rt.size += commonLen

	// Move the remainder of the old child under mid.
	child.tokenIDs = child.tokenIDs[commonLen:]
	child.blocks = child.blocks[commonLen:]
	rt.size -= commonLen // these blocks were already counted in child
	mid.children[child.tokenIDs[0]] = child

	node.children[key] = mid

	// If there are remaining tokens to insert, recurse.
	if commonLen < len(tokenIDs) {
		rt.insert(mid, tokenIDs[commonLen:], blocks[commonLen:], now)
	}
}

// Match returns the KV blocks for the longest matching prefix and the number
// of tokens matched. If no prefix matches, matchedLen is 0 and blocks is nil.
func (rt *RadixTree[T]) Match(prefix []int32) (matchedBlocks []*Block[T], matchedLen int) {
	if len(prefix) == 0 {
		return nil, 0
	}

	rt.mu.Lock()
	defer rt.mu.Unlock()

	now := time.Now()
	node := rt.root
	remaining := prefix

	for len(remaining) > 0 {
		key := remaining[0]
		child, exists := node.children[key]
		if !exists {
			break
		}

		// Check how many tokens of the edge match.
		edgeMatch := commonPrefix(child.tokenIDs, remaining)
		if edgeMatch == 0 {
			break
		}

		matchedBlocks = append(matchedBlocks, child.blocks[:edgeMatch]...)
		matchedLen += edgeMatch
		child.lastUsed = now
		remaining = remaining[edgeMatch:]

		if edgeMatch < len(child.tokenIDs) {
			// Partial edge match — stop here.
			break
		}

		node = child
	}

	return matchedBlocks, matchedLen
}

// Evict removes the least recently used leaf node from the tree.
func (rt *RadixTree[T]) Evict() {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.evictOne()
}

func (rt *RadixTree[T]) evictOne() bool {
	// Find the LRU leaf across the entire tree.
	var lruParent *RadixNode[T]
	var lruKey int32
	var lruTime time.Time
	found := false

	var walk func(parent *RadixNode[T])
	walk = func(parent *RadixNode[T]) {
		for k, child := range parent.children {
			if len(child.children) == 0 {
				// Leaf node.
				if !found || child.lastUsed.Before(lruTime) {
					lruParent = parent
					lruKey = k
					lruTime = child.lastUsed
					found = true
				}
			} else {
				walk(child)
			}
		}
	}
	walk(rt.root)

	if !found {
		return false
	}

	evicted := lruParent.children[lruKey]
	rt.size -= len(evicted.blocks)
	delete(lruParent.children, lruKey)
	return true
}

// Size returns the total number of blocks cached in the tree.
func (rt *RadixTree[T]) Size() int {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return rt.size
}

// commonPrefix returns the length of the common prefix between a and b.
func commonPrefix(a, b []int32) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

package kv

import (
	"sync"
	"testing"
)

func TestBlockPool_AllocFree(t *testing.T) {
	tests := []struct {
		name      string
		numBlocks int
		numLayers int
		blockSize int
		headDim   int
	}{
		{"small", 4, 2, 16, 64},
		{"default-blocksize", 8, 4, 0, 128}, // blockSize=0 → DefaultBlockSize
		{"single-block", 1, 1, 16, 32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pool, err := NewBlockPool[float32](tt.numBlocks, tt.numLayers, tt.blockSize, tt.headDim)
			if err != nil {
				t.Fatalf("NewBlockPool: %v", err)
			}

			bs := tt.blockSize
			if bs == 0 {
				bs = DefaultBlockSize
			}

			if got := pool.Cap(); got != tt.numBlocks {
				t.Errorf("Cap() = %d, want %d", got, tt.numBlocks)
			}
			if got := pool.Available(); got != tt.numBlocks {
				t.Errorf("Available() = %d, want %d", got, tt.numBlocks)
			}
			if got := pool.BlockSize(); got != bs {
				t.Errorf("BlockSize() = %d, want %d", got, bs)
			}

			// Allocate all blocks.
			allocated := make([]*Block[float32], tt.numBlocks)
			for i := 0; i < tt.numBlocks; i++ {
				b, err := pool.Alloc()
				if err != nil {
					t.Fatalf("Alloc() #%d: %v", i, err)
				}
				allocated[i] = b

				wantElems := tt.numLayers * bs * tt.headDim
				if len(b.K) != wantElems {
					t.Errorf("block %d: len(K) = %d, want %d", i, len(b.K), wantElems)
				}
				if len(b.V) != wantElems {
					t.Errorf("block %d: len(V) = %d, want %d", i, len(b.V), wantElems)
				}
			}

			// Pool should be exhausted.
			if _, err := pool.Alloc(); err == nil {
				t.Fatal("Alloc() on exhausted pool should return error")
			}
			if got := pool.Available(); got != 0 {
				t.Errorf("Available() after exhaustion = %d, want 0", got)
			}

			// Free all and verify re-availability.
			for _, b := range allocated {
				pool.Free(b)
			}
			if got := pool.Available(); got != tt.numBlocks {
				t.Errorf("Available() after Free = %d, want %d", got, tt.numBlocks)
			}

			// Re-alloc should succeed and reset Used.
			allocated[0].Used = 5
			pool.Free(allocated[0])
			b, err := pool.Alloc()
			if err != nil {
				t.Fatalf("re-Alloc: %v", err)
			}
			if b.Used != 0 {
				t.Errorf("re-allocated block Used = %d, want 0", b.Used)
			}
		})
	}
}

func TestBlockPool_NewErrors(t *testing.T) {
	tests := []struct {
		name                             string
		numBlocks, numLayers, bs, headDim int
	}{
		{"zero-blocks", 0, 2, 16, 64},
		{"negative-layers", 4, -1, 16, 64},
		{"negative-blocksize", 4, 2, -1, 64},
		{"zero-headdim", 4, 2, 16, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewBlockPool[float32](tt.numBlocks, tt.numLayers, tt.bs, tt.headDim)
			if err == nil {
				t.Fatal("NewBlockPool should return error for invalid params")
			}
		})
	}
}

func TestBlockPool_ZeroAllocWarmPath(t *testing.T) {
	pool, err := NewBlockPool[float32](8, 4, 16, 64)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	// Warm up: alloc and free once to fill the free list.
	b, err := pool.Alloc()
	if err != nil {
		t.Fatalf("warm-up Alloc: %v", err)
	}
	pool.Free(b)

	// Measure allocations on the warm path.
	allocs := testing.AllocsPerRun(100, func() {
		b, err := pool.Alloc()
		if err != nil {
			t.Fatalf("Alloc: %v", err)
		}
		pool.Free(b)
	})

	if allocs > 0 {
		t.Errorf("warm-path Alloc/Free allocated %.1f times, want 0", allocs)
	}
}

func TestBlockPool_FragmentationRatio(t *testing.T) {
	pool, err := NewBlockPool[float32](4, 2, 16, 64)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	// No blocks allocated → 0.0.
	if got := pool.FragmentationRatio(); got != 0.0 {
		t.Errorf("empty pool FragmentationRatio() = %f, want 0.0", got)
	}

	// Allocate 4 blocks: none used → 0.0 (Used=0 is not "partially used").
	blocks := make([]*Block[float32], 4)
	for i := range blocks {
		blocks[i], err = pool.Alloc()
		if err != nil {
			t.Fatalf("Alloc: %v", err)
		}
	}
	if got := pool.FragmentationRatio(); got != 0.0 {
		t.Errorf("all-empty FragmentationRatio() = %f, want 0.0", got)
	}

	// Mark 2 blocks as partially used.
	blocks[0].Used = 5
	blocks[1].Used = 10
	// blocks[2] stays at 0 (empty), blocks[3] fully used.
	blocks[3].Used = 16
	// 2 fragmented out of 4 allocated → 0.5
	if got := pool.FragmentationRatio(); got != 0.5 {
		t.Errorf("FragmentationRatio() = %f, want 0.5", got)
	}

	// Free one fragmented block → 1 fragmented out of 3 allocated ≈ 0.333...
	pool.Free(blocks[0])
	want := 1.0 / 3.0
	got := pool.FragmentationRatio()
	if got < want-0.01 || got > want+0.01 {
		t.Errorf("FragmentationRatio() = %f, want ~%f", got, want)
	}
}

func TestBlockPool_Concurrent(t *testing.T) {
	pool, err := NewBlockPool[float32](64, 4, 16, 64)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	const goroutines = 16
	const opsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < opsPerGoroutine; i++ {
				b, err := pool.Alloc()
				if err != nil {
					// Pool exhausted — expected under contention.
					continue
				}
				b.Used = 1 // simulate partial write
				pool.Free(b)
			}
		}()
	}

	wg.Wait()

	// All blocks should be returned.
	if got := pool.Available(); got != 64 {
		t.Errorf("Available() after concurrent ops = %d, want 64", got)
	}
}

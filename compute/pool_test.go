package compute

import (
	"sync"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestTensorPool_AcquireRelease(t *testing.T) {
	pool := NewTensorPool[float32]()

	// Acquire a new tensor.
	t1, err := pool.Acquire([]int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(t1.Shape()) != 2 || t1.Shape()[0] != 2 || t1.Shape()[1] != 3 {
		t.Fatalf("unexpected shape: %v", t1.Shape())
	}

	// Write data to it.
	data := t1.Data()
	for i := range data {
		data[i] = float32(i + 1)
	}

	// Release and re-acquire: should get same buffer back (zeroed).
	pool.Release(t1)
	if pool.Len() != 1 {
		t.Fatalf("expected pool len 1, got %d", pool.Len())
	}

	t2, err := pool.Acquire([]int{2, 3})
	if err != nil {
		t.Fatal(err)
	}

	// After acquire, pool should be empty again.
	if pool.Len() != 0 {
		t.Fatalf("expected pool len 0 after acquire, got %d", pool.Len())
	}

	// The reused tensor should be zeroed.
	for i, v := range t2.Data() {
		if v != 0 {
			t.Fatalf("data[%d] = %f, expected 0 after reuse", i, v)
		}
	}
}

func TestTensorPool_DifferentShapes(t *testing.T) {
	pool := NewTensorPool[float32]()

	t1, _ := pool.Acquire([]int{2, 3})
	t2, _ := pool.Acquire([]int{4, 5})

	pool.Release(t1)
	pool.Release(t2)

	if pool.Len() != 2 {
		t.Fatalf("expected 2 pooled tensors, got %d", pool.Len())
	}

	// Acquiring [2,3] should not return the [4,5] tensor.
	t3, _ := pool.Acquire([]int{2, 3})
	if t3.Shape()[0] != 2 || t3.Shape()[1] != 3 {
		t.Fatalf("wrong shape returned: %v", t3.Shape())
	}

	if pool.Len() != 1 {
		t.Fatalf("expected 1 pooled tensor remaining, got %d", pool.Len())
	}
}

func TestTensorPool_ReleaseNil(t *testing.T) {
	pool := NewTensorPool[float32]()
	pool.Release(nil) // should not panic
	if pool.Len() != 0 {
		t.Fatal("nil release should not add to pool")
	}
}

func TestTensorPool_Concurrent(t *testing.T) {
	pool := NewTensorPool[float32]()
	const goroutines = 16
	const iterations = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := range goroutines {
		go func(id int) {
			defer wg.Done()
			shape := []int{id + 1, 4}
			for range iterations {
				tn, err := pool.Acquire(shape)
				if err != nil {
					t.Errorf("acquire failed: %v", err)
					return
				}
				// Simulate some work.
				data := tn.Data()
				for i := range data {
					data[i] = float32(id)
				}
				pool.Release(tn)
			}
		}(g)
	}
	wg.Wait()
}

func TestTensorPool_ScalarShape(t *testing.T) {
	pool := NewTensorPool[float32]()

	t1, err := pool.Acquire([]int{})
	if err != nil {
		t.Fatal(err)
	}
	if t1.Size() != 1 {
		t.Fatalf("scalar tensor should have size 1, got %d", t1.Size())
	}

	pool.Release(t1)
	t2, err := pool.Acquire([]int{})
	if err != nil {
		t.Fatal(err)
	}
	if t2.Size() != 1 {
		t.Fatalf("reacquired scalar should have size 1, got %d", t2.Size())
	}
}

func BenchmarkTensorPool_AcquireRelease(b *testing.B) {
	pool := NewTensorPool[float32]()
	shape := []int{1, 16, 256, 64}

	// Prime the pool with one tensor.
	t1, _ := pool.Acquire(shape)
	pool.Release(t1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tn, _ := pool.Acquire(shape)
		pool.Release(tn)
	}
}

func BenchmarkTensorNew_Baseline(b *testing.B) {
	shape := []int{1, 16, 256, 64}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tensor.New[float32](shape, nil)
	}
}

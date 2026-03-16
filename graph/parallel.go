package graph

import (
	"context"
	"fmt"
	"runtime"
	"sync"

	"github.com/zerfoo/ztensor/tensor"
)

// ParallelForward executes the forward pass with dependency-aware parallelism.
// Independent nodes are dispatched to a goroutine pool concurrently.
// The result is identical to sequential Forward.
func ParallelForward[T tensor.Numeric](ctx context.Context, g *Graph[T], inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if len(inputs) != len(g.inputs) {
		return nil, fmt.Errorf("expected %d inputs, got %d", len(g.inputs), len(inputs))
	}

	nodes := g.nodes
	deps := g.dependencies

	// Build reverse dependency map and compute in-degrees.
	inDegree := make(map[Node[T]]int, len(nodes))
	dependents := make(map[Node[T]][]Node[T], len(nodes))
	for _, n := range nodes {
		if _, isInput := n.(*inputNode[T]); isInput {
			continue
		}
		inDegree[n] = len(deps[n])
		for _, dep := range deps[n] {
			dependents[dep] = append(dependents[dep], n)
		}
	}

	// Memo for results.
	memo := make(map[Node[T]]*tensor.TensorNumeric[T], len(nodes))
	for i, n := range g.inputs {
		memo[n] = inputs[i]
	}

	// Count non-input nodes that need execution.
	remaining := 0
	for _, n := range nodes {
		if _, isInput := n.(*inputNode[T]); !isInput {
			remaining++
		}
	}

	if remaining == 0 {
		return memo[g.output], nil
	}

	// "Complete" input nodes: decrement in-degrees of their dependents.
	ready := make(chan Node[T], len(nodes))
	for _, n := range g.inputs {
		for _, dep := range dependents[n] {
			inDegree[dep]--
			if inDegree[dep] == 0 {
				ready <- dep
			}
		}
	}

	var (
		mu       sync.Mutex
		firstErr error
		once     sync.Once
		done     = make(chan struct{})
	)

	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup

	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case n, ok := <-ready:
					if !ok {
						return
					}
					if ctx.Err() != nil {
						mu.Lock()
						if firstErr == nil {
							firstErr = ctx.Err()
						}
						mu.Unlock()
						once.Do(func() { close(done) })
						return
					}

					// Gather inputs.
					mu.Lock()
					nodeDeps := deps[n]
					nodeInputs := make([]*tensor.TensorNumeric[T], len(nodeDeps))
					for i, dep := range nodeDeps {
						nodeInputs[i] = memo[dep]
					}
					mu.Unlock()

					// Execute node.
					output, err := n.Forward(ctx, nodeInputs...)

					mu.Lock()
					if err != nil {
						if firstErr == nil {
							firstErr = fmt.Errorf("node %s: %w", n.OpType(), err)
						}
						mu.Unlock()
						once.Do(func() { close(done) })
						return
					}

					memo[n] = output

					// Decrement dependents' in-degrees and enqueue newly-ready nodes.
					for _, d := range dependents[n] {
						inDegree[d]--
						if inDegree[d] == 0 {
							ready <- d
						}
					}

					remaining--
					allDone := remaining == 0
					mu.Unlock()

					if allDone {
						once.Do(func() { close(done) })
						return
					}

				case <-done:
					return
				case <-ctx.Done():
					mu.Lock()
					if firstErr == nil {
						firstErr = ctx.Err()
					}
					mu.Unlock()
					once.Do(func() { close(done) })
					return
				}
			}
		}()
	}

	<-done
	close(ready)
	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}

	return memo[g.output], nil
}

package nccl

import (
	"sync"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// requireNccl skips the test when libnccl.so.2 cannot be dlopen'd. Pure
// constant/marshaling tests do not need this guard and should not call it.
func requireNccl(t *testing.T) {
	t.Helper()
	if !Available() {
		t.Skip("libnccl.so.2 not available on this host")
	}
}

func TestConstants(t *testing.T) {
	if Float32 != 7 || Float64 != 8 || Int32 != 2 || Int64 != 4 {
		t.Fatalf("unexpected NCCL DataType ABI constants: f32=%d f64=%d i32=%d i64=%d",
			Float32, Float64, Int32, Int64)
	}
	if Sum != 0 || Avg != 4 || Max != 2 || Min != 3 {
		t.Fatalf("unexpected NCCL ReduceOp ABI constants: sum=%d avg=%d max=%d min=%d",
			Sum, Avg, Max, Min)
	}
}

func TestUniqueIDFromBytesRoundTripNoLib(t *testing.T) {
	// This exercises the pure-Go marshaling path and runs on every platform.
	src := make([]byte, 128)
	for i := range src {
		src[i] = byte(i)
	}
	uid, err := UniqueIDFromBytes(src)
	if err != nil {
		t.Fatalf("UniqueIDFromBytes: %v", err)
	}
	out := uid.Bytes()
	if len(out) != 128 {
		t.Fatalf("Bytes length = %d, want 128", len(out))
	}
	for i := range src {
		if out[i] != src[i] {
			t.Fatalf("byte %d: got %d want %d", i, out[i], src[i])
		}
	}
}

func TestGetUniqueID(t *testing.T) {
	requireNccl(t)
	uid, err := GetUniqueID()
	if err != nil {
		t.Fatalf("GetUniqueID: %v", err)
	}
	b := uid.Bytes()
	if len(b) == 0 {
		t.Fatal("UniqueID.Bytes returned empty slice")
	}
}

func TestUniqueIDRoundTrip(t *testing.T) {
	requireNccl(t)
	uid, err := GetUniqueID()
	if err != nil {
		t.Fatalf("GetUniqueID: %v", err)
	}
	b := uid.Bytes()
	uid2, err := UniqueIDFromBytes(b)
	if err != nil {
		t.Fatalf("UniqueIDFromBytes: %v", err)
	}
	b2 := uid2.Bytes()
	if len(b) != len(b2) {
		t.Fatalf("byte length mismatch: %d vs %d", len(b), len(b2))
	}
	for i := range b {
		if b[i] != b2[i] {
			t.Fatalf("byte mismatch at index %d", i)
		}
	}
}

func TestUniqueIDFromBytesInvalidLength(t *testing.T) {
	_, err := UniqueIDFromBytes([]byte{1, 2, 3})
	if err == nil {
		t.Fatal("expected error for invalid byte length")
	}
}

func TestSingleRankInitDestroy(t *testing.T) {
	requireNccl(t)
	count, err := cuda.GetDeviceCount()
	if err != nil || count < 1 {
		t.Skip("requires at least 1 CUDA device")
	}
	if err := cuda.SetDevice(0); err != nil {
		t.Fatalf("SetDevice: %v", err)
	}
	uid, err := GetUniqueID()
	if err != nil {
		t.Fatalf("GetUniqueID: %v", err)
	}
	comm, err := InitRank(uid, 1, 0)
	if err != nil {
		t.Fatalf("InitRank: %v", err)
	}
	if err := comm.GetAsyncError(); err != nil {
		t.Errorf("GetAsyncError: %v", err)
	}
	if err := comm.Destroy(); err != nil {
		t.Errorf("Destroy: %v", err)
	}
}

func TestSingleRankAllReduce(t *testing.T) {
	requireNccl(t)
	count, err := cuda.GetDeviceCount()
	if err != nil || count < 1 {
		t.Skip("requires at least 1 CUDA device")
	}
	if err := cuda.SetDevice(0); err != nil {
		t.Fatalf("SetDevice: %v", err)
	}
	uid, err := GetUniqueID()
	if err != nil {
		t.Fatalf("GetUniqueID: %v", err)
	}
	comm, err := InitRank(uid, 1, 0)
	if err != nil {
		t.Fatalf("InitRank: %v", err)
	}
	defer comm.Destroy()

	// Allocate a 4-element float32 buffer on GPU.
	n := 4
	byteSize := n * 4
	devPtr, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc: %v", err)
	}
	defer cuda.Free(devPtr)

	// Upload [1, 2, 3, 4].
	host := []float32{1, 2, 3, 4}
	if err := cuda.Memcpy(devPtr, unsafe.Pointer(&host[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer stream.Destroy()

	// In-place all-reduce with 1 rank is a no-op.
	if err := comm.AllReduce(devPtr, devPtr, n, Float32, Sum, stream.Ptr()); err != nil {
		t.Fatalf("AllReduce: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	// Read back.
	result := make([]float32, n)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devPtr, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	for i, want := range host {
		if result[i] != want {
			t.Errorf("result[%d] = %f, want %f", i, result[i], want)
		}
	}
}

func TestTwoGPUAllReduce(t *testing.T) {
	requireNccl(t)
	count, err := cuda.GetDeviceCount()
	if err != nil || count < 2 {
		t.Skip("requires at least 2 CUDA devices")
	}

	uid, err := GetUniqueID()
	if err != nil {
		t.Fatalf("GetUniqueID: %v", err)
	}

	nRanks := 2
	n := 4
	byteSize := n * 4

	var wg sync.WaitGroup
	errs := make([]error, nRanks)
	results := make([][]float32, nRanks)

	for rank := range nRanks {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()

			if err := cuda.SetDevice(rank); err != nil {
				errs[rank] = err
				return
			}

			comm, err := InitRank(uid, nRanks, rank)
			if err != nil {
				errs[rank] = err
				return
			}
			defer comm.Destroy()

			devPtr, err := cuda.Malloc(byteSize)
			if err != nil {
				errs[rank] = err
				return
			}
			defer cuda.Free(devPtr)

			// Rank 0: [1, 2, 3, 4], Rank 1: [10, 20, 30, 40].
			host := make([]float32, n)
			for i := range n {
				host[i] = float32((rank*10 + 1) * (i + 1))
			}
			if err := cuda.Memcpy(devPtr, unsafe.Pointer(&host[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
				errs[rank] = err
				return
			}

			stream, err := cuda.CreateStream()
			if err != nil {
				errs[rank] = err
				return
			}
			defer stream.Destroy()

			if err := comm.AllReduce(devPtr, devPtr, n, Float32, Sum, stream.Ptr()); err != nil {
				errs[rank] = err
				return
			}
			if err := stream.Synchronize(); err != nil {
				errs[rank] = err
				return
			}

			result := make([]float32, n)
			if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devPtr, byteSize, cuda.MemcpyDeviceToHost); err != nil {
				errs[rank] = err
				return
			}
			results[rank] = result
		}(rank)
	}
	wg.Wait()

	for rank, err := range errs {
		if err != nil {
			t.Fatalf("rank %d error: %v", rank, err)
		}
	}

	// Expected: rank0_data + rank1_data element-wise.
	// Rank 0: [1,2,3,4], Rank 1: [11,22,33,44] -> Sum: [12,24,36,48].
	for rank := range nRanks {
		for i := range n {
			want := float32((1)*(i+1)) + float32((11)*(i+1))
			if results[rank][i] != want {
				t.Errorf("rank %d result[%d] = %f, want %f", rank, i, results[rank][i], want)
			}
		}
	}
}

func TestTwoGPUBroadcast(t *testing.T) {
	requireNccl(t)
	count, err := cuda.GetDeviceCount()
	if err != nil || count < 2 {
		t.Skip("requires at least 2 CUDA devices")
	}

	uid, err := GetUniqueID()
	if err != nil {
		t.Fatalf("GetUniqueID: %v", err)
	}

	nRanks := 2
	n := 4
	byteSize := n * 4

	var wg sync.WaitGroup
	errs := make([]error, nRanks)
	results := make([][]float32, nRanks)

	for rank := range nRanks {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()

			if err := cuda.SetDevice(rank); err != nil {
				errs[rank] = err
				return
			}

			comm, err := InitRank(uid, nRanks, rank)
			if err != nil {
				errs[rank] = err
				return
			}
			defer comm.Destroy()

			devPtr, err := cuda.Malloc(byteSize)
			if err != nil {
				errs[rank] = err
				return
			}
			defer cuda.Free(devPtr)

			// Only root (rank 0) has data.
			if rank == 0 {
				host := []float32{42, 43, 44, 45}
				if err := cuda.Memcpy(devPtr, unsafe.Pointer(&host[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
					errs[rank] = err
					return
				}
			}

			stream, err := cuda.CreateStream()
			if err != nil {
				errs[rank] = err
				return
			}
			defer stream.Destroy()

			if err := comm.Broadcast(devPtr, devPtr, n, Float32, 0, stream.Ptr()); err != nil {
				errs[rank] = err
				return
			}
			if err := stream.Synchronize(); err != nil {
				errs[rank] = err
				return
			}

			result := make([]float32, n)
			if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devPtr, byteSize, cuda.MemcpyDeviceToHost); err != nil {
				errs[rank] = err
				return
			}
			results[rank] = result
		}(rank)
	}
	wg.Wait()

	for rank, err := range errs {
		if err != nil {
			t.Fatalf("rank %d error: %v", rank, err)
		}
	}

	// Both ranks should have root's data.
	want := []float32{42, 43, 44, 45}
	for rank := range nRanks {
		for i := range n {
			if results[rank][i] != want[i] {
				t.Errorf("rank %d result[%d] = %f, want %f", rank, i, results[rank][i], want[i])
			}
		}
	}
}

func TestGroupStartEnd(t *testing.T) {
	requireNccl(t)
	// GroupStart/GroupEnd can be called without a communicator.
	if err := GroupStart(); err != nil {
		t.Fatalf("GroupStart: %v", err)
	}
	if err := GroupEnd(); err != nil {
		t.Fatalf("GroupEnd: %v", err)
	}
}

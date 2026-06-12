package tensor

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/internal/gpuapi"
)

// fakeAsyncRuntime models a device whose pending stream work only becomes
// visible at synchronization time. flush is the "pending kernel": the
// host-access sync hook must run it before any host access touches memory.
// Memcpy applies the same rule a real device does NOT enforce for pageable
// memory on coherent platforms: it copies whatever bytes are currently
// visible, pending work included only if the hook already flushed it.
type fakeAsyncRuntime struct{}

func (fakeAsyncRuntime) DeviceType() device.Type        { return device.CUDA }
func (fakeAsyncRuntime) SetDevice(int) error            { return nil }
func (fakeAsyncRuntime) GetDeviceCount() (int, error)   { return 1, nil }
func (fakeAsyncRuntime) Malloc(int) (unsafe.Pointer, error) {
	return nil, errors.New("fakeAsyncRuntime: Malloc not supported")
}
func (fakeAsyncRuntime) Free(unsafe.Pointer) error { return nil }
func (fakeAsyncRuntime) Memcpy(dst, src unsafe.Pointer, count int, _ gpuapi.MemcpyKind) error {
	copy(unsafe.Slice((*byte)(dst), count), unsafe.Slice((*byte)(src), count))
	return nil
}
func (fakeAsyncRuntime) MemcpyAsync(dst, src unsafe.Pointer, count int, kind gpuapi.MemcpyKind, _ gpuapi.Stream) error {
	return fakeAsyncRuntime{}.Memcpy(dst, src, count, kind)
}
func (fakeAsyncRuntime) MemsetAsync(unsafe.Pointer, int, int, gpuapi.Stream) error { return nil }
func (fakeAsyncRuntime) MemcpyPeer(unsafe.Pointer, int, unsafe.Pointer, int, int) error {
	return nil
}
func (fakeAsyncRuntime) CreateStream() (gpuapi.Stream, error) {
	return nil, errors.New("fakeAsyncRuntime: CreateStream not supported")
}

// newFakeDeviceStorage wraps a host slice as GPU storage on deviceID. managed
// selects the direct unified-pointer access path; non-managed goes through
// the runtime Memcpy. Both must be stream-ordered by the sync hook.
func newFakeDeviceStorage(buf []float32, deviceID int, managed bool) *GPUStorage[float32] {
	return &GPUStorage[float32]{
		devicePtr: unsafe.Pointer(unsafe.SliceData(buf)),
		length:    len(buf),
		byteSize:  len(buf) * 4,
		deviceID:  deviceID,
		runtime:   fakeAsyncRuntime{},
		managed:   managed,
		view:      true, // the test owns buf; Free must not touch it
	}
}

// TestHostAccessSync_ReadsAreStreamOrdered is the Bug 11 regression test:
// an async op has written new values, but they only become visible when the
// owning stream synchronizes. A host read that skips the sync observes the
// stale bytes -- exactly the GB10 gradient corruption. Each host-access path
// must invoke the registered hook before touching memory.
func TestHostAccessSync_ReadsAreStreamOrdered(t *testing.T) {
	t.Parallel()

	stale := []float32{1, 2, 3, 4}
	fresh := []float32{10, 20, 30, 40}

	read := map[string]func(s *GPUStorage[float32]) ([]float32, error){
		"TrySlice": func(s *GPUStorage[float32]) ([]float32, error) {
			return s.TrySlice()
		},
		"Slice": func(s *GPUStorage[float32]) ([]float32, error) {
			return s.Slice(), nil
		},
		"CopyTo": func(s *GPUStorage[float32]) ([]float32, error) {
			dst := make([]float32, s.Len())
			if err := s.CopyTo(dst); err != nil {
				return nil, err
			}
			return dst, nil
		},
	}

	for name, fn := range read {
		for _, managed := range []bool{true, false} {
			label := name + "/discrete"
			if managed {
				label = name + "/managed"
			}
			t.Run(label, func(t *testing.T) {
				deviceID := 100 + len(label) // isolate registry state per subtest

				buf := make([]float32, len(stale))
				copy(buf, stale)
				s := newFakeDeviceStorage(buf, deviceID, managed)

				// The "pending kernel": its write becomes visible only when
				// the hook (stream sync) runs.
				synced := false
				unregister := RegisterHostAccessSync(deviceID, func() error {
					synced = true
					copy(buf, fresh)
					return nil
				})
				defer unregister()

				got, err := fn(s)
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				if !synced {
					t.Fatalf("%s: host read did not synchronize the owning stream", name)
				}
				for i := range fresh {
					if got[i] != fresh[i] {
						t.Fatalf("%s: read stale data %v before stream sync; want %v", name, got, fresh)
					}
				}
			})
		}
	}
}

// TestHostAccessSync_WritesAreStreamOrdered: TrySet/Set must synchronize the
// owning stream before overwriting device memory a pending kernel may still
// be reading (or, on resize, freeing it).
func TestHostAccessSync_WritesAreStreamOrdered(t *testing.T) {
	t.Parallel()

	for _, managed := range []bool{true, false} {
		label := "discrete"
		if managed {
			label = "managed"
		}
		t.Run(label, func(t *testing.T) {
			deviceID := 200
			if managed {
				deviceID = 201
			}

			buf := make([]float32, 4)
			s := newFakeDeviceStorage(buf, deviceID, managed)

			synced := false
			unregister := RegisterHostAccessSync(deviceID, func() error {
				synced = true
				return nil
			})
			defer unregister()

			if err := s.TrySet([]float32{5, 6, 7, 8}); err != nil {
				t.Fatalf("TrySet: %v", err)
			}
			if !synced {
				t.Fatal("TrySet: host write did not synchronize the owning stream")
			}
			for i, want := range []float32{5, 6, 7, 8} {
				if buf[i] != want {
					t.Fatalf("TrySet: buf=%v, want [5 6 7 8]", buf)
				}
			}
		})
	}
}

// TestHostAccessSync_HookErrorPropagates: a failing sync must fail the host
// access instead of silently reading unordered bytes.
func TestHostAccessSync_HookErrorPropagates(t *testing.T) {
	t.Parallel()

	const deviceID = 300
	wantErr := errors.New("stream synchronize failed")
	unregister := RegisterHostAccessSync(deviceID, func() error { return wantErr })
	defer unregister()

	buf := make([]float32, 2)
	s := newFakeDeviceStorage(buf, deviceID, true)

	if _, err := s.TrySlice(); !errors.Is(err, wantErr) {
		t.Fatalf("TrySlice error = %v, want wrapped %v", err, wantErr)
	}
	if err := s.CopyTo(make([]float32, 2)); !errors.Is(err, wantErr) {
		t.Fatalf("CopyTo error = %v, want wrapped %v", err, wantErr)
	}
	if err := s.TrySet([]float32{1, 2}); !errors.Is(err, wantErr) {
		t.Fatalf("TrySet error = %v, want wrapped %v", err, wantErr)
	}
}

// TestHostAccessSync_Registry: hooks are per-device, support multiple
// registrations, and stop firing after unregister.
func TestHostAccessSync_Registry(t *testing.T) {
	t.Parallel()

	const deviceID = 400
	var aCalls, bCalls int
	unregA := RegisterHostAccessSync(deviceID, func() error { aCalls++; return nil })
	unregB := RegisterHostAccessSync(deviceID, func() error { bCalls++; return nil })
	unregOther := RegisterHostAccessSync(deviceID+1, func() error {
		t.Error("hook for another device must not fire")
		return nil
	})
	defer unregOther()

	if err := syncForHostAccess(deviceID); err != nil {
		t.Fatalf("syncForHostAccess: %v", err)
	}
	if aCalls != 1 || bCalls != 1 {
		t.Fatalf("after first sync: aCalls=%d bCalls=%d, want 1 1", aCalls, bCalls)
	}

	unregA()
	if err := syncForHostAccess(deviceID); err != nil {
		t.Fatalf("syncForHostAccess: %v", err)
	}
	if aCalls != 1 || bCalls != 2 {
		t.Fatalf("after unregister: aCalls=%d bCalls=%d, want 1 2", aCalls, bCalls)
	}
	unregB()

	// Devices with no hooks sync nothing and succeed.
	if err := syncForHostAccess(deviceID); err != nil {
		t.Fatalf("syncForHostAccess with no hooks: %v", err)
	}
}

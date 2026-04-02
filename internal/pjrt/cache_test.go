package pjrt

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestCacheKey(t *testing.T) {
	// Same inputs produce same key.
	k1 := Key("module { func.func @main() {} }", "cpu")
	k2 := Key("module { func.func @main() {} }", "cpu")
	if k1 != k2 {
		t.Fatalf("same input produced different keys: %s vs %s", k1, k2)
	}

	// Different programs produce different keys.
	k3 := Key("module { func.func @other() {} }", "cpu")
	if k1 == k3 {
		t.Fatal("different programs produced same key")
	}

	// Different platforms produce different keys.
	k4 := Key("module { func.func @main() {} }", "cuda")
	if k1 == k4 {
		t.Fatal("different platforms produced same key")
	}

	// Key is hex-encoded SHA256 (64 chars).
	if len(k1) != 64 {
		t.Fatalf("expected 64-char hex key, got %d chars", len(k1))
	}
}

func TestCacheMiss(t *testing.T) {
	dir := t.TempDir()
	c := NewCache(WithCacheDir(dir))

	data, err := c.Get("nonexistent")
	if err != nil {
		t.Fatalf("unexpected error on miss: %v", err)
	}
	if data != nil {
		t.Fatal("expected nil data on cache miss")
	}

	stats := c.Stats()
	if stats.Misses != 1 {
		t.Fatalf("expected 1 miss, got %d", stats.Misses)
	}
	if stats.Hits != 0 {
		t.Fatalf("expected 0 hits, got %d", stats.Hits)
	}
}

func TestCachePutGet(t *testing.T) {
	dir := t.TempDir()
	c := NewCache(WithCacheDir(dir))

	key := Key("program1", "cpu")
	payload := []byte("serialized-executable-bytes")

	if err := c.Put(key, payload); err != nil {
		t.Fatalf("Put: %v", err)
	}

	data, err := c.Get(key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(data) != string(payload) {
		t.Fatalf("Get returned %q, want %q", data, payload)
	}

	stats := c.Stats()
	if stats.Hits != 1 {
		t.Fatalf("expected 1 hit, got %d", stats.Hits)
	}
	if stats.Files != 1 {
		t.Fatalf("expected 1 file, got %d", stats.Files)
	}
	if stats.Size != int64(len(payload)) {
		t.Fatalf("expected size %d, got %d", len(payload), stats.Size)
	}
}

func TestCacheLRUEviction(t *testing.T) {
	dir := t.TempDir()
	// Max size = 50 bytes. Each entry is 20 bytes.
	c := NewCache(WithCacheDir(dir), WithMaxCacheSize(50))

	k1 := Key("prog1", "cpu")
	k2 := Key("prog2", "cpu")
	k3 := Key("prog3", "cpu")

	d := make([]byte, 20)

	// Put k1, then sleep briefly so mod time differs.
	if err := c.Put(k1, d); err != nil {
		t.Fatal(err)
	}
	// Backdate k1 so it's the oldest.
	p1 := c.entryPath(k1)
	old := time.Now().Add(-2 * time.Second)
	if err := os.Chtimes(p1, old, old); err != nil {
		t.Fatal(err)
	}

	if err := c.Put(k2, d); err != nil {
		t.Fatal(err)
	}

	// At this point: 40 bytes total, under 50 limit. Both should exist.
	stats := c.Stats()
	if stats.Files != 2 {
		t.Fatalf("expected 2 files before eviction, got %d", stats.Files)
	}

	// Put k3: total would be 60 bytes, exceeding 50. k1 (oldest) evicted.
	if err := c.Put(k3, d); err != nil {
		t.Fatal(err)
	}

	stats = c.Stats()
	if stats.Files != 2 {
		t.Fatalf("expected 2 files after eviction, got %d", stats.Files)
	}
	if stats.Size != 40 {
		t.Fatalf("expected size 40 after eviction, got %d", stats.Size)
	}

	// k1 should be evicted.
	data, _ := c.Get(k1)
	if data != nil {
		t.Fatal("expected k1 to be evicted")
	}

	// k2 and k3 should still exist.
	data, _ = c.Get(k2)
	if data == nil {
		t.Fatal("expected k2 to still exist")
	}
	data, _ = c.Get(k3)
	if data == nil {
		t.Fatal("expected k3 to still exist")
	}
}

func TestCacheClear(t *testing.T) {
	dir := t.TempDir()
	c := NewCache(WithCacheDir(dir))

	for i := 0; i < 5; i++ {
		key := Key(string(rune('a'+i)), "cpu")
		if err := c.Put(key, []byte("data")); err != nil {
			t.Fatal(err)
		}
	}

	stats := c.Stats()
	if stats.Files != 5 {
		t.Fatalf("expected 5 files, got %d", stats.Files)
	}

	if err := c.Clear(); err != nil {
		t.Fatalf("Clear: %v", err)
	}

	stats = c.Stats()
	if stats.Files != 0 {
		t.Fatalf("expected 0 files after clear, got %d", stats.Files)
	}
}

func TestCacheAtomicWrite(t *testing.T) {
	dir := t.TempDir()
	c := NewCache(WithCacheDir(dir))

	key := Key("atomic-test", "cpu")
	if err := c.Put(key, []byte("good-data")); err != nil {
		t.Fatal(err)
	}

	// Verify no .tmp files remain.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".tmp" {
			t.Fatalf("found leftover .tmp file: %s", e.Name())
		}
	}

	data, _ := c.Get(key)
	if string(data) != "good-data" {
		t.Fatalf("expected 'good-data', got %q", data)
	}
}

func TestCacheOverwrite(t *testing.T) {
	dir := t.TempDir()
	c := NewCache(WithCacheDir(dir))

	key := Key("overwrite-test", "cpu")
	if err := c.Put(key, []byte("v1")); err != nil {
		t.Fatal(err)
	}
	if err := c.Put(key, []byte("v2")); err != nil {
		t.Fatal(err)
	}

	data, err := c.Get(key)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "v2" {
		t.Fatalf("expected 'v2', got %q", data)
	}

	stats := c.Stats()
	if stats.Files != 1 {
		t.Fatalf("expected 1 file after overwrite, got %d", stats.Files)
	}
}

func TestCacheEnvVar(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ZERFOO_PJRT_CACHE", dir)

	c := NewCache()
	if c.Dir() != dir {
		t.Fatalf("expected dir %s from env, got %s", dir, c.Dir())
	}
}

func TestCacheManualEvict(t *testing.T) {
	dir := t.TempDir()
	c := NewCache(WithCacheDir(dir), WithMaxCacheSize(10))

	// Put 30 bytes, which exceeds budget.
	k1 := Key("evict1", "cpu")
	k2 := Key("evict2", "cpu")
	if err := c.Put(k1, make([]byte, 15)); err != nil {
		t.Fatal(err)
	}
	// Backdate k1.
	old := time.Now().Add(-2 * time.Second)
	_ = os.Chtimes(c.entryPath(k1), old, old)

	if err := c.Put(k2, make([]byte, 15)); err != nil {
		t.Fatal(err)
	}

	// After Put(k2), eviction already ran. But let's also test manual Evict().
	c.maxSize = 5
	c.Evict()

	stats := c.Stats()
	if stats.Size > 5 {
		t.Fatalf("expected size <= 5 after manual evict, got %d", stats.Size)
	}
}

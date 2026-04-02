package pjrt

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// DefaultCacheDir is the default directory for cached PJRT executables.
const DefaultCacheDir = ".cache/zerfoo/pjrt"

// DefaultMaxCacheSize is the default maximum cache size in bytes (2 GB).
const DefaultMaxCacheSize int64 = 2 << 30

// CacheOption configures a Cache.
type CacheOption func(*Cache)

// WithCacheDir sets the cache directory. If empty, defaults to
// $ZERFOO_PJRT_CACHE or ~/.cache/zerfoo/pjrt/.
func WithCacheDir(dir string) CacheOption {
	return func(c *Cache) { c.dir = dir }
}

// WithMaxCacheSize sets the maximum total size of cached files in bytes.
func WithMaxCacheSize(n int64) CacheOption {
	return func(c *Cache) { c.maxSize = n }
}

// CacheStats holds cache hit/miss/size statistics.
type CacheStats struct {
	Hits   int64
	Misses int64
	Size   int64 // total bytes on disk
	Files  int   // number of cached entries
}

// Cache stores serialized PJRT executables keyed by a content hash of
// the StableHLO program text and platform name. It provides LRU eviction
// when the total size exceeds MaxSize.
type Cache struct {
	mu      sync.Mutex
	dir     string
	maxSize int64
	hits    int64
	misses  int64
}

// NewCache creates a new executable cache. The cache directory is created
// on first Put if it does not already exist.
func NewCache(opts ...CacheOption) *Cache {
	c := &Cache{maxSize: DefaultMaxCacheSize}
	for _, o := range opts {
		o(c)
	}
	if c.dir == "" {
		c.dir = resolveCacheDir()
	}
	return c
}

// Key returns the content-addressed cache key for the given StableHLO
// program and platform name: SHA256(program + "\x00" + platform).
func Key(stablehloMLIR, platformName string) string {
	h := sha256.New()
	h.Write([]byte(stablehloMLIR))
	h.Write([]byte{0})
	h.Write([]byte(platformName))
	return hex.EncodeToString(h.Sum(nil))
}

// Get looks up a cached serialized executable by key. If found, the raw
// bytes are returned (caller must DeserializeAndLoad). Returns nil, nil
// on cache miss.
func (c *Cache) Get(key string) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	path := c.entryPath(key)
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		c.misses++
		return nil, nil
	}
	if err != nil {
		c.misses++
		return nil, fmt.Errorf("pjrt cache: read %s: %w", key, err)
	}

	// Touch access time for LRU tracking.
	now := time.Now()
	_ = os.Chtimes(path, now, now)

	c.hits++
	return data, nil
}

// Put stores serialized executable bytes under the given key. If storing
// the new entry would exceed MaxSize, the least-recently-used entries are
// evicted first.
func (c *Cache) Put(key string, data []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return fmt.Errorf("pjrt cache: create dir: %w", err)
	}

	path := c.entryPath(key)

	// Write atomically: write to tmp then rename.
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("pjrt cache: write %s: %w", key, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("pjrt cache: rename %s: %w", key, err)
	}

	// Evict if over budget.
	c.evictLocked()
	return nil
}

// Evict removes the least-recently-used entries until total cache size
// is within MaxSize.
func (c *Cache) Evict() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.evictLocked()
}

// Clear removes all cached entries.
func (c *Cache) Clear() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	entries, _ := os.ReadDir(c.dir)
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		_ = os.Remove(filepath.Join(c.dir, e.Name()))
	}
	return nil
}

// Stats returns current cache statistics.
func (c *Cache) Stats() CacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()

	var totalSize int64
	var fileCount int
	entries, _ := os.ReadDir(c.dir)
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		totalSize += info.Size()
		fileCount++
	}

	return CacheStats{
		Hits:   c.hits,
		Misses: c.misses,
		Size:   totalSize,
		Files:  fileCount,
	}
}

// Dir returns the cache directory path.
func (c *Cache) Dir() string {
	return c.dir
}

// entryPath returns the filesystem path for a cache key.
func (c *Cache) entryPath(key string) string {
	return filepath.Join(c.dir, key+".pjrt")
}

// cacheEntry holds file info for LRU sorting.
type cacheEntry struct {
	path    string
	size    int64
	modTime time.Time
}

// evictLocked removes LRU entries until total size <= maxSize. Caller must hold mu.
func (c *Cache) evictLocked() {
	entries, err := os.ReadDir(c.dir)
	if err != nil {
		return
	}

	var files []cacheEntry
	var totalSize int64
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		files = append(files, cacheEntry{
			path:    filepath.Join(c.dir, e.Name()),
			size:    info.Size(),
			modTime: info.ModTime(),
		})
		totalSize += info.Size()
	}

	if totalSize <= c.maxSize {
		return
	}

	// Sort oldest first (least recently used).
	sort.Slice(files, func(i, j int) bool {
		return files[i].modTime.Before(files[j].modTime)
	})

	for _, f := range files {
		if totalSize <= c.maxSize {
			break
		}
		if err := os.Remove(f.path); err == nil {
			totalSize -= f.size
		}
	}
}

// resolveCacheDir returns the cache directory, checking ZERFOO_PJRT_CACHE
// env var first, then falling back to ~/.cache/zerfoo/pjrt/.
func resolveCacheDir() string {
	if dir := os.Getenv("ZERFOO_PJRT_CACHE"); dir != "" {
		return dir
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "zerfoo-pjrt-cache")
	}
	return filepath.Join(home, DefaultCacheDir)
}


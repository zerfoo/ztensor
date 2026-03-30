# Quality Standards

This document describes the testing and code quality policies for ztensor.

## Test Coverage

All new code must include tests. Coverage is not enforced by a hard numeric
threshold, but reviewers should ensure that non-trivial logic paths are
exercised. The CI pipeline runs `go test -race -timeout 300s ./...` on every
push and pull request against `main`.

## GPU-Only Test Tagging

Tests and source files that require a GPU use build tags to keep the default
`go test ./...` runnable on any machine:

| Tag      | When to use                                    |
|----------|------------------------------------------------|
| `cuda`   | Requires an NVIDIA GPU with CUDA runtime       |
| `rocm`   | Requires an AMD GPU with ROCm/HIP runtime      |
| `opencl` | Requires an OpenCL-capable device               |
| `sycl`   | Requires a SYCL runtime (Intel oneAPI)          |
| `fpga`   | Requires FPGA hardware                          |

Apply the tag at the top of the file:

```go
//go:build cuda
```

Files that provide a CPU fallback when no GPU is present use the negated form:

```go
//go:build !cuda
```

Run GPU tests explicitly:

```bash
go test -tags cuda ./...
```

## Race Detector

CI runs the race detector (`-race`) on every test invocation. All tests must
pass under the race detector. If a test legitimately cannot run with `-race`
(e.g., performance benchmarks that are too slow), skip it:

```go
func BenchmarkXxx(b *testing.B) {
    // benchmarks are not run in CI; -race is acceptable overhead
}
```

Do not disable the race detector in CI.

## Vet Exclusions

The CI workflow excludes the following packages from `go vet`:

- `internal/cuda`
- `internal/hip`
- `internal/opencl`
- `internal/cudnn`
- `internal/tensorrt`
- `internal/fpga`
- `internal/sycl`

### Rationale

These packages use `unsafe.Pointer` conversions to pass Go values to GPU
runtime libraries loaded at runtime via purego/dlopen. The `go vet`
`unsafeptr` check flags these conversions because it cannot verify that the
pointer remains valid across the FFI boundary. In ztensor this usage is
intentional and well-scoped: each binding function converts a Go pointer to
`uintptr` immediately before the purego call and does not retain it afterward.

Excluding these packages from `go vet` keeps CI signal clean while allowing
the necessary low-level GPU interop. All other packages in the module are
vetted normally.

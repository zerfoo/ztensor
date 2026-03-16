# Contributing to ztensor

Thank you for your interest in contributing to ztensor, the GPU-accelerated tensor, compute engine, and computation graph library for the Zerfoo ML ecosystem. This guide will help you get started.

## Table of Contents

- [Development Setup](#development-setup)
- [Building from Source](#building-from-source)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Good First Issues](#good-first-issues)
- [Key Conventions](#key-conventions)

## Development Setup

### Prerequisites

- **Go 1.25+** (generics with `tensor.Numeric` constraint)
- **Git**
- **CUDA Toolkit** (optional, for GPU tests — CUDA 13.0 recommended)
- **ROCm** (optional, for AMD GPU tests)
- **OpenCL** (optional, for CLBlast backend tests)

### Clone and Verify

```bash
git clone https://github.com/zerfoo/ztensor.git
cd ztensor
go mod tidy
go test ./...
```

ztensor depends on:

- [`github.com/zerfoo/float16`](https://github.com/zerfoo/float16) — IEEE 754 half-precision arithmetic
- [`github.com/zerfoo/float8`](https://github.com/zerfoo/float8) — FP8 E4M3FN arithmetic
- [`gonum.org/v1/gonum`](https://github.com/gonum/gonum) — numerical routines

These are fetched automatically by `go mod tidy`.

## Building from Source

```bash
go build ./...
```

No CGo is required for CPU-only builds. GPU support is loaded dynamically at runtime via purego/dlopen, so `go build` works on any platform without a CUDA toolkit installed.

## Running Tests

```bash
# Run all CPU tests (no GPU required)
go test ./...

# Run tests with race detector
go test -race ./...

# Run GPU tests — CUDA backend (requires CUDA toolkit and a GPU)
go test -tags cuda ./...

# Run GPU tests — ROCm backend (requires ROCm and an AMD GPU)
go test -tags rocm ./...

# Run GPU tests — OpenCL backend (requires OpenCL runtime)
go test -tags opencl ./...

# Run tests with coverage
go test -cover ./...
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html
```

GPU tests are skipped automatically when no GPU is available. All new code must have tests. Aim for at least 80% coverage on new packages.

## Code Style

### Formatting and Linting

- **`gofmt`** — all code must be formatted with `gofmt`
- **`goimports`** — imports must be organized (stdlib, external, internal)
- **`golangci-lint`** — run `golangci-lint run` before submitting

### Go Conventions

- Follow standard Go naming: PascalCase for exported symbols, camelCase for unexported
- Use table-driven tests with `t.Run` subtests
- Write documentation comments for all exported functions, types, and methods
- Use generics with `[T tensor.Numeric]` constraints — avoid type-specific code where generics work

## Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/) for automated versioning with release-please.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `perf` | A performance improvement |
| `docs` | Documentation only changes |
| `test` | Adding or correcting tests |
| `chore` | Maintenance tasks, CI, dependencies |
| `refactor` | Code change that neither fixes a bug nor adds a feature |

### Examples

```
feat(compute): add ROCm backend for Engine[T]
fix(graph): correct topological sort for diamond dependencies
perf(cuda): fuse elementwise ops in CUDA graph capture
docs(tensor): document memory layout for quantized types
test(device): add multi-GPU allocation tests
```

## Pull Request Process

1. **One logical change per PR** — keep PRs focused and reviewable
2. **Branch from `main`** and keep your branch up to date with rebase
3. **All CI checks must pass** — tests, linting, formatting
4. **Rebase and merge** — we do not use squash merges or merge commits
5. **Reference related issues** — use `Fixes #123` or `Closes #123` in the PR description
6. **Respond to review feedback** promptly

### Before Submitting

```bash
go test ./...
go test -race ./...
go vet ./...
golangci-lint run
```

## Issue Reporting

### Bug Reports

Please include:

- **Description**: Clear summary of the bug
- **Steps to reproduce**: Minimal code or commands to trigger the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What happens instead
- **Environment**: Go version, OS, architecture, GPU model and driver version (if GPU-related)

### Feature Requests

Please include:

- **Problem statement**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you thought about
- **Use case**: How would you use this feature in practice?

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/zerfoo/ztensor/labels/good%20first%20issue) on GitHub. These are scoped, well-defined tasks suitable for new contributors.

Good areas for first contributions:

- Adding test coverage for existing packages
- Documentation improvements
- CPU compute engine optimizations
- New numeric type support in `tensor/`

## Key Conventions

These conventions are critical to maintaining consistency across the codebase:

### Engine[T] interface is law

All tensor arithmetic must flow through `compute.Engine[T]`. Never operate on raw slices outside the engine — this enables transparent CPU/GPU switching and CUDA graph capture.

```go
// Good
engine.MatMul(ctx, out, a, b)

// Bad — bypasses the engine, breaks GPU support
for i := range out.Data() {
    out.Data()[i] = a.Data()[i] * b.Data()[i]
}
```

### No CGo by default

GPU bindings use purego/dlopen. A plain `go build ./...` must compile on any platform without a C compiler. Build tags (`cuda`, `rocm`, `opencl`) are optional and only used for CGo-based alternative paths.

### purego GPU bindings

All GPU runtime calls (CUDA, ROCm, OpenCL) are loaded dynamically via purego. This means:

- Function signatures are declared as Go types and resolved at runtime
- No `#cgo` directives or C header includes
- GPU availability is detected at runtime, not compile time

### Assembly SIMD code

ARM NEON and x86 AVX2 assembly lives in `internal/xblas/`. When writing or modifying assembly:

- Keep Go fallback implementations in sync
- Test both assembly and pure-Go paths
- Use `go test -tags noasm` to verify the Go fallback

### Generics throughout

Use `[T tensor.Numeric]` constraints. Do not write float32-specific code where generics work.

### GPU Runtime Abstraction Layer (GRAL)

The `internal/gpuapi/` package provides a unified interface across CUDA, ROCm, and OpenCL. New GPU features should be added through GRAL, not directly to a single backend.

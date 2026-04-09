# ADR-002: Migrate internal/nccl from CGo to purego/dlopen

**Status:** Accepted
**Date:** 2026-04-09
**Authors:** David Ndungu

## Context

The original `internal/nccl` package was a CGo binding (`#include <nccl.h>`,
`-lnccl`) gated behind `//go:build cuda`. This forced any build that wanted
NCCL to link against the system NCCL headers/library at compile time and
exposed the package only when the `cuda` build tag was set. It also broke the
project-wide rule that `go build ./...` must succeed on every supported
platform without CGo.

Every other GPU runtime binding in ztensor (cuBLAS, cuDNN, cuRAND, TensorRT,
HIP/ROCm, OpenCL) is already loaded at runtime via `internal/cuda.DlopenPath`
and `internal/cuda.Ccall`. NCCL was the lone holdout.

## Decision

`internal/nccl` is implemented in Go-only via runtime dlopen of
`libnccl.so.2`, mirroring the pattern in `internal/cublas/cublas_purego.go`.

Key points:

- **No build tag** on `nccl_purego.go` — it compiles on every platform.
- On non-linux GOOS, `loadNccl` returns a `nccl: not supported on $GOOS`
  error without attempting `dlopen`. Every exported entry point surfaces this
  as a clean error rather than a panic.
- ABI constants (`ncclSuccess`, the data-type and reduction-op enums, and
  `NCCL_UNIQUE_ID_BYTES = 128`) are hardcoded against the stable NCCL 2.x ABI.
- `UniqueID` is marshaled as a fixed-size `[128]byte` array. `Bytes()` and
  `UniqueIDFromBytes` provide the serialization round-trip used to ferry the
  bootstrap blob between ranks.
- The legacy CGo implementation is retained as `nccl_cgo.go` behind
  `//go:build cuda && cgo && nccl_cgo`. It is OFF by default and exists only
  as a debugging fallback if a future NCCL release introduces an ABI quirk
  that the dlopen path cannot handle.

### AArch64 hidden-pointer ABI for ncclCommInitRank

`ncclCommInitRank` takes the 128-byte `ncclUniqueId` **by value**:

```c
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                              ncclUniqueId commId, int rank);
```

The shared `cuda.Ccall` trampoline only marshals `uintptr`-sized arguments,
so passing 128 bytes by value is not directly possible. Fortunately the only
supported NCCL platform for ztensor is **linux/arm64** (the DGX Spark host).
Per AAPCS64 §6.8.2 rule B.4, *"if the argument type is a Composite Type that
is larger than 16 bytes, then the argument is copied to memory allocated by
the caller and the argument is replaced by a pointer to the copy."* That
means at the ABI level the third argument is already a pointer; passing
`uintptr(unsafe.Pointer(&uid.id[0]))` is the correct calling convention.

If we ever need to support NCCL on linux/amd64 (System V ABI), this trick
will not work — SysV passes large aggregates on the stack by value — and we
will need a small assembly trampoline, or we can fall back to the CGo path
via the `nccl_cgo` build tag.

## Consequences

- `go build ./...` works everywhere with no `-tags cuda` and no system NCCL
  installed.
- Tests in `internal/nccl/nccl_test.go` no longer require a build tag; they
  call `requireNccl(t)` to skip when `libnccl.so.2` is not dlopen-able. Pure
  constant and marshaling tests run on every platform.
- The duplicate `internal/nccl` copy inside the `zerfoo` repository is **not**
  touched by this change and should be migrated in a follow-up.
- CI's `go vet` exclude list adds `/internal/nccl$` since the dlopen
  trampoline relies on the same `unsafe.Pointer(uintptr(...))` pattern as
  every other GPU runtime binding (already documented in `docs/QUALITY.md`).

## References

- Issue: zerfoo/ztensor#78
- Reference pattern: `internal/cublas/cublas_purego.go`
- AAPCS64: <https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst>

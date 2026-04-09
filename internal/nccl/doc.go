// Package nccl provides a zero-CGo binding for the NVIDIA Collective
// Communications Library (NCCL). The library is loaded at runtime via dlopen
// (see nccl_purego.go); a legacy CGo implementation is retained behind the
// `nccl_cgo` build tag for opt-in fallback (nccl_cgo.go).
package nccl

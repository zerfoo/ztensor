//go:build !cuda

package kernels

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
	"unsafe"
)

// --- ztensor#180 regression -------------------------------------------------
//
// openKernelLib treats a set of kernel symbols as OPTIONAL: when dlsym cannot
// find one in the deployed libkernels.so it leaves the function pointer at 0
// and continues, on the contract (purego.go) that "callers check before use".
//
// RepeatInterleaveF32 did not check. On the GB10 the deployed
// /opt/zerfoo/lib/libkernels.so does not export launch_repeat_interleave_f32,
// so the wrapper called cuda.Ccall(0, ...) -- a jump to address 0. That is a
// SIGSEGV with PC=0x0 inside cgo, which kills the process outright: it is not
// an error return, so no caller fallback can run. Every GQA model on GPU
// (Llama, Mistral, Qwen, Gemma) died there. See ztensor#180.

// stubKernelLib makes klib() return lib for the duration of the test.
//
// It deliberately forces the real load FIRST so that kernelLibOnce is consumed
// by the genuine dlopen path, never by this stub -- otherwise a test running
// after this one would silently see "no kernel lib" on a machine that has one.
func stubKernelLib(t *testing.T, lib *KernelLib) {
	t.Helper()
	_, _ = openKernelLib()
	prevLib, prevErr := kernelLib, errKernelLib
	kernelLib, errKernelLib = lib, nil
	t.Cleanup(func() { kernelLib, errKernelLib = prevLib, prevErr })
}

// TestRepeatInterleaveF32_UnresolvedSymbol_ReturnsError is the ztensor#180
// regression. With a KernelLib whose launchRepeatInterleaveF32 is 0 -- exactly
// the state produced by a libkernels.so that predates the kernel -- the
// wrapper must return an error. Before the fix this test does not fail, it
// CRASHES the test binary with SIGSEGV PC=0x0.
func TestRepeatInterleaveF32_UnresolvedSymbol_ReturnsError(t *testing.T) {
	// handle is non-zero: the library loaded fine, only the optional symbol
	// is missing. This is the real-world state, and it is what makes the
	// k == nil check insufficient.
	stubKernelLib(t, &KernelLib{handle: 1})

	// Non-nil, non-null buffers: the issue's original hypothesis was an
	// unallocated dst, so pass plainly valid pointers to rule that out. The
	// only null in play is the function pointer.
	in := make([]float32, 1*2*2*8)
	out := make([]float32, 1*4*2*8)

	err := RepeatInterleaveF32(
		unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]),
		1, 2, 2, 8, 2,
		nil,
	)
	if err == nil {
		t.Fatal("RepeatInterleaveF32 returned nil error with an unresolved kernel symbol; " +
			"the caller fallback can never engage")
	}
	// Sensitivity: the error must name the missing SYMBOL, not merely report
	// that kernels are unavailable. A blanket "kernels not available" would
	// satisfy err != nil while hiding which of the two distinct failures
	// occurred, and would pass even if the guard were removed again and
	// replaced with an unconditional error.
	if !strings.Contains(err.Error(), "launch_repeat_interleave_f32") {
		t.Fatalf("error does not identify the unresolved symbol: %v", err)
	}
}

// TestRepeatInterleaveF32_NoKernelLib_ReturnsDistinctError is the other half of
// the sensitivity pair: when the library itself is absent the wrapper must
// report THAT, not the missing-symbol condition. Two distinguishable errors
// prove the guard discriminates rather than failing blanket.
func TestRepeatInterleaveF32_NoKernelLib_ReturnsDistinctError(t *testing.T) {
	stubKernelLib(t, nil)

	in := make([]float32, 8)
	out := make([]float32, 16)
	err := RepeatInterleaveF32(
		unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]),
		1, 1, 1, 8, 2,
		nil,
	)
	if err == nil {
		t.Fatal("RepeatInterleaveF32 returned nil error with no kernel library")
	}
	if strings.Contains(err.Error(), "launch_repeat_interleave_f32") {
		t.Fatalf("no-library case reported as a missing-symbol case: %v", err)
	}
}

// --- class gate -------------------------------------------------------------

var (
	symBindingRe  = regexp.MustCompile(`\{"([A-Za-z0-9_]+)",\s*&k\.([A-Za-z0-9_]+)\}`)
	optionalKeyRe = regexp.MustCompile(`^\s*"([A-Za-z0-9_]+)":\s*true,`)
)

// loadSymbolTables parses purego.go for the symbol -> struct-field bindings and
// for the optionalSyms key set.
func loadSymbolTables(t *testing.T) (fieldForSym map[string]string, optional map[string]bool) {
	t.Helper()
	src, err := os.ReadFile("purego.go")
	if err != nil {
		t.Fatalf("read purego.go: %v", err)
	}
	text := string(src)

	fieldForSym = map[string]string{}
	for _, m := range symBindingRe.FindAllStringSubmatch(text, -1) {
		fieldForSym[m[1]] = m[2]
	}

	optional = map[string]bool{}
	inBlock := false
	for _, line := range strings.Split(text, "\n") {
		switch {
		case strings.Contains(line, "optionalSyms := map"):
			inBlock = true
		case inBlock && strings.TrimSpace(line) == "}":
			inBlock = false
		case inBlock:
			if m := optionalKeyRe.FindStringSubmatch(line); m != nil {
				optional[m[1]] = true
			}
		}
	}

	// Anti-vacuous: if either extraction silently returned nothing (a format
	// change in purego.go), this gate would pass while checking nothing.
	if len(fieldForSym) < 100 {
		t.Fatalf("symbol/field extraction found only %d bindings; parser is out of "+
			"sync with purego.go and this gate is not checking anything", len(fieldForSym))
	}
	if len(optional) < 20 {
		t.Fatalf("optionalSyms extraction found only %d entries; parser is out of "+
			"sync with purego.go", len(optional))
	}
	return fieldForSym, optional
}

// TestOptionalKernelSymbolsAreGuarded enforces the contract stated at
// purego.go's optionalSyms ("callers check before use") across the whole
// package: every cuda.Ccall whose function pointer is an OPTIONAL kernel
// symbol must sit in a function that first compares that pointer to 0.
//
// This is the class gate for ztensor#180. RepeatInterleaveF32 was not a
// one-off: an optional symbol is exactly the symbol most likely to be absent
// from a deployed libkernels.so, and an unguarded launch of it is an
// unrecoverable process kill rather than a fallback-able error.
func TestOptionalKernelSymbolsAreGuarded(t *testing.T) {
	fieldForSym, optional := loadSymbolTables(t)

	optionalField := map[string]string{} // field -> symbol name
	for sym := range optional {
		f, ok := fieldForSym[sym]
		if !ok {
			t.Errorf("optionalSyms lists %q but no {%q, &k.Field} binding exists", sym, sym)
			continue
		}
		optionalField[f] = sym
	}

	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	fset := token.NewFileSet()

	type violation struct{ file, fn, field, sym string }
	var violations []violation
	callSites := 0

	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") || file == "purego.go" {
			continue
		}
		f, perr := parser.ParseFile(fset, file, nil, 0)
		if perr != nil {
			t.Fatalf("parse %s: %v", file, perr)
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			guarded := guardedFields(fn.Body)
			for _, field := range ccallFields(fn.Body) {
				callSites++
				sym, isOptional := optionalField[field]
				if !isOptional || guarded[field] {
					continue
				}
				violations = append(violations, violation{file, fn.Name.Name, field, sym})
			}
		}
	}

	// Anti-vacuous: a walker that matches no call sites proves nothing.
	if callSites == 0 {
		t.Fatal("found no cuda.Ccall(k.<field>, ...) call sites; the AST walker is " +
			"not matching and this gate is not checking anything")
	}
	t.Logf("checked %d cuda.Ccall sites against %d optional symbols", callSites, len(optionalField))

	sort.Slice(violations, func(i, j int) bool { return violations[i].sym < violations[j].sym })
	for _, v := range violations {
		t.Errorf("%s: %s launches optional symbol %s (k.%s) without checking it is non-zero; "+
			"a missing symbol makes this a SIGSEGV, not an error (ztensor#180)",
			v.file, v.fn, v.sym, v.field)
	}
}

// ccallFields returns the KernelLib field names used as the function-pointer
// argument of cuda.Ccall within body.
func ccallFields(body *ast.BlockStmt) []string {
	var out []string
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) == 0 {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Ccall" {
			return true
		}
		if pkg, ok := sel.X.(*ast.Ident); !ok || pkg.Name != "cuda" {
			return true
		}
		if field, ok := kernelLibField(call.Args[0]); ok {
			out = append(out, field)
		}
		return true
	})
	return out
}

// guardedFields returns the KernelLib field names compared against 0 anywhere
// in body.
func guardedFields(body *ast.BlockStmt) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(body, func(n ast.Node) bool {
		bin, ok := n.(*ast.BinaryExpr)
		if !ok || (bin.Op != token.EQL && bin.Op != token.NEQ) {
			return true
		}
		for _, pair := range [2][2]ast.Expr{{bin.X, bin.Y}, {bin.Y, bin.X}} {
			field, isField := kernelLibField(pair[0])
			lit, isLit := pair[1].(*ast.BasicLit)
			if isField && isLit && lit.Kind == token.INT && lit.Value == "0" {
				out[field] = true
			}
		}
		return true
	})
	return out
}

// kernelLibField reports whether e is a selector on the local KernelLib
// variable (conventionally `k`), returning the field name.
func kernelLibField(e ast.Expr) (string, bool) {
	sel, ok := e.(*ast.SelectorExpr)
	if !ok {
		return "", false
	}
	recv, ok := sel.X.(*ast.Ident)
	if !ok || recv.Name != "k" {
		return "", false
	}
	return sel.Sel.Name, true
}

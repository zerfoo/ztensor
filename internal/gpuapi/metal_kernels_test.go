package gpuapi_test

import (
	"encoding/binary"
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/gpuapi"
	"github.com/zerfoo/ztensor/internal/metal"
)

// metalTestEnv sets up a Metal runtime + compute context for testing.
// Returns nil if Metal is not available.
type metalTestEnv struct {
	rt *gpuapi.MetalRuntime
	cc *metal.ComputeContext
	k  *gpuapi.MetalKernels
}

func newMetalTestEnv(t *testing.T) *metalTestEnv {
	t.Helper()
	if runtime.GOOS != "darwin" {
		t.Skip("Metal is only available on darwin")
	}
	if !metal.Available() {
		t.Skip("Metal framework not available")
	}

	rt := gpuapi.NewMetalRuntime()
	if rt == nil {
		t.Fatal("NewMetalRuntime returned nil")
	}
	if err := rt.SetDevice(0); err != nil {
		t.Fatalf("SetDevice: %v", err)
	}

	ctx, err := metal.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	cc, err := metal.NewComputeContext(ctx)
	if err != nil {
		t.Fatalf("NewComputeContext: %v", err)
	}

	return &metalTestEnv{
		rt: rt,
		cc: cc,
		k:  gpuapi.NewMetalKernelsWithCompute(cc),
	}
}

// allocF32 allocates a Metal buffer with n float32 values and copies data into it.
func (e *metalTestEnv) allocF32(t *testing.T, data []float32) unsafe.Pointer {
	t.Helper()
	n := len(data)
	buf, err := e.rt.Malloc(n * 4)
	if err != nil {
		t.Fatalf("Malloc(%d): %v", n*4, err)
	}
	src := unsafe.Pointer(unsafe.SliceData(data))
	if err := e.rt.Memcpy(buf, src, n*4, gpuapi.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}
	return buf
}

// allocEmptyF32 allocates a zeroed Metal buffer for n float32 values.
func (e *metalTestEnv) allocEmptyF32(t *testing.T, n int) unsafe.Pointer {
	t.Helper()
	data := make([]float32, n)
	return e.allocF32(t, data)
}

// readF32 reads n float32 values from a Metal buffer.
func (e *metalTestEnv) readF32(t *testing.T, buf unsafe.Pointer, n int) []float32 {
	t.Helper()
	out := make([]float32, n)
	dst := unsafe.Pointer(unsafe.SliceData(out))
	if err := e.rt.Memcpy(dst, buf, n*4, gpuapi.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}
	return out
}

// readI32 reads a single int32 from a Metal buffer.
func (e *metalTestEnv) readI32(t *testing.T, buf unsafe.Pointer) int32 {
	t.Helper()
	var out [1]int32
	dst := unsafe.Pointer(&out[0])
	if err := e.rt.Memcpy(dst, buf, 4, gpuapi.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}
	return out[0]
}

// allocI64 allocates a Metal buffer with int64 values.
func (e *metalTestEnv) allocI64(t *testing.T, data []int64) unsafe.Pointer {
	t.Helper()
	n := len(data)
	buf, err := e.rt.Malloc(n * 8)
	if err != nil {
		t.Fatalf("Malloc(%d): %v", n*8, err)
	}
	b := make([]byte, n*8)
	for i, v := range data {
		binary.LittleEndian.PutUint64(b[i*8:], uint64(v))
	}
	if err := e.rt.Memcpy(buf, unsafe.Pointer(unsafe.SliceData(b)), n*8, gpuapi.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}
	return buf
}

func approxEqual(a, b, tol float32) bool {
	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
		return false
	}
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= tol
}

func assertF32Slice(t *testing.T, label string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if !approxEqual(got[i], want[i], tol) {
			t.Errorf("%s[%d]: got %v, want %v (tol=%v)", label, i, got[i], want[i], tol)
		}
	}
}

// ============================================================================
// Tests
// ============================================================================

func TestMetal_Add(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{1, 2, 3, 4})
	b := env.allocF32(t, []float32{10, 20, 30, 40})
	c := env.allocEmptyF32(t, 4)

	if err := env.k.Add(a, b, c, 4, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 4)
	assertF32Slice(t, "Add", got, []float32{11, 22, 33, 44}, 1e-6)
}

func TestMetal_Sub(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{10, 20, 30, 40})
	b := env.allocF32(t, []float32{1, 2, 3, 4})
	c := env.allocEmptyF32(t, 4)

	if err := env.k.Sub(a, b, c, 4, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 4)
	assertF32Slice(t, "Sub", got, []float32{9, 18, 27, 36}, 1e-6)
}

func TestMetal_Mul(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{2, 3, 4, 5})
	b := env.allocF32(t, []float32{10, 20, 30, 40})
	c := env.allocEmptyF32(t, 4)

	if err := env.k.Mul(a, b, c, 4, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 4)
	assertF32Slice(t, "Mul", got, []float32{20, 60, 120, 200}, 1e-6)
}

func TestMetal_Div(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{10, 20, 30, 40})
	b := env.allocF32(t, []float32{2, 4, 5, 8})
	c := env.allocEmptyF32(t, 4)

	if err := env.k.Div(a, b, c, 4, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 4)
	assertF32Slice(t, "Div", got, []float32{5, 5, 6, 5}, 1e-6)
}

func TestMetal_Exp(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{0, 1, -1, 2})
	c := env.allocEmptyF32(t, 4)

	if err := env.k.Exp(a, c, 4, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 4)
	want := []float32{1.0, float32(math.E), float32(1.0 / math.E), float32(math.E * math.E)}
	assertF32Slice(t, "Exp", got, want, 1e-5)
}

func TestMetal_Log(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{1, float32(math.E), float32(math.E * math.E)})
	c := env.allocEmptyF32(t, 3)

	if err := env.k.Log(a, c, 3, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 3)
	assertF32Slice(t, "Log", got, []float32{0, 1, 2}, 1e-5)
}

func TestMetal_Sqrt(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{0, 1, 4, 9, 16})
	c := env.allocEmptyF32(t, 5)

	if err := env.k.Sqrt(a, c, 5, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 5)
	assertF32Slice(t, "Sqrt", got, []float32{0, 1, 2, 3, 4}, 1e-6)
}

func TestMetal_Sin(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{0, float32(math.Pi / 2), float32(math.Pi)})
	c := env.allocEmptyF32(t, 3)

	if err := env.k.Sin(a, c, 3, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 3)
	assertF32Slice(t, "Sin", got, []float32{0, 1, 0}, 1e-5)
}

func TestMetal_Cos(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{0, float32(math.Pi / 2), float32(math.Pi)})
	c := env.allocEmptyF32(t, 3)

	if err := env.k.Cos(a, c, 3, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 3)
	assertF32Slice(t, "Cos", got, []float32{1, 0, -1}, 1e-5)
}

func TestMetal_Tanh(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{0, 1, -1})
	c := env.allocEmptyF32(t, 3)

	if err := env.k.Tanh(a, c, 3, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 3)
	want := []float32{0, float32(math.Tanh(1)), float32(math.Tanh(-1))}
	assertF32Slice(t, "Tanh", got, want, 1e-5)
}

func TestMetal_AddScalar(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{1, 2, 3, 4})
	c := env.allocEmptyF32(t, 4)

	if err := env.k.AddScalar(a, 10, c, 4, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 4)
	assertF32Slice(t, "AddScalar", got, []float32{11, 12, 13, 14}, 1e-6)
}

func TestMetal_MulScalar(t *testing.T) {
	env := newMetalTestEnv(t)
	a := env.allocF32(t, []float32{1, 2, 3, 4})
	c := env.allocEmptyF32(t, 4)

	if err := env.k.MulScalar(a, 5, c, 4, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 4)
	assertF32Slice(t, "MulScalar", got, []float32{5, 10, 15, 20}, 1e-6)
}

func TestMetal_Fill(t *testing.T) {
	env := newMetalTestEnv(t)
	buf := env.allocEmptyF32(t, 4)

	if err := env.k.Fill(buf, 3.14, 4, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, buf, 4)
	assertF32Slice(t, "Fill", got, []float32{3.14, 3.14, 3.14, 3.14}, 1e-5)
}

func TestMetal_RMSNorm(t *testing.T) {
	env := newMetalTestEnv(t)
	// 1 row, D=4
	input := env.allocF32(t, []float32{1, 2, 3, 4})
	weight := env.allocF32(t, []float32{1, 1, 1, 1})
	output := env.allocEmptyF32(t, 4)
	scales := env.allocEmptyF32(t, 1)
	eps := float32(1e-5)

	if err := env.k.RMSNorm(input, weight, output, scales, eps, 1, 4, nil); err != nil {
		t.Fatal(err)
	}

	// CPU reference: rms = sqrt(mean([1,4,9,16]) + eps) = sqrt(7.5 + 1e-5)
	rms := float32(math.Sqrt(float64((1+4+9+16)/4.0) + float64(eps)))
	s := 1.0 / rms
	want := []float32{1 * s, 2 * s, 3 * s, 4 * s}

	got := env.readF32(t, output, 4)
	assertF32Slice(t, "RMSNorm", got, want, 1e-4)
}

func TestMetal_Softmax(t *testing.T) {
	env := newMetalTestEnv(t)
	// 1 row, 4 classes
	input := env.allocF32(t, []float32{1, 2, 3, 4})
	output := env.allocEmptyF32(t, 4)

	if err := env.k.Softmax(input, output, 1, 1, 4, nil); err != nil {
		t.Fatal(err)
	}

	// CPU reference
	in := []float64{1, 2, 3, 4}
	maxV := in[3]
	var sum float64
	for _, v := range in {
		sum += math.Exp(v - maxV)
	}
	want := make([]float32, 4)
	for i, v := range in {
		want[i] = float32(math.Exp(v-maxV) / sum)
	}

	got := env.readF32(t, output, 4)
	assertF32Slice(t, "Softmax", got, want, 1e-5)
}

func TestMetal_ScaledSoftmax(t *testing.T) {
	env := newMetalTestEnv(t)
	scale := float32(0.5)
	input := env.allocF32(t, []float32{2, 4, 6, 8})
	output := env.allocEmptyF32(t, 4)

	if err := env.k.ScaledSoftmaxF32(input, output, 1, 1, 4, scale, nil); err != nil {
		t.Fatal(err)
	}

	// CPU reference: softmax of [1,2,3,4]
	in := []float64{1, 2, 3, 4}
	maxV := in[3]
	var sum float64
	for _, v := range in {
		sum += math.Exp(v - maxV)
	}
	want := make([]float32, 4)
	for i, v := range in {
		want[i] = float32(math.Exp(v-maxV) / sum)
	}

	got := env.readF32(t, output, 4)
	assertF32Slice(t, "ScaledSoftmax", got, want, 1e-5)
}

func TestMetal_SgemvM1(t *testing.T) {
	env := newMetalTestEnv(t)
	// A = [[1,2],[3,4],[5,6]], x = [1,1] => y = [3,7,11]
	M, N := 3, 2
	A := env.allocF32(t, []float32{1, 2, 3, 4, 5, 6})
	x := env.allocF32(t, []float32{1, 1})
	y := env.allocEmptyF32(t, M)

	if err := env.k.SgemvM1(y, A, x, M, N, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, y, M)
	assertF32Slice(t, "SgemvM1", got, []float32{3, 7, 11}, 1e-5)
}

func TestMetal_Gather(t *testing.T) {
	env := newMetalTestEnv(t)
	// table: 3 rows x 2 cols = [[10,20],[30,40],[50,60]]
	// indices: [2, 0]
	// output: [[50,60],[10,20]]
	table := env.allocF32(t, []float32{10, 20, 30, 40, 50, 60})
	indices := env.allocI64(t, []int64{2, 0})
	output := env.allocEmptyF32(t, 4)

	if err := env.k.Gather(table, indices, output, 2, 2, 3, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, output, 4)
	assertF32Slice(t, "Gather", got, []float32{50, 60, 10, 20}, 1e-6)
}

func TestMetal_Transpose2D(t *testing.T) {
	env := newMetalTestEnv(t)
	// [[1,2,3],[4,5,6]] => [[1,4],[2,5],[3,6]]
	input := env.allocF32(t, []float32{1, 2, 3, 4, 5, 6})
	output := env.allocEmptyF32(t, 6)

	if err := env.k.Transpose2D(input, output, 2, 3, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, output, 6)
	assertF32Slice(t, "Transpose2D", got, []float32{1, 4, 2, 5, 3, 6}, 1e-6)
}

func TestMetal_SumAxis(t *testing.T) {
	env := newMetalTestEnv(t)
	// input: [2, 3] => sum along axis 1 => [2]
	// [[1,2,3],[4,5,6]] => [6, 15]
	input := env.allocF32(t, []float32{1, 2, 3, 4, 5, 6})
	output := env.allocEmptyF32(t, 2)

	if err := env.k.SumAxis(input, output, 2, 1, 3, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, output, 2)
	assertF32Slice(t, "SumAxis", got, []float32{6, 15}, 1e-5)
}

func TestMetal_FusedSwiGLU(t *testing.T) {
	env := newMetalTestEnv(t)
	w1 := env.allocF32(t, []float32{0, 1, -1, 2})
	w3 := env.allocF32(t, []float32{1, 1, 1, 1})
	output := env.allocEmptyF32(t, 4)

	if err := env.k.FusedSwiGLUF32(w1, w3, output, 4, nil); err != nil {
		t.Fatal(err)
	}

	// CPU ref: swiglu(x) = x * sigmoid(x) = x / (1 + exp(-x))
	vals := []float64{0, 1, -1, 2}
	want := make([]float32, 4)
	for i, x := range vals {
		want[i] = float32(x / (1.0 + math.Exp(-x)))
	}

	got := env.readF32(t, output, 4)
	assertF32Slice(t, "FusedSwiGLU", got, want, 1e-5)
}

func TestMetal_FusedAddRMSNorm(t *testing.T) {
	env := newMetalTestEnv(t)
	D := 4
	input := env.allocF32(t, []float32{1, 2, 3, 4})
	residual := env.allocF32(t, []float32{0.1, 0.2, 0.3, 0.4})
	weight := env.allocF32(t, []float32{1, 1, 1, 1})
	normedOut := env.allocEmptyF32(t, D)
	sumOut := env.allocEmptyF32(t, D)
	eps := float32(1e-5)

	if err := env.k.FusedAddRMSNormF32(input, residual, weight, normedOut, sumOut, eps, 1, D, nil); err != nil {
		t.Fatal(err)
	}

	// CPU ref: sum = input + residual, then RMSNorm
	sums := []float64{1.1, 2.2, 3.3, 4.4}
	var ss float64
	for _, v := range sums {
		ss += v * v
	}
	rms := math.Sqrt(ss/4.0 + float64(eps))
	want := make([]float32, D)
	for i, v := range sums {
		want[i] = float32(v / rms)
	}

	gotN := env.readF32(t, normedOut, D)
	assertF32Slice(t, "FusedAddRMSNorm normedOut", gotN, want, 1e-4)

	gotS := env.readF32(t, sumOut, D)
	wantS := []float32{1.1, 2.2, 3.3, 4.4}
	assertF32Slice(t, "FusedAddRMSNorm sumOut", gotS, wantS, 1e-5)
}

func TestMetal_FusedNormAdd(t *testing.T) {
	env := newMetalTestEnv(t)
	D := 4
	input := env.allocF32(t, []float32{1, 2, 3, 4})
	weight := env.allocF32(t, []float32{1, 1, 1, 1})
	residual := env.allocF32(t, []float32{0.1, 0.2, 0.3, 0.4})
	output := env.allocEmptyF32(t, D)
	eps := float32(1e-5)

	if err := env.k.FusedNormAddF32(input, weight, residual, output, eps, 1, D, nil); err != nil {
		t.Fatal(err)
	}

	// CPU ref: rmsnorm(input) + residual
	vals := []float64{1, 2, 3, 4}
	var ss float64
	for _, v := range vals {
		ss += v * v
	}
	rms := math.Sqrt(ss/4.0 + float64(eps))
	want := make([]float32, D)
	for i, v := range vals {
		want[i] = float32(v/rms) + []float32{0.1, 0.2, 0.3, 0.4}[i]
	}

	got := env.readF32(t, output, D)
	assertF32Slice(t, "FusedNormAdd", got, want, 1e-4)
}

func TestMetal_Repeat(t *testing.T) {
	env := newMetalTestEnv(t)
	// [1,2,3] repeated 2x => [1,2,3,1,2,3]
	src := env.allocF32(t, []float32{1, 2, 3})
	dst := env.allocEmptyF32(t, 6)

	if err := env.k.Repeat(src, dst, 1, 3, 1, 2, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, dst, 6)
	assertF32Slice(t, "Repeat", got, []float32{1, 2, 3, 1, 2, 3}, 1e-6)
}

func TestMetal_AddBroadcast(t *testing.T) {
	env := newMetalTestEnv(t)
	// a: [2,3] = [[1,2,3],[4,5,6]], b: [1,3] = [10,20,30] broadcast over rows
	a := env.allocF32(t, []float32{1, 2, 3, 4, 5, 6})
	b := env.allocF32(t, []float32{10, 20, 30})
	c := env.allocEmptyF32(t, 6)

	// saRow=3, saCol=1 (full stride), sbRow=0 (broadcast), sbCol=1
	if err := env.k.AddBroadcast(a, b, c, 3, 1, 0, 1, 2, 3, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, 6)
	assertF32Slice(t, "AddBroadcast", got, []float32{11, 22, 33, 14, 25, 36}, 1e-6)
}

func TestMetal_FusedRoPE(t *testing.T) {
	env := newMetalTestEnv(t)
	// batch=1, seqLen=1, headDim=4, halfRotary=2, cosStride=2
	input := env.allocF32(t, []float32{1, 2, 3, 4})
	cosA := env.allocF32(t, []float32{1, 0}) // cos(0), cos(pi/2)
	sinA := env.allocF32(t, []float32{0, 1}) // sin(0), sin(pi/2)
	output := env.allocEmptyF32(t, 4)

	if err := env.k.FusedRoPEF32(input, cosA, sinA, output, 1, 1, 4, 2, 2, nil); err != nil {
		t.Fatal(err)
	}

	// x0=1, x1=3 (at halfRotary offset): out[0] = 1*1 - 3*0 = 1, out[2] = 1*0 + 3*1 = 3
	// x0=2, x1=4: out[1] = 2*0 - 4*1 = -4, out[3] = 2*1 + 4*0 = 2
	got := env.readF32(t, output, 4)
	assertF32Slice(t, "FusedRoPE", got, []float32{1, -4, 3, 2}, 1e-5)
}

func TestMetal_LargeAdd(t *testing.T) {
	env := newMetalTestEnv(t)
	n := 10000
	aData := make([]float32, n)
	bData := make([]float32, n)
	want := make([]float32, n)
	for i := range aData {
		aData[i] = float32(i)
		bData[i] = float32(i * 2)
		want[i] = float32(i * 3)
	}

	a := env.allocF32(t, aData)
	b := env.allocF32(t, bData)
	c := env.allocEmptyF32(t, n)

	if err := env.k.Add(a, b, c, n, nil); err != nil {
		t.Fatal(err)
	}
	got := env.readF32(t, c, n)
	assertF32Slice(t, "LargeAdd", got, want, 1e-3)
}

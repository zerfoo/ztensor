package stablehlo

import (
	"fmt"
	"slices"
)

// InferStructuralShape computes the output shape for structural operations:
// MatMul, Transpose, Reshape, Concat, Slice, Gather, ReduceSum, ReduceMax, ReduceMean.
//
// attrs supports the following keys depending on the operation:
//
//   - "perm" ([]int): axis permutation for Transpose
//   - "shape" ([]int): target shape for Reshape
//   - "axis" (int): concatenation axis for Concat, reduction axis for Reduce*
//   - "start" ([]int): start indices for Slice
//   - "end" ([]int): end indices for Slice
//   - "sliceSizes" ([]int): slice sizes for Gather
//   - "keepDims" (bool): whether to keep the reduced dimension for Reduce*
func InferStructuralShape(opName string, inputShapes [][]int, attrs map[string]any) ([]int, error) {
	switch opName {
	case "MatMul":
		return inferMatMul(inputShapes)
	case "Transpose":
		return inferTranspose(inputShapes, attrs)
	case "Reshape":
		return inferReshape(inputShapes, attrs)
	case "Concat":
		return inferConcat(inputShapes, attrs)
	case "Slice":
		return inferSlice(inputShapes, attrs)
	case "Gather":
		return inferGather(inputShapes, attrs)
	case "ReduceSum", "ReduceMax", "ReduceMean":
		return inferReduce(opName, inputShapes, attrs)
	default:
		return nil, fmt.Errorf("stablehlo.InferStructuralShape: unsupported op %q", opName)
	}
}

// inferMatMul computes the output shape for matrix multiplication (dot_general).
//
// 2D: [M,K] @ [K,N] -> [M,N]
// 3D batched: [B,M,K] @ [B,K,N] -> [B,M,N]
// Higher-rank batched: [...,M,K] @ [...,K,N] -> [...,M,N]
func inferMatMul(inputShapes [][]int) ([]int, error) {
	if len(inputShapes) != 2 {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(MatMul): expected 2 input shapes, got %d", len(inputShapes))
	}
	a, b := inputShapes[0], inputShapes[1]
	if len(a) < 2 || len(b) < 2 {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(MatMul): inputs must be at least rank 2, got rank %d and %d", len(a), len(b))
	}

	// Contraction dimension: last of a must match second-to-last of b.
	m, k := a[len(a)-2], a[len(a)-1]
	k2, n := b[len(b)-2], b[len(b)-1]
	if k != k2 {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(MatMul): contraction dimension mismatch: %v (K=%d) and %v (K=%d)", a, k, b, k2)
	}

	// Batch dimensions: everything except the last two dims.
	batchA := a[:len(a)-2]
	batchB := b[:len(b)-2]
	if len(batchA) != len(batchB) {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(MatMul): batch rank mismatch: %v and %v", a, b)
	}
	for i := range batchA {
		if batchA[i] != batchB[i] {
			return nil, fmt.Errorf("stablehlo.InferStructuralShape(MatMul): batch dimension %d mismatch: %d vs %d", i, batchA[i], batchB[i])
		}
	}

	out := make([]int, 0, len(batchA)+2)
	out = append(out, batchA...)
	out = append(out, m, n)
	return out, nil
}

// inferTranspose computes the output shape for axis permutation.
// Requires attrs["perm"] ([]int) specifying the permutation.
// Example: [2,3,4] with perm [2,0,1] -> [4,2,3].
func inferTranspose(inputShapes [][]int, attrs map[string]any) ([]int, error) {
	if len(inputShapes) != 1 {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Transpose): expected 1 input shape, got %d", len(inputShapes))
	}
	shape := inputShapes[0]

	permRaw, ok := attrs["perm"]
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Transpose): missing required attr \"perm\"")
	}
	perm, ok := permRaw.([]int)
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Transpose): attr \"perm\" must be []int, got %T", permRaw)
	}
	if len(perm) != len(shape) {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Transpose): perm length %d does not match rank %d", len(perm), len(shape))
	}

	// Validate permutation: must be a valid permutation of [0..rank-1].
	seen := make([]bool, len(perm))
	for i, p := range perm {
		if p < 0 || p >= len(shape) {
			return nil, fmt.Errorf("stablehlo.InferStructuralShape(Transpose): perm[%d]=%d out of range [0,%d)", i, p, len(shape))
		}
		if seen[p] {
			return nil, fmt.Errorf("stablehlo.InferStructuralShape(Transpose): duplicate axis %d in perm", p)
		}
		seen[p] = true
	}

	out := make([]int, len(shape))
	for i, p := range perm {
		out[i] = shape[p]
	}
	return out, nil
}

// inferReshape computes the output shape for a reshape operation.
// Requires attrs["shape"] ([]int) specifying the target shape.
// Validates that the total element count matches.
func inferReshape(inputShapes [][]int, attrs map[string]any) ([]int, error) {
	if len(inputShapes) != 1 {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Reshape): expected 1 input shape, got %d", len(inputShapes))
	}

	targetRaw, ok := attrs["shape"]
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Reshape): missing required attr \"shape\"")
	}
	target, ok := targetRaw.([]int)
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Reshape): attr \"shape\" must be []int, got %T", targetRaw)
	}

	srcElems := numElements(inputShapes[0])
	dstElems := numElements(target)
	if srcElems != dstElems {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Reshape): element count mismatch: input has %d elements, target shape %v has %d", srcElems, target, dstElems)
	}
	return slices.Clone(target), nil
}

// inferConcat computes the output shape for axis concatenation.
// Requires attrs["axis"] (int) specifying the concatenation axis.
// All input shapes must match on all dimensions except the concat axis.
func inferConcat(inputShapes [][]int, attrs map[string]any) ([]int, error) {
	if len(inputShapes) < 2 {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Concat): expected at least 2 input shapes, got %d", len(inputShapes))
	}

	axisRaw, ok := attrs["axis"]
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Concat): missing required attr \"axis\"")
	}
	axis, ok := axisRaw.(int)
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Concat): attr \"axis\" must be int, got %T", axisRaw)
	}

	rank := len(inputShapes[0])
	if rank == 0 {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Concat): cannot concatenate scalar tensors")
	}
	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Concat): axis %d out of range [0,%d)", axis, rank)
	}

	// Validate all inputs have the same rank and match on non-concat dims.
	concatSize := 0
	for i, s := range inputShapes {
		if len(s) != rank {
			return nil, fmt.Errorf("stablehlo.InferStructuralShape(Concat): input %d has rank %d, expected %d", i, len(s), rank)
		}
		for d := range rank {
			if d == axis {
				continue
			}
			if s[d] != inputShapes[0][d] {
				return nil, fmt.Errorf("stablehlo.InferStructuralShape(Concat): input %d dim %d is %d, expected %d", i, d, s[d], inputShapes[0][d])
			}
		}
		concatSize += s[axis]
	}

	out := slices.Clone(inputShapes[0])
	out[axis] = concatSize
	return out, nil
}

// inferSlice computes the output shape for a slice operation.
// Requires attrs["start"] ([]int) and attrs["end"] ([]int).
// Output shape[i] = end[i] - start[i].
func inferSlice(inputShapes [][]int, attrs map[string]any) ([]int, error) {
	if len(inputShapes) != 1 {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Slice): expected 1 input shape, got %d", len(inputShapes))
	}
	shape := inputShapes[0]

	startRaw, ok := attrs["start"]
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Slice): missing required attr \"start\"")
	}
	start, ok := startRaw.([]int)
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Slice): attr \"start\" must be []int, got %T", startRaw)
	}

	endRaw, ok := attrs["end"]
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Slice): missing required attr \"end\"")
	}
	end, ok := endRaw.([]int)
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Slice): attr \"end\" must be []int, got %T", endRaw)
	}

	if len(start) != len(shape) || len(end) != len(shape) {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Slice): start/end length (%d, %d) must match rank %d", len(start), len(end), len(shape))
	}

	out := make([]int, len(shape))
	for i := range shape {
		if start[i] < 0 || end[i] > shape[i] || start[i] > end[i] {
			return nil, fmt.Errorf("stablehlo.InferStructuralShape(Slice): invalid range [%d:%d] for dimension %d (size %d)", start[i], end[i], i, shape[i])
		}
		out[i] = end[i] - start[i]
	}
	return out, nil
}

// inferGather computes the output shape for a gather operation.
// Requires attrs["sliceSizes"] ([]int) specifying the size of each slice dimension.
// The output shape is determined by the indices shape (first input) and the slice sizes.
//
// For a simple gather: output shape = indices_shape + sliceSizes (with collapsed dims removed).
// We use a simplified model: output = indices_shape[:-1] + sliceSizes.
func inferGather(inputShapes [][]int, attrs map[string]any) ([]int, error) {
	if len(inputShapes) != 2 {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Gather): expected 2 input shapes (operand, indices), got %d", len(inputShapes))
	}
	indices := inputShapes[1]

	sliceSizesRaw, ok := attrs["sliceSizes"]
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Gather): missing required attr \"sliceSizes\"")
	}
	sliceSizes, ok := sliceSizesRaw.([]int)
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Gather): attr \"sliceSizes\" must be []int, got %T", sliceSizesRaw)
	}

	// Output shape: batch dims from indices (all but last) + slice sizes.
	// The last dim of indices is the index vector dimension.
	if len(indices) == 0 {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(Gather): indices must be at least rank 1")
	}

	out := make([]int, 0, len(indices)-1+len(sliceSizes))
	out = append(out, indices[:len(indices)-1]...)
	out = append(out, sliceSizes...)
	return out, nil
}

// inferReduce computes the output shape for a reduction operation.
// Requires attrs["axis"] (int) specifying the reduction axis.
// Optional attrs["keepDims"] (bool) to retain the reduced dimension as size 1.
func inferReduce(opName string, inputShapes [][]int, attrs map[string]any) ([]int, error) {
	if len(inputShapes) != 1 {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(%s): expected 1 input shape, got %d", opName, len(inputShapes))
	}
	shape := inputShapes[0]

	axisRaw, ok := attrs["axis"]
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(%s): missing required attr \"axis\"", opName)
	}
	axis, ok := axisRaw.(int)
	if !ok {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(%s): attr \"axis\" must be int, got %T", opName, axisRaw)
	}
	if axis < 0 || axis >= len(shape) {
		return nil, fmt.Errorf("stablehlo.InferStructuralShape(%s): axis %d out of range [0,%d)", opName, axis, len(shape))
	}

	keepDims := false
	if kd, ok := attrs["keepDims"]; ok {
		if b, ok := kd.(bool); ok {
			keepDims = b
		}
	}

	if keepDims {
		out := slices.Clone(shape)
		out[axis] = 1
		return out, nil
	}

	// Remove the reduced dimension.
	out := make([]int, 0, len(shape)-1)
	for i, d := range shape {
		if i != axis {
			out = append(out, d)
		}
	}
	return out, nil
}

// numElements returns the total number of elements in a tensor with the given shape.
// Returns 1 for a scalar (empty shape).
func numElements(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

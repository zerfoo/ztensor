package stablehlo

import (
	"fmt"
	"strings"
)

// EmitMatMul emits a stablehlo.dot_general operation for matrix multiplication.
// Handles 2D (MxK @ KxN) and batched (BxMxK @ BxKxN) cases.
// Returns the MLIR line and the SSA name assigned to the result.
func EmitMatMul(namer *SSANamer, lhs, rhs string, lhsShape, rhsShape []int, dtype string) (string, string, error) {
	if len(lhsShape) < 2 || len(rhsShape) < 2 {
		return "", "", fmt.Errorf("stablehlo.EmitMatMul: inputs must be at least rank 2, got rank %d and %d", len(lhsShape), len(rhsShape))
	}
	if len(lhsShape) != len(rhsShape) {
		return "", "", fmt.Errorf("stablehlo.EmitMatMul: rank mismatch: %d vs %d", len(lhsShape), len(rhsShape))
	}

	rank := len(lhsShape)
	// Contraction dimension: last axis of LHS, second-to-last axis of RHS.
	lhsContract := rank - 1
	rhsContract := rank - 2

	if lhsShape[lhsContract] != rhsShape[rhsContract] {
		return "", "", fmt.Errorf("stablehlo.EmitMatMul: contraction dimension mismatch: %d vs %d", lhsShape[lhsContract], rhsShape[rhsContract])
	}

	outShape, err := InferStructuralShape("MatMul", [][]int{lhsShape, rhsShape}, nil)
	if err != nil {
		return "", "", err
	}

	result := namer.NextName()
	lhsType := FormatTensorType(lhsShape, dtype)
	rhsType := FormatTensorType(rhsShape, dtype)
	outType := FormatTensorType(outShape, dtype)

	// Build batch dimensions list.
	var batchDims []string
	for i := 0; i < rank-2; i++ {
		batchDims = append(batchDims, fmt.Sprintf("%d", i))
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s = %s %s, %s, batching_dims = [%s] x [%s], contracting_dims = [%d] x [%d] : (%s, %s) -> %s",
		result, OpDotGeneral, lhs, rhs,
		strings.Join(batchDims, ", "), strings.Join(batchDims, ", "),
		lhsContract, rhsContract,
		lhsType, rhsType, outType,
	)

	return b.String(), result, nil
}

// EmitTranspose emits a stablehlo.transpose operation.
// perm specifies the axis permutation (e.g., [2, 0, 1]).
func EmitTranspose(namer *SSANamer, operand string, shape []int, perm []int, dtype string) (string, string, error) {
	if len(perm) != len(shape) {
		return "", "", fmt.Errorf("stablehlo.EmitTranspose: perm length %d does not match rank %d", len(perm), len(shape))
	}

	outShape, err := InferStructuralShape("Transpose", [][]int{shape}, map[string]any{"perm": perm})
	if err != nil {
		return "", "", err
	}

	result := namer.NextName()
	inType := FormatTensorType(shape, dtype)
	outType := FormatTensorType(outShape, dtype)

	permStrs := make([]string, len(perm))
	for i, p := range perm {
		permStrs[i] = fmt.Sprintf("%d", p)
	}

	line := fmt.Sprintf("%s = %s %s, permutation = [%s] : (%s) -> %s",
		result, OpTranspose, operand,
		strings.Join(permStrs, ", "),
		inType, outType,
	)

	return line, result, nil
}

// EmitReshape emits a stablehlo.reshape operation.
// targetShape is the desired output shape.
func EmitReshape(namer *SSANamer, operand string, inShape, targetShape []int, dtype string) (string, string, error) {
	outShape, err := InferStructuralShape("Reshape", [][]int{inShape}, map[string]any{"shape": targetShape})
	if err != nil {
		return "", "", err
	}

	result := namer.NextName()
	inType := FormatTensorType(inShape, dtype)
	outType := FormatTensorType(outShape, dtype)

	line := fmt.Sprintf("%s = %s %s : (%s) -> %s",
		result, OpReshape, operand,
		inType, outType,
	)

	return line, result, nil
}

// EmitConcat emits a stablehlo.concatenate operation along the given axis.
// operands are the SSA names, shapes are the corresponding tensor shapes.
func EmitConcat(namer *SSANamer, operands []string, shapes [][]int, axis int, dtype string) (string, string, error) {
	if len(operands) != len(shapes) {
		return "", "", fmt.Errorf("stablehlo.EmitConcat: operand count %d does not match shape count %d", len(operands), len(shapes))
	}

	outShape, err := InferStructuralShape("Concat", shapes, map[string]any{"axis": axis})
	if err != nil {
		return "", "", err
	}

	result := namer.NextName()
	outType := FormatTensorType(outShape, dtype)

	line := fmt.Sprintf("%s = %s %s, dimension = %d : %s",
		result, OpConcatenate, strings.Join(operands, ", "),
		axis, outType,
	)

	return line, result, nil
}

// EmitSlice emits a stablehlo.slice operation with start, limit, and stride indices.
// strides may be nil, in which case all strides default to 1.
func EmitSlice(namer *SSANamer, operand string, shape, start, limit, strides []int, dtype string) (string, string, error) {
	if len(start) != len(shape) || len(limit) != len(shape) {
		return "", "", fmt.Errorf("stablehlo.EmitSlice: start/limit length must match rank %d", len(shape))
	}
	if strides == nil {
		strides = make([]int, len(shape))
		for i := range strides {
			strides[i] = 1
		}
	}
	if len(strides) != len(shape) {
		return "", "", fmt.Errorf("stablehlo.EmitSlice: strides length %d must match rank %d", len(strides), len(shape))
	}

	// Compute output shape: ceil((limit[i] - start[i]) / strides[i]).
	outShape := make([]int, len(shape))
	for i := range shape {
		if start[i] < 0 || limit[i] > shape[i] || start[i] > limit[i] || strides[i] <= 0 {
			return "", "", fmt.Errorf("stablehlo.EmitSlice: invalid range [%d:%d] stride %d for dimension %d (size %d)", start[i], limit[i], strides[i], i, shape[i])
		}
		outShape[i] = (limit[i] - start[i] + strides[i] - 1) / strides[i]
	}

	result := namer.NextName()
	inType := FormatTensorType(shape, dtype)
	outType := FormatTensorType(outShape, dtype)

	line := fmt.Sprintf("%s = %s %s, starts = [%s], limits = [%s], strides = [%s] : (%s) -> %s",
		result, OpSlice, operand,
		formatIntSlice(start), formatIntSlice(limit), formatIntSlice(strides),
		inType, outType,
	)

	return line, result, nil
}

// EmitGather emits a stablehlo.gather operation.
// operandShape is the shape of the data tensor, indicesShape is the shape of the index tensor.
// sliceSizes specifies the size of each gathered slice.
// offsetDims, collapsedSliceDims, startIndexMap are the gather dimension numbers.
// indexVectorDim is the dimension in the indices tensor that contains the index vector.
func EmitGather(namer *SSANamer, operand, indices string,
	operandShape, indicesShape, sliceSizes []int,
	offsetDims, collapsedSliceDims, startIndexMap []int,
	indexVectorDim int,
	dtype string,
) (string, string, error) {
	// Compute output shape from the gather semantics.
	outShape, err := InferStructuralShape("Gather", [][]int{operandShape, indicesShape}, map[string]any{"sliceSizes": sliceSizes})
	if err != nil {
		return "", "", err
	}

	result := namer.NextName()
	outType := FormatTensorType(outShape, dtype)

	var b strings.Builder
	fmt.Fprintf(&b, "%s = %s %s, %s, offset_dims = [%s], collapsed_slice_dims = [%s], start_index_map = [%s], index_vector_dim = %d, slice_sizes = [%s] : %s",
		result, OpGather, operand, indices,
		formatIntSlice(offsetDims),
		formatIntSlice(collapsedSliceDims),
		formatIntSlice(startIndexMap),
		indexVectorDim,
		formatIntSlice(sliceSizes),
		outType,
	)

	return b.String(), result, nil
}

// formatIntSlice formats an int slice as a comma-separated string (e.g., "0, 1, 2").
func formatIntSlice(s []int) string {
	parts := make([]string, len(s))
	for i, v := range s {
		parts[i] = fmt.Sprintf("%d", v)
	}
	return strings.Join(parts, ", ")
}

package core

// Matrix represents a 2D matrix of float64 values.
type Matrix [][]float64

// AddMatrices performs element-wise addition of two matrices.
// It panics if the dimensions of the matrices do not match, indicating a programming error.
func AddMatrices(a, b Matrix) Matrix {
	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		panic("Matrix dimensions do not match for addition (programming error)")
	}

	result := make(Matrix, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

// MatMul performs matrix multiplication of two matrices.
// It panics if the dimensions are incompatible for multiplication, indicating a programming error.
func MatMul(a, b Matrix) Matrix {
	if len(a[0]) != len(b) {
		panic("Matrix dimensions do not match for multiplication (programming error)")
	}

	result := make(Matrix, len(a))
	for i := range result {
		result[i] = make([]float64, len(b[0]))
		for j := range result[i] {
			for k := 0; k < len(b); k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

// Transpose returns the transpose of a matrix.
func Transpose(m Matrix) Matrix {
	result := make(Matrix, len(m[0]))
	for i := range result {
		result[i] = make([]float64, len(m))
		for j := range result[i] {
			result[i][j] = m[j][i]
		}
	}
	return result
}

// ScaleMatrix scales all elements of a matrix by a given factor.
func ScaleMatrix(m Matrix, factor float64) Matrix {
	result := make(Matrix, len(m))
	for i := range m {
		result[i] = make([]float64, len(m[i]))
		for j := range m[i] {
			result[i][j] = m[i][j] * factor
		}
	}
	return result
}

// SumRows sums the elements across rows of a matrix, returning a single-row matrix.
func SumRows(m Matrix) Matrix {
	result := make(Matrix, 1)
	result[0] = make([]float64, len(m[0]))
	for i := range m {
		for j := range m[i] {
			result[0][j] += m[i][j]
		}
	}
	return result
}

// Zeros creates a new matrix of specified dimensions filled with zeros.
func Zeros(rows, cols int) Matrix {
	result := make(Matrix, rows)
	for i := range result {
		result[i] = make([]float64, cols)
	}
	return result
}

// Ones creates a new matrix of specified dimensions filled with ones.
func Ones(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = 1.0
		}
	}
	return matrix
}

// Sum sums all elements in a float64 slice with an optional offset.
func Sum(arr []float64, offset float64) float64 {
	s := 0.0
	for _, v := range arr {
		s += v + offset
	}
	return s
}

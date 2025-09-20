package core

// 矩阵类型
type Matrix [][]float64

// 矩阵加法
func AddMatrices(a, b Matrix) Matrix {
	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		panic("Matrix dimensions do not match for addition")
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

// 矩阵乘法
func MatMul(a, b Matrix) Matrix {
	if len(a[0]) != len(b) {
		panic("Matrix dimensions do not match for multiplication")
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

// 矩阵转置
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

// 矩阵缩放
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

// SumRows 对矩阵的行求和，返回一个单行矩阵
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

// Zeros 创建一个指定维度全为零的矩阵
func Zeros(rows, cols int) Matrix {
	result := make(Matrix, rows)
	for i := range result {
		result[i] = make([]float64, cols)
	}
	return result
}

// Sum 对一个 float64 切片的所有元素求和
func Sum(arr []float64, offset float64) float64 {
	s := 0.0
	for _, v := range arr {
		s += v + offset
	}
	return s
}

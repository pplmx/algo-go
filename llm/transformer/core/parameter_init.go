package core

import (
	"math"
	"math/rand"
)

// 参数初始化器
type ParameterInitializer struct{}

// XavierUniform initializes a matrix with values drawn from a uniform distribution
// based on the Xavier initialization method.
// Note: Uses global rand.Float64(). For deterministic behavior, consider using a custom rand.Source.
func (pi *ParameterInitializer) XavierUniform(rows, cols int, gain float64) Matrix {
	limit := gain * math.Sqrt(6.0/float64(rows+cols))
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = (rand.Float64()*2 - 1) * limit
		}
	}
	return matrix
}

// XavierNormal initializes a matrix with values drawn from a normal distribution
// based on the Xavier initialization method.
// Note: Uses global rand.NormFloat64(). For deterministic behavior, consider using a custom rand.Source.
func (pi *ParameterInitializer) XavierNormal(rows, cols int, gain float64) Matrix {
	std := gain * math.Sqrt(2.0/float64(rows+cols))
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = rand.NormFloat64() * std
		}
	}
	return matrix
}

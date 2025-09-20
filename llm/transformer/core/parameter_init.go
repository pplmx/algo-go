package core

import (
	"math"
	"math/rand"
)

// 参数初始化器
type ParameterInitializer struct{}

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

func (pi *ParameterInitializer) Zeros(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
	}
	return matrix
}

func (pi *ParameterInitializer) Ones(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = 1.0
		}
	}
	return matrix
}

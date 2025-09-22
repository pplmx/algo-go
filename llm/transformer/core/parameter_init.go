package core

import (
	"math"
	"math/rand"
)

// 参数初始化器
type ParameterInitializer struct{}

// XavierUniform initializes a tensor with values drawn from a uniform distribution
// based on the Xavier initialization method.
// Note: Uses global rand.Float64(). For deterministic behavior, consider using a custom rand.Source.
func (pi *ParameterInitializer) XavierUniform(rows, cols int, gain float64) *Tensor {
	limit := gain * math.Sqrt(6.0/float64(rows+cols))
	tensor := NewTensor(rows, cols)
	for i := 0; i < tensor.Size(); i++ {
		tensor.data[i] = (rand.Float64()*2 - 1) * limit
	}
	return tensor
}

// XavierNormal initializes a tensor with values drawn from a normal distribution
// based on the Xavier initialization method.
// Note: Uses global rand.NormFloat64(). For deterministic behavior, consider using a custom rand.Source.
func (pi *ParameterInitializer) XavierNormal(rows, cols int, gain float64) *Tensor {
	std := gain * math.Sqrt(2.0/float64(rows+cols))
	tensor := NewTensor(rows, cols)
	for i := 0; i < tensor.Size(); i++ {
		tensor.data[i] = rand.NormFloat64() * std
	}
	return tensor
}

package core

import "math"

// 层归一化模块
type LayerNorm struct {
	Gamma Matrix
	Beta  Matrix
	Eps   float64
}

func NewLayerNorm(dModel int, eps float64) *LayerNorm {
	pi := &ParameterInitializer{}
	return &LayerNorm{
		Gamma: pi.Ones(1, dModel),
		Beta:  pi.Zeros(1, dModel),
		Eps:   eps,
	}
}

func (l *LayerNorm) Forward(x Matrix) Matrix {
	result := make(Matrix, len(x))

	for i := range x {
		result[i] = make([]float64, len(x[i]))

		// 计算均值和方差
		mean := 0.0
		for j := range x[i] {
			mean += x[i][j]
		}
		mean /= float64(len(x[i]))

		variance := 0.0
		for j := range x[i] {
			variance += math.Pow(x[i][j]-mean, 2)
		}
		variance /= float64(len(x[i]))
		std := math.Sqrt(variance + l.Eps)

		// 归一化
		for j := range x[i] {
			result[i][j] = (x[i][j] - mean) / std
		}

		// 缩放和平移
		for j := range result[i] {
			result[i][j] = result[i][j]*l.Gamma[0][j] + l.Beta[0][j]
		}
	}

	return result
}
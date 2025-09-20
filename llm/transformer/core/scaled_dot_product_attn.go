package core

import (
	"math"
)

// 缩放点积注意力
type ScaledDotProductAttention struct {
	DK      int
	Dropout *Dropout
}

func NewScaledDotProductAttention(dK int, dropoutRate float64) *ScaledDotProductAttention {
	return &ScaledDotProductAttention{
		DK:      dK,
		Dropout: NewDropout(dropoutRate, true),
	}
}

func (a *ScaledDotProductAttention) SetTraining(training bool) {
	a.Dropout.SetTraining(training)
}

func (a *ScaledDotProductAttention) Forward(query, key, value Matrix, mask Matrix) (Matrix, Matrix) {
	// 计算注意力分数: Q * K^T
	scores := MatMul(query, Transpose(key))

	// 缩放分数
	scores = ScaleMatrix(scores, 1.0/math.Sqrt(float64(a.DK)))

	// 应用掩码（如果有）
	if mask != nil {
		scores = AddMatrices(scores, mask)
	}

	// 应用 softmax
	weights := a.softmax(scores)

	// 应用 dropout
	weights = a.Dropout.Forward(weights)

	// 计算加权值
	output := MatMul(weights, value)

	return output, weights
}

func (a *ScaledDotProductAttention) softmax(x Matrix) Matrix {
	result := make(Matrix, len(x))
	for i := range x {
		result[i] = make([]float64, len(x[i]))

		// 减去最大值以提高数值稳定性
		maxVal := x[i][0]
		for j := 1; j < len(x[i]); j++ {
			if x[i][j] > maxVal {
				maxVal = x[i][j]
			}
		}

		// 计算指数和
		sum := 0.0
		for j := range x[i] {
			result[i][j] = math.Exp(x[i][j] - maxVal)
			sum += result[i][j]
		}

		// 应用 softmax
		for j := range result[i] {
			result[i][j] /= sum
		}
	}
	return result
}

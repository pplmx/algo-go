package core

import (
	"math"
)

// 缩放点积注意力
type ScaledDotProductAttention struct {
	DK      int
	Dropout *Dropout

	lastQuery Matrix
	lastKey   Matrix
	lastValue Matrix
	lastScores Matrix
	lastWeights Matrix
}

// NewScaledDotProductAttention creates a new ScaledDotProductAttention module.
func NewScaledDotProductAttention(dK int, dropoutRate float64) *ScaledDotProductAttention {
	return &ScaledDotProductAttention{
		DK:      dK,
		Dropout: NewDropout(dropoutRate, true),
	}
}

// SetTraining sets the training mode for the ScaledDotProductAttention module.
func (a *ScaledDotProductAttention) SetTraining(training bool) {
	a.Dropout.SetTraining(training)
}

// Forward performs the forward pass for the ScaledDotProductAttention module.
func (a *ScaledDotProductAttention) Forward(query, key, value Matrix, mask Matrix) (Matrix, Matrix) {
	a.lastQuery = query
	a.lastKey = key
	a.lastValue = value

	// 计算注意力分数: Q * K^T
	scores := MatMul(query, Transpose(key))

	// 缩放分数
	scores = ScaleMatrix(scores, 1.0/math.Sqrt(float64(a.DK)))

	// 应用掩码（如果有）
	if mask != nil {
		scores = AddMatrices(scores, mask)
	}
	a.lastScores = scores

	// 应用 softmax
	weights := a.softmax(scores)

	// 应用 dropout
	weights = a.Dropout.Forward(weights)
	a.lastWeights = weights

	// 计算加权值
	output := MatMul(weights, value)

	return output, weights
}

// Backward performs the backward pass for the ScaledDotProductAttention module.
func (a *ScaledDotProductAttention) Backward(gradOutput Matrix) (gradQuery, gradKey, gradValue Matrix) {
	// gradOutput is dL/d(output)
	var gradScores Matrix

	// 1. Gradient through MatMul(weights, value)
	// dL/d(weights) = dL/d(output) * Transpose(value)
	gradWeights := MatMul(gradOutput, Transpose(a.lastValue))
	// dL/d(value) = Transpose(weights) * dL/d(output)
	gradValue = MatMul(Transpose(a.lastWeights), gradOutput)

	// 2. Gradient through Dropout
	gradWeights = a.Dropout.Backward(gradWeights)

	// 3. Gradient through Softmax
	gradScores = make(Matrix, len(gradWeights))
	for i := range gradWeights {
		gradScores[i] = make([]float64, len(gradWeights[i]))
		// Calculate sum(dL/dy_j * y_j) for the current row
		sumGradYTimesY := 0.0
		for j := range gradWeights[i] {
			sumGradYTimesY += gradWeights[i][j] * a.lastWeights[i][j]
		}
		// dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
		for j := range gradWeights[i] {
			gradScores[i][j] = a.lastWeights[i][j] * (gradWeights[i][j] - sumGradYTimesY)
		}
	}

	// 4. Gradient through AddMatrices (mask)
	// If mask was applied, gradScores is also gradMask. We only care about gradScores.
	// No change to gradScores from AddMatrices.

	// 5. Gradient through ScaleMatrix (1.0 / sqrt(DK))
	gradScores = ScaleMatrix(gradScores, 1.0/math.Sqrt(float64(a.DK)))

	// 6. Gradient through MatMul(query, Transpose(key))
	// dL/d(query) = MatMul(dL/d(scores), key)
	gradQuery = MatMul(gradScores, a.lastKey)
	// dL/d(key) = MatMul(Transpose(query), dL/d(scores))
	gradKey = MatMul(Transpose(a.lastQuery), gradScores)

	return gradQuery, gradKey, gradValue
}

// softmax applies the softmax function row-wise to a matrix.
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

package core

import (
	"math"
)

// 缩放点积注意力
type ScaledDotProductAttention struct {
	DK      int
	Dropout *Dropout

	lastQuery   *Tensor
	lastKey     *Tensor
	lastValue   *Tensor
	lastScores  *Tensor
	lastWeights *Tensor
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
func (a *ScaledDotProductAttention) Forward(query, key, value *Tensor, mask *Tensor) (*Tensor, *Tensor) {
	a.lastQuery = query
	a.lastKey = key
	a.lastValue = value

	// 计算注意力分数: Q * K^T
	var transposedKey *Tensor
	if key.Ndim() == 3 {
		transposedKey = key.Transpose(0, 2, 1)
	} else {
		transposedKey = key.Transpose()
	}
	scores := query.Dot(transposedKey)

	// 缩放分数
	scores = scores.MulScalar(1.0 / math.Sqrt(float64(a.DK)))

	// 应用掩码（如果有）
	if mask != nil {
		scores = scores.Add(mask)
	}
	a.lastScores = scores

	// 应用 softmax
	weights := scores.Softmax(len(scores.Shape()) - 1)

	// 应用 dropout
	weights = a.Dropout.Forward(weights)
	a.lastWeights = weights

	// 计算加权值
	output := weights.Dot(value)

	return output, weights
}

// Backward performs the backward pass for the ScaledDotProductAttention module.
func (a *ScaledDotProductAttention) Backward(gradOutput *Tensor) (gradQuery, gradKey, gradValue *Tensor) {
	// gradOutput is dL/d(output)

	// 1. Gradient through MatMul(weights, value)
	gradWeights := gradOutput.Dot(a.lastValue.Transpose(0, 2, 1))
	gradValue = a.lastWeights.Transpose(0, 2, 1).Dot(gradOutput)

	// 2. Gradient through Dropout
	gradWeights = a.Dropout.Backward(gradWeights)

	// 3. Gradient through Softmax
	// dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
	// This is a simplification. A full softmax gradient is a bit more involved.
	sumGradYTimesY := gradWeights.Mul(a.lastWeights).Sum(len(gradWeights.Shape()) - 1)
	gradScores := a.lastWeights.Mul(gradWeights.Sub(sumGradYTimesY.ExpandDims(len(gradWeights.Shape()) - 1)))

	// 4. Gradient through AddMatrices (mask) - no change to gradScores

	// 5. Gradient through ScaleMatrix (1.0 / sqrt(DK))
	gradScores = gradScores.MulScalar(1.0 / math.Sqrt(float64(a.DK)))

	// 6. Gradient through MatMul(query, Transpose(key))
	gradQuery = gradScores.Dot(a.lastKey)
	gradKey = gradScores.Transpose(0, 2, 1).Dot(a.lastQuery)

	return gradQuery, gradKey, gradValue
}

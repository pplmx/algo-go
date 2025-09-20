package core

import (
	"math"
)

type LayerNorm struct {
	Gamma Matrix
	Beta  Matrix
	Eps   float64

	lastInput           Matrix
	lastMean            []float64
	lastVariance        []float64
	lastNormalizedInput Matrix

	gradGamma Matrix
	gradBeta  Matrix
}

// NewLayerNorm creates a new LayerNorm layer.
func NewLayerNorm(dModel int, eps float64) *LayerNorm {
	return &LayerNorm{
		Gamma: Ones(1, dModel),
		Beta:  Zeros(1, dModel),
		Eps:   eps,
	}
}

// Forward performs the forward pass for the LayerNorm layer.
func (l *LayerNorm) Forward(x Matrix) Matrix {
	l.lastInput = x
	result := make(Matrix, len(x))
	l.lastMean = make([]float64, len(x))
	l.lastVariance = make([]float64, len(x))
	l.lastNormalizedInput = make(Matrix, len(x))

	for i := range x {
		result[i] = make([]float64, len(x[i]))
		l.lastNormalizedInput[i] = make([]float64, len(x[i]))

		// 计算均值和方差
		mean := 0.0
		for j := range x[i] {
			mean += x[i][j]
		}
		mean /= float64(len(x[i]))
		l.lastMean[i] = mean

		variance := 0.0
		for j := range x[i] {
			variance += math.Pow(x[i][j]-mean, 2)
		}
		variance /= float64(len(x[i]))
		l.lastVariance[i] = variance
		std := math.Sqrt(variance + l.Eps)

		// 归一化
		for j := range x[i] {
			l.lastNormalizedInput[i][j] = (x[i][j] - mean) / std
			result[i][j] = l.lastNormalizedInput[i][j]
		}

		// 缩放和平移
		for j := range result[i] {
			result[i][j] = result[i][j]*l.Gamma[0][j] + l.Beta[0][j]
		}
	}

	return result
}

// Backward performs the backward pass for the LayerNorm layer.
func (l *LayerNorm) Backward(gradOutput Matrix) Matrix {
	gradInput := make(Matrix, len(gradOutput))
	l.gradGamma = Zeros(1, len(l.Gamma[0]))
	l.gradBeta = Zeros(1, len(l.Beta[0]))

	for i := range gradOutput {
		gradInput[i] = make([]float64, len(gradOutput[i]))

		mean := l.lastMean[i]
		variance := l.lastVariance[i]
		std := math.Sqrt(variance + l.Eps)
		N := float64(len(l.lastInput[i]))

		// Calculate gradients for gamma and beta
		for j := range gradOutput[i] {
			l.gradGamma[0][j] += gradOutput[i][j] * l.lastNormalizedInput[i][j]
			l.gradBeta[0][j] += gradOutput[i][j]
		}

		// Calculate gradient for input
		dNormalized := make([]float64, len(gradOutput[i]))
		for j := range gradOutput[i] {
			dNormalized[j] = gradOutput[i][j] * l.Gamma[0][j]
		}

		dVariance := 0.0
		for j := range gradOutput[i] {
			dVariance += dNormalized[j] * (l.lastInput[i][j] - mean) * -0.5 * math.Pow(variance+l.Eps, -1.5)
		}

		dMean := 0.0
		for j := range gradOutput[i] {
			dMean += dNormalized[j] * (-1.0 / std)
		}
		dMean += dVariance * (-2.0 / N) * Sum(l.lastInput[i], -mean)

		for j := range gradOutput[i] {
			gradInput[i][j] = dNormalized[j]*(1.0/std) + dVariance*(2.0*(l.lastInput[i][j]-mean)/N) + dMean/N
		}
	}
	return gradInput
}

// GetParameters returns the trainable parameters (Gamma and Beta) of the LayerNorm layer.
func (l *LayerNorm) GetParameters() []Matrix {
	return []Matrix{l.Gamma, l.Beta}
}

// GetGradients returns the gradients of the trainable parameters (gradGamma and gradBeta).
func (l *LayerNorm) GetGradients() []Matrix {
	return []Matrix{l.gradGamma, l.gradBeta}
}

// ZeroGradients sets the gradients of the trainable parameters to zero.
func (l *LayerNorm) ZeroGradients() {
	if l.gradGamma != nil {
		l.gradGamma = Zeros(len(l.gradGamma), len(l.gradGamma[0]))
	}
	if l.gradBeta != nil {
		l.gradBeta = Zeros(len(l.gradBeta), len(l.gradBeta[0]))
	}
}

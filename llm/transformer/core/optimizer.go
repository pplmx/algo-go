package core

import (
	"math"
)

// AdamOptimizer implements the Adam optimization algorithm.
type AdamOptimizer struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	t            int
	m            *Tensor // First moment estimates
	v            *Tensor // Second moment estimates
}

// NewAdamOptimizer creates a new AdamOptimizer.
func NewAdamOptimizer(learningRate, beta1, beta2, epsilon float64) *AdamOptimizer {
	return &AdamOptimizer{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      epsilon,
		t:            0,
	}
}

// Update updates the parameters using the Adam optimization algorithm.
func (a *AdamOptimizer) Update(param, grad *Tensor) {
	if a.m == nil {
		a.m = Zeros(param.Shape()...)
		a.v = Zeros(param.Shape()...)
	}

	a.t++

	// Update biased first moment estimate
	a.m = a.m.MulScalar(a.Beta1).Add(grad.MulScalar(1 - a.Beta1))

	// Update biased second raw moment estimate
	a.v = a.v.MulScalar(a.Beta2).Add(grad.Power(2).MulScalar(1 - a.Beta2))

	// Compute bias-corrected first moment estimate
	mHat := a.m.DivScalar(1 - math.Pow(a.Beta1, float64(a.t)))

	// Compute bias-corrected second raw moment estimate
	vHat := a.v.DivScalar(1 - math.Pow(a.Beta2, float64(a.t)))

	// Update parameters in place
	for i := 0; i < param.Size(); i++ {
		param.data[i] -= a.LearningRate * mHat.data[i] / (math.Sqrt(vHat.data[i]) + a.Epsilon)
	}
}

func (a *AdamOptimizer) GetParameters() []*Tensor {
	// Optimizers don't have parameters in the same way layers do.
	return []*Tensor{}
}

func (a *AdamOptimizer) GetGradients() []*Tensor {
	// Optimizers don't have gradients in the same way layers do.
	return []*Tensor{}
}

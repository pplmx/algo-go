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
	m            map[*Tensor]*Tensor // First moment estimates, per parameter
	v            map[*Tensor]*Tensor // Second moment estimates, per parameter
}

// NewAdamOptimizer creates a new AdamOptimizer.
func NewAdamOptimizer(learningRate, beta1, beta2, epsilon float64) *AdamOptimizer {
	return &AdamOptimizer{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      epsilon,
		t:            0,
		m:            make(map[*Tensor]*Tensor),
		v:            make(map[*Tensor]*Tensor),
	}
}

// Update updates the parameters using the Adam optimization algorithm.
func (a *AdamOptimizer) Update(param, grad *Tensor) {
	// Initialize moment estimates for this parameter if not already present
	if _, ok := a.m[param]; !ok {
		a.m[param] = Zeros(param.Shape()...)
		a.v[param] = Zeros(param.Shape()...)
	}

	a.t++

	// Update biased first moment estimate
	a.m[param] = a.m[param].MulScalar(a.Beta1).Add(grad.MulScalar(1 - a.Beta1))

	// Update biased second raw moment estimate
	a.v[param] = a.v[param].MulScalar(a.Beta2).Add(grad.Power(2).MulScalar(1 - a.Beta2))

	// Compute bias-corrected first moment estimate
	mHat := a.m[param].DivScalar(1 - math.Pow(a.Beta1, float64(a.t)))

	// Compute bias-corrected second raw moment estimate
	vHat := a.v[param].DivScalar(1 - math.Pow(a.Beta2, float64(a.t)))

	// Update parameters
	update := mHat.Div(vHat.Sqrt().AddScalar(a.Epsilon))
	param.Sub_(update.MulScalar(a.LearningRate))
}

func (a *AdamOptimizer) GetParameters() []*Tensor {
	// Optimizers don't have parameters in the same way layers do.
	return []*Tensor{}
}

func (a *AdamOptimizer) GetGradients() []*Tensor {
	// Optimizers don't have gradients in the same way layers do.
	return []*Tensor{}
}

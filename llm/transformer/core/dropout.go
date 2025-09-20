package core

import (
	"math/rand"
	"time"
)

// Dropout 模块
type Dropout struct {
	Rate     float64
	Training bool
	rng      *rand.Rand
	lastMask Matrix // Stores the mask applied during the forward pass
}

// NewDropout creates a new Dropout layer.
// The random number generator is seeded with the current time, which might lead to non-deterministic
// behavior if multiple Dropout layers are created in rapid succession. For deterministic behavior
// in tests, consider passing a fixed rand.Source.
func NewDropout(rate float64, training bool) *Dropout {
	return &Dropout{
		Rate:     rate,
		Training: training,
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// SetTraining sets the training mode for the Dropout layer.
func (d *Dropout) SetTraining(training bool) {
	d.Training = training
}

// Forward performs the forward pass for the Dropout layer.
// During training, it randomly sets elements to zero and scales the remaining ones.
// During evaluation, it simply passes the input through.
func (d *Dropout) Forward(x Matrix) Matrix {
	if !d.Training || d.Rate == 0.0 {
		return x
	}

	result := make(Matrix, len(x))
	d.lastMask = make(Matrix, len(x)) // Initialize lastMask
	for i := range x {
		result[i] = make([]float64, len(x[i]))
		d.lastMask[i] = make([]float64, len(x[i])) // Initialize inner slice of lastMask
		for j := range x[i] {
			if d.rng.Float64() > d.Rate {
				result[i][j] = x[i][j] / (1.0 - d.Rate)
				d.lastMask[i][j] = 1.0 / (1.0 - d.Rate) // Store scaling factor
			} else {
				result[i][j] = 0.0
				d.lastMask[i][j] = 0.0 // Store zero
			}
		}
	}
	return result
}

// Backward performs the backward pass for the Dropout layer.
// It applies the same mask used in the forward pass to the gradients.
func (d *Dropout) Backward(gradOutput Matrix) Matrix {
	if !d.Training || d.Rate == 0.0 {
		return gradOutput
	}

	gradInput := make(Matrix, len(gradOutput))
	for i := range gradOutput {
		gradInput[i] = make([]float64, len(gradOutput[i]))
		for j := range gradOutput[i] {
			gradInput[i][j] = gradOutput[i][j] * d.lastMask[i][j]
		}
	}
	return gradInput
}

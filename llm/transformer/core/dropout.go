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
	lastMask *Tensor // Stores the mask applied during the forward pass
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
func (d *Dropout) Forward(x *Tensor) *Tensor {
	if !d.Training || d.Rate == 0.0 {
		return x
	}

	maskData := make([]float64, x.Size())
	scale := 1.0 / (1.0 - d.Rate)
	for i := 0; i < x.Size(); i++ {
		if d.rng.Float64() > d.Rate {
			maskData[i] = scale
		} else {
			maskData[i] = 0.0
		}
	}
	d.lastMask = NewTensorFromData(maskData, x.Shape()...)

	return x.Mul(d.lastMask)
}

// Backward performs the backward pass for the Dropout layer.
// It applies the same mask used in the forward pass to the gradients.
func (d *Dropout) Backward(gradOutput *Tensor) *Tensor {
	if !d.Training || d.Rate == 0.0 {
		return gradOutput
	}

	return gradOutput.Mul(d.lastMask)
}

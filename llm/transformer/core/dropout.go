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
}

func NewDropout(rate float64, training bool) *Dropout {
	return &Dropout{
		Rate:     rate,
		Training: training,
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (d *Dropout) SetTraining(training bool) {
	d.Training = training
}

func (d *Dropout) Forward(x Matrix) Matrix {
	if !d.Training || d.Rate == 0.0 {
		return x
	}

	result := make(Matrix, len(x))
	for i := range x {
		result[i] = make([]float64, len(x[i]))
		for j := range x[i] {
			if d.rng.Float64() > d.Rate {
				result[i][j] = x[i][j] / (1.0 - d.Rate)
			} else {
				result[i][j] = 0.0
			}
		}
	}
	return result
}

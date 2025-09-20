package core

import (
	"github.com/pplmx/algo-go/llm/transformer/config"
	"math"
)

// Adam 优化器
type AdamOptimizer struct {
	Config config.TransformerConfig
	Step   int
}

func NewAdamOptimizer(cfg config.TransformerConfig) *AdamOptimizer {
	return &AdamOptimizer{
		Config: cfg,
		Step:   0,
	}
}

func (a *AdamOptimizer) Update(param, grad, m, v Matrix) {
	a.Step++

	// 更新一阶矩估计
	for i := range m {
		for j := range m[i] {
			m[i][j] = a.Config.Beta1*m[i][j] + (1-a.Config.Beta1)*grad[i][j]
		}
	}

	// 更新二阶矩估计
	for i := range v {
		for j := range v[i] {
			v[i][j] = a.Config.Beta2*v[i][j] + (1-a.Config.Beta2)*grad[i][j]*grad[i][j]
		}
	}

	// 计算偏差校正
	mHat := make(Matrix, len(m))
	vHat := make(Matrix, len(v))
	for i := range m {
		mHat[i] = make([]float64, len(m[i]))
		vHat[i] = make([]float64, len(v[i]))
		for j := range m[i] {
			mHat[i][j] = m[i][j] / (1 - math.Pow(a.Config.Beta1, float64(a.Step)))
			vHat[i][j] = v[i][j] / (1 - math.Pow(a.Config.Beta2, float64(a.Step)))
		}
	}

	// 更新参数
	for i := range param {
		for j := range param[i] {
			param[i][j] -= a.Config.LearningRate * mHat[i][j] / (math.Sqrt(vHat[i][j]) + a.Config.Eps)
		}
	}
}

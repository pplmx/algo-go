package core

import (
	"github.com/pplmx/algo-go/llm/transformer/config"
	"math"
)

// Adam 优化器
type AdamOptimizer struct {
	Config config.TransformerConfig
	Step   int
	m      Matrix // First moment estimates
	v      Matrix // Second moment estimates
}

func NewAdamOptimizer(cfg config.TransformerConfig) *AdamOptimizer {
	return &AdamOptimizer{
		Config: cfg,
		Step:   0,
	}
}

func (a *AdamOptimizer) Update(param, grad Matrix) {
	a.Step++

	// Initialize m and v if they are nil (first update)
	if a.m == nil || len(a.m) != len(param) || len(a.m[0]) != len(param[0]) {
		a.m = Zeros(len(param), len(param[0]))
		a.v = Zeros(len(param), len(param[0]))
	}

	// 更新一阶矩估计
	for i := range a.m {
		for j := range a.m[i] {
			a.m[i][j] = a.Config.Beta1*a.m[i][j] + (1-a.Config.Beta1)*grad[i][j]
		}
	}

	// 更新二阶矩估计
	for i := range a.v {
		for j := range a.v[i] {
			a.v[i][j] = a.Config.Beta2*a.v[i][j] + (1-a.Config.Beta2)*grad[i][j]*grad[i][j]
		}
	}

	// 计算偏差校正
	mHat := make(Matrix, len(a.m))
	vHat := make(Matrix, len(a.v))
	for i := range a.m {
		mHat[i] = make([]float64, len(a.m[i]))
		vHat[i] = make([]float64, len(a.v[i]))
		for j := range a.m[i] {
			mHat[i][j] = a.m[i][j] / (1 - math.Pow(a.Config.Beta1, float64(a.Step)))
			vHat[i][j] = a.v[i][j] / (1 - math.Pow(a.Config.Beta2, float64(a.Step)))
		}
	}

	// 更新参数
	for i := range param {
		for j := range param[i] {
			param[i][j] -= a.Config.LearningRate * mHat[i][j] / (math.Sqrt(vHat[i][j]) + a.Config.Eps)
		}
	}
}

// AdamOptimizer does not have trainable parameters in the traditional sense,
// but it manages internal state (m, v) that needs to be zeroed or reset.
// For simplicity, we'll make it conform to Trainable interface by returning its internal state.
func (a *AdamOptimizer) GetParameters() []Matrix {
	// Return empty slice as optimizer itself doesn't have parameters to optimize
	return []Matrix{}
}

func (a *AdamOptimizer) GetGradients() []Matrix {
	// Return empty slice as optimizer itself doesn't have gradients
	return []Matrix{}
}

func (a *AdamOptimizer) ZeroGradients() {
	// Reset internal state m and v
	a.m = nil
	a.v = nil
	a.Step = 0
}

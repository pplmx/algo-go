package core

import (
	"math"
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
)

func TestAdamOptimizer_Update(t *testing.T) {
	// 1. 配置和初始化
	cfg := config.DefaultConfig()
	optimizer := core.NewAdamOptimizer(cfg)

	// 2. 准备输入数据
	param := core.Matrix{{1.0, 2.0}, {3.0, 4.0}}
	grad := core.Matrix{{0.1, 0.2}, {0.3, 0.4}}
	m := core.Matrix{{0.0, 0.0}, {0.0, 0.0}}
	v := core.Matrix{{0.0, 0.0}, {0.0, 0.0}}

	// 3. 手动计算预期结果 (after one step)
	step := 1
	beta1 := cfg.Beta1
	beta2 := cfg.Beta2
	lr := cfg.LearningRate
	eps := cfg.Eps

	expectedM := core.Matrix{{0.0, 0.0}, {0.0, 0.0}}
	expectedV := core.Matrix{{0.0, 0.0}, {0.0, 0.0}}
	for i := range grad {
		for j := range grad[i] {
			expectedM[i][j] = (1 - beta1) * grad[i][j]
			expectedV[i][j] = (1 - beta2) * grad[i][j] * grad[i][j]
		}
	}

	mHat := core.ScaleMatrix(expectedM, 1.0/(1-math.Pow(beta1, float64(step))))
	vHat := core.ScaleMatrix(expectedV, 1.0/(1-math.Pow(beta2, float64(step))))

	expectedParam := make(core.Matrix, len(param))
	for i := range param {
		expectedParam[i] = make([]float64, len(param[i]))
		for j := range param[i] {
			expectedParam[i][j] = param[i][j] - lr*mHat[i][j]/(math.Sqrt(vHat[i][j])+eps)
		}
	}

	// 4. 执行更新
	optimizer.Update(param, grad, m, v)

	// 5. 验证结果
	if !MatricesAlmostEqual(param, expectedParam, 1e-9) {
		t.Errorf("Parameter update incorrect.\nGot:  %v\nWant: %v", param, expectedParam)
	}
	if !MatricesAlmostEqual(m, expectedM, 1e-9) {
		t.Errorf("First moment vector 'm' incorrect.\nGot:  %v\nWant: %v", m, expectedM)
	}
	if !MatricesAlmostEqual(v, expectedV, 1e-9) {
		t.Errorf("Second moment vector 'v' incorrect.\nGot:  %v\nWant: %v", v, expectedV)
	}
}
package core_test

import (
	"math"
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/tests/llm/helpers"
)

func TestAdamOptimizer_Update(t *testing.T) {
	// 1. 配置和初始化
	cfg := config.NewDefaultTrainConfig()
	optimizer := core.NewAdamOptimizer(cfg.LearningRate, cfg.Beta1, cfg.Beta2, cfg.Eps)

	// 2. 准备输入数据
	param := core.NewTensorFromData([]float64{1.0, 2.0, 3.0, 4.0}, 2, 2)
	grad := core.NewTensorFromData([]float64{0.1, 0.2, 0.3, 0.4}, 2, 2)

	// 3. 手动计算预期结果 (after one step)
	step := 1
	beta1 := cfg.Beta1
	beta2 := cfg.Beta2
	lr := cfg.LearningRate
	eps := cfg.Eps

	expectedMData := make([]float64, grad.Size())
	expectedVData := make([]float64, grad.Size())
	for i, g := range grad.Data() {
		expectedMData[i] = (1 - beta1) * g
		expectedVData[i] = (1 - beta2) * g * g
	}
	expectedM := core.NewTensorFromData(expectedMData, grad.Shape()...)
	expectedV := core.NewTensorFromData(expectedVData, grad.Shape()...)

	mHat := expectedM.DivScalar(1 - math.Pow(beta1, float64(step)))
	vHat := expectedV.DivScalar(1 - math.Pow(beta2, float64(step)))

	expectedParamData := make([]float64, param.Size())
	for i, p := range param.Data() {
		expectedParamData[i] = p - lr*mHat.Data()[i]/(math.Sqrt(vHat.Data()[i])+eps)
	}
	expectedParam := core.NewTensorFromData(expectedParamData, param.Shape()...)

	// 4. 执行更新
	optimizer.Update(param, grad)

	// 5. 验证结果
	if !helpers.TensorsAlmostEqual(param, expectedParam, 1e-6) {
		t.Errorf("Parameter update incorrect.\nGot:  %v\nWant: %v", param, expectedParam)
	}
}

package layer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
	"github.com/pplmx/algo-go/tests/llm/helpers"
)

func TestLinearLayer_Forward(t *testing.T) {
	// 1. 配置和初始化
	inFeatures := 3
	outFeatures := 2
	linear := layer.NewLinearLayer(inFeatures, outFeatures, true)

	// 手动设置权重和偏置以进行确定性测试
	linear.Weight = core.NewTensorFromData([]float64{
		1, 2,
		3, 4,
		5, 6,
	}, inFeatures, outFeatures)
	linear.Bias = core.NewTensorFromData([]float64{0.5, -0.5}, 1, outFeatures)

	// 2. 准备输入数据
	input := core.NewTensorFromData([]float64{
		1, 2, 3,
		4, 5, 6,
	}, 2, inFeatures)

	// 3. 计算预期输出
	// MatMul(input, weight) + bias
	// a. MatMul
	//    row 1: {1*1+2*3+3*5, 1*2+2*4+3*6} = {22, 28}
	//    row 2: {4*1+5*3+6*5, 4*2+5*4+6*6} = {49, 64}
	// b. Add bias
	//    row 1: {22+0.5, 28-0.5} = {22.5, 27.5}
	//    row 2: {49+0.5, 64-0.5} = {49.5, 63.5}
	expectedOutput := core.NewTensorFromData([]float64{
		22.5, 27.5,
		49.5, 63.5,
	}, 2, outFeatures)

	// 4. 执行前向传播
	output := linear.Forward(input)

	// 5. 验证结果
	if !helpers.TensorsAlmostEqual(output, expectedOutput, 1e-9) {
		t.Errorf("Forward() output = %v, want %v", output, expectedOutput)
	}
}

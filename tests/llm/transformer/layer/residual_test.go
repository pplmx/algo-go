package layer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
	"github.com/pplmx/algo-go/tests/llm/helpers"
)

func TestResidualConnection_Forward(t *testing.T) {
	// 1. 配置和初始化
	dModel := 3
	dropoutRate := 0.0
	residual := layer.NewResidualConnection(dModel, dropoutRate)
	residual.SetTraining(false) // 关闭 dropout

	// 2. 准备输入数据
	x := core.NewTensorFromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	sublayerOutput := core.NewTensorFromData([]float64{0.1, 0.2, 0.3, -0.1, -0.2, -0.3}, 2, 3)

	// 3. 计算预期输出
	// a. Add
	added := x.Add(sublayerOutput)
	// b. Norm
	// Manually perform layer normalization on `added`
	ln := core.NewLayerNorm(dModel, 1e-5)
	expectedOutput := ln.Forward(added)

	// 4. 执行前向传播
	output := residual.Forward(x, sublayerOutput)

	// 5. 验证结果
	if !helpers.TensorsAlmostEqual(output, expectedOutput, 1e-9) {
		t.Errorf("Forward() output = %v, want %v", output, expectedOutput)
	}
}

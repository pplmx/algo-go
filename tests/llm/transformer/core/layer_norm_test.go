package core_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/tests/llm/helpers"
)

func TestLayerNorm_Forward(t *testing.T) {
	// 1. 初始化
	dModel := 3
	eps := 1e-5
	ln := core.NewLayerNorm(dModel, eps)

	// 2. 准备输入数据
	input := core.NewTensorFromData([]float64{1, 2, 3}, 1, 3)

	// 3. 计算预期输出
	expectedOutput := core.NewTensorFromData([]float64{-1.2247356859083902, 0, 1.2247356859083902}, 1, 3)

	// 4. 执行前向传播
	output := ln.Forward(input)

	// 5. 验证结果
	if !helpers.TensorsAlmostEqual(output, expectedOutput, 1e-9) {
		t.Errorf("Forward() output = %v, want %v", output, expectedOutput)
	}
}

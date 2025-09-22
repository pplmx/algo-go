package core_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/tests/llm/helpers"
)

func TestScaledDotProductAttention_Forward(t *testing.T) {
	// 1. 初始化
	dK := 1
	attn := core.NewScaledDotProductAttention(dK, 0.0) // No dropout for deterministic test
	attn.SetTraining(false)

	// 2. 准备输入数据
	query := core.NewTensorFromData([]float64{10, 0}, 2, 1)
	key := core.NewTensorFromData([]float64{10, 0}, 2, 1)
	value := core.NewTensorFromData([]float64{1, 2, 3, 4}, 2, 2)

	// 3. 计算预期输出
	expectedOutput := core.NewTensorFromData([]float64{1, 2, 2, 3}, 2, 2)

	// 4. 执行前向传播
	output, _ := attn.Forward(query, key, value, nil)

	// 5. 验证结果
	if !helpers.TensorsAlmostEqual(output, expectedOutput, 1e-9) {
		t.Errorf("Forward() output = %v, want %v", output, expectedOutput)
	}
}

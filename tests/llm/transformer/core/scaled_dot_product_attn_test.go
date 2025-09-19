package core_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
)

func TestScaledDotProductAttention_Forward(t *testing.T) {
	// 1. 初始化
	dK := 1
	attn := core.NewScaledDotProductAttention(dK, 0.0) // No dropout for deterministic test
	attn.SetTraining(false)

	// 2. 准备输入数据
	// Q 和 K 的设计使得注意力权重会非常极端 (接近1和0)，便于验证
	query := core.Matrix{{10}, {0}} // seq_len=2, d_k=1
	key := core.Matrix{{10}, {0}}   // seq_len=2, d_k=1
	value := core.Matrix{{1, 2}, {3, 4}} // seq_len=2, d_v=2

	// 3. 计算预期输出
	// 手动计算步骤:
	// a. scores = MatMul(Q, Transpose(K)) -> {{100, 0}, {0, 0}}
	// b. scaled_scores = scores / sqrt(dK=1) -> {{100, 0}, {0, 0}}
	// c. weights = softmax(scaled_scores) -> {{1.0, 0.0}, {0.5, 0.5}} (approx.)
	//    - 第一行: exp(100) 远大于 exp(0)，所以权重是 [1, 0]
	//    - 第二行: exp(0) 等于 exp(0)，所以权重是 [0.5, 0.5]
	// d. output = MatMul(weights, V)
	//    - output[0] = 1.0 * V[0] + 0.0 * V[1] = {1, 2}
	//    - output[1] = 0.5 * V[0] + 0.5 * V[1] = {0.5*1+0.5*3, 0.5*2+0.5*4} = {2, 3}
	expectedOutput := core.Matrix{{1, 2}, {2, 3}}

	// 4. 执行前向传播
	output, _ := attn.Forward(query, key, value, nil)

	// 5. 验证结果
	if !matricesAlmostEqual(output, expectedOutput, 1e-9) {
		t.Errorf("Forward() output = %v, want %v", output, expectedOutput)
	}
}
package core

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
)

func TestLayerNorm_Forward(t *testing.T) {
	// 1. 初始化
	dModel := 3
	eps := 1e-5
	ln := core.NewLayerNorm(dModel, eps)

	// 2. 准备输入数据
	input := core.Matrix{{1, 2, 3}}

	// 3. 计算预期输出
	// 对于初始化的 LayerNorm, gamma 是 [1,1,1], beta 是 [0,0,0]
	// a. mean = (1+2+3)/3 = 2
	// b. variance = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3 = 0.6666...
	// c. std = sqrt(variance + eps) = sqrt(0.666666... + 1e-5) = 0.8165026...
	// d. normalized_x = (x - mean) / std
	//    - (1-2)/0.8165026 = -1.2247356...
	//    - (2-2)/0.8165026 = 0
	//    - (3-2)/0.8165026 = 1.2247356...
	expectedOutput := core.Matrix{{-1.2247356859083902, 0, 1.2247356859083902}}

	// 4. 执行前向传播
	output := ln.Forward(input)

	// 5. 验证结果
	if !MatricesAlmostEqual(output, expectedOutput, 1e-9) {
		t.Errorf("Forward() output = %v, want %v", output, expectedOutput)
	}
}

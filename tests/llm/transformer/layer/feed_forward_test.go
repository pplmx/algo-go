package layer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
)

func TestFeedForwardNetwork_Forward(t *testing.T) {
	// 1. 配置
	dModel := 8
	ffnHiddenSize := 16
	dropoutRate := 0.0
	seqLen := 3

	// 2. 初始化 FFN 层
	ffn := layer.NewFeedForwardNetwork(dModel, ffnHiddenSize, dropoutRate, false)
	ffn.SetTraining(false) // 关闭 dropout

	// 3. 准备输入数据
	inputData := []float64{
		1, 2, 3, 4, 5, 6, 7, 8,
		8, 7, 6, 5, 4, 3, 2, 1,
		1, 3, 5, 7, 9, 2, 4, 6,
	}
	input := core.NewTensorFromData(inputData, seqLen, dModel)

	// 4. 执行前向传播
	output := ffn.Forward(input)

	// 5. 验证输出维度
	if output.Shape()[0] != seqLen {
		t.Errorf("Output sequence length = %d, want %d", output.Shape()[0], seqLen)
	}
	if output.Shape()[1] != dModel {
		t.Errorf("Output dimension = %d, want %d", output.Shape()[1], dModel)
	}

	// 6. 验证输出值 (sanity check)
	isAllZero := true
	for i := 0; i < output.Shape()[0]; i++ {
		for j := 0; j < output.Shape()[1]; j++ {
			if output.Get(i, j) != 0 {
				isAllZero = false
				break
			}
		}
		if !isAllZero {
			break
		}
	}
	if isAllZero {
		t.Errorf("Output matrix is all zeros, which is unlikely.")
	}
}

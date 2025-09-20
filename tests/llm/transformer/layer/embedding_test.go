package layer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
	test_core "github.com/pplmx/algo-go/tests/llm/transformer/core"
)

func TestEmbedding_Forward(t *testing.T) {
	// 1. 配置和初始化
	vocabSize := 10
	dModel := 4
	embedding := layer.NewEmbedding(vocabSize, dModel)

	// 为了测试确定性，手动设置权重
	weights := make(core.Matrix, vocabSize)
	for i := 0; i < vocabSize; i++ {
		weights[i] = make([]float64, dModel)
		for j := 0; j < dModel; j++ {
			weights[i][j] = float64(i*10 + j)
		}
	}
	embedding.Weight = weights

	// 2. 准备输入数据
	// batchSize=2, seqLen=3
	input := [][]int{
		{1, 3, 5},
		{2, 4, 6},
	}

	// 3. 计算预期输出
	batchSize := len(input)
	seqLen := len(input[0])
	expectedOutput := make(core.Matrix, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			tokenID := input[i][j]
			rowIndex := i*seqLen + j
			expectedOutput[rowIndex] = weights[tokenID]
		}
	}

	// 4. 执行前向传播
	output := embedding.Forward(input)

	// 5. 验证结果
	if !test_core.MatricesAlmostEqual(output, expectedOutput, 1e-9) {
		t.Errorf("Forward() output = %v, want %v", output, expectedOutput)
	}
}

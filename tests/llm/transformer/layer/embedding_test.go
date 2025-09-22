package layer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
	"github.com/pplmx/algo-go/tests/llm/helpers"
)

func TestEmbedding_Forward(t *testing.T) {
	// 1. 配置和初始化
	vocabSize := 10
	dModel := 4
	embedding := layer.NewEmbedding(vocabSize, dModel)

	// 为了测试确定性，手动设置权重
	weightsData := make([]float64, vocabSize*dModel)
	for i := 0; i < vocabSize; i++ {
		for j := 0; j < dModel; j++ {
			weightsData[i*dModel+j] = float64(i*10 + j)
		}
	}
	embedding.Weight = core.NewTensorFromData(weightsData, vocabSize, dModel)

	// 2. 准备输入数据
	// batchSize=2, seqLen=3
	input := [][]int{
		{1, 3, 5},
		{2, 4, 6},
	}

	// 3. 计算预期输出
	batchSize := len(input)
	seqLen := len(input[0])
	expectedOutputData := make([]float64, batchSize*seqLen*dModel)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			tokenID := input[i][j]
			// Get the row from the embedding weights (which is now a *core.Tensor)
			for k := 0; k < dModel; k++ {
				expectedOutputData[(i*seqLen+j)*dModel+k] = embedding.Weight.Get(tokenID, k)
			}
		}
	}
	expectedOutput := core.NewTensorFromData(expectedOutputData, batchSize*seqLen, dModel)

	// 4. 执行前向传播
	output := embedding.Forward(input)

	// 5. 验证结果
	if !helpers.TensorsAlmostEqual(output, expectedOutput, 1e-9) {
		t.Errorf("Forward() output = %v, want %v", output, expectedOutput)
	}
}

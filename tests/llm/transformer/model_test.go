package transformer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer"
	"github.com/pplmx/algo-go/llm/transformer/config"
)

func TestTransformerModel_Forward(t *testing.T) {
	// 1. 配置
	cfg := config.TransformerConfig{
		DModel:        8,
		VocabSize:     10,
		MaxSeqLen:     10,
		NumHeads:      2,
		DK:            4,
		DV:            4,
		FFNHiddenSize: 16,
		DropoutRate:   0.0,
		UseBias:       false,
	}
	batchSize := 2
	seqLen := 5

	// 2. 初始化模型
	model := transformer.NewTransformerModel(cfg)
	model.SetTraining(false)

	// 3. 准备输入数据
	input := make([][]int, batchSize)
	for i := range input {
		input[i] = make([]int, seqLen)
		for j := range input[i] {
			input[i][j] = (i*seqLen + j) % cfg.VocabSize
		}
	}

	// 4. 执行前向传播
	logits, attnWeights := model.Forward(input, nil)

	// 5. 验证输出维度
	if len(logits) != batchSize*seqLen {
		t.Errorf("Logits output rows = %d, want %d", len(logits), batchSize*seqLen)
	}
	if len(logits[0]) != cfg.VocabSize {
		t.Errorf("Logits output columns = %d, want %d", len(logits[0]), cfg.VocabSize)
	}

	// 6. 验证注意力权重维度
	numEncoderLayers := 6 // Hardcoded in NewTransformerModel
	if len(attnWeights) != numEncoderLayers {
		t.Errorf("Number of encoder layers in attention weights = %d, want %d", len(attnWeights), numEncoderLayers)
	}
}

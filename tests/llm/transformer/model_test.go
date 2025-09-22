package transformer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer"
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
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
	// Dummy target input and masks
	tgtInput := make([][]int, batchSize)
	for i := range tgtInput {
		tgtInput[i] = make([]int, seqLen)
		for j := range tgtInput[i] {
			tgtInput[i][j] = (i*seqLen + j) % cfg.VocabSize
		}
	}
	srcMaskData := make([]float64, batchSize*cfg.NumHeads*seqLen*seqLen)
	srcMask := core.NewTensorFromData(srcMaskData, batchSize, cfg.NumHeads, seqLen, seqLen)
	tgtMaskData := make([]float64, batchSize*cfg.NumHeads*seqLen*seqLen)
	tgtMask := core.NewTensorFromData(tgtMaskData, batchSize, cfg.NumHeads, seqLen, seqLen)

	logits, encAttnWeights, selfAttnWeights, encDecAttnWeights := model.Forward(input, tgtInput, srcMask, tgtMask)

	// 5. 验证输出维度
	if logits.Shape()[0] != batchSize*seqLen {
		t.Errorf("Logits output rows = %d, want %d", logits.Shape()[0], batchSize*seqLen)
	}
	if logits.Shape()[1] != cfg.VocabSize {
		t.Errorf("Logits output columns = %d, want %d", logits.Shape()[1], cfg.VocabSize)
	}

	// 6. 验证编码器注意力权重维度
	numEncoderLayers := 6 // Hardcoded in NewTransformerModel
	if len(encAttnWeights) != numEncoderLayers {
		t.Errorf("Number of encoder layers in attention weights = %d, want %d", len(encAttnWeights), numEncoderLayers)
	}

	// 7. 验证解码器自注意力权重维度
	numDecoderLayers := 6 // Hardcoded in NewTransformerModel
	if len(selfAttnWeights) != numDecoderLayers {
		t.Errorf("Number of decoder self-attention weights = %d, want %d", len(selfAttnWeights), numDecoderLayers)
	}

	// 8. 验证解码器编码器-解码器注意力权重维度
	if len(encDecAttnWeights) != numDecoderLayers {
		t.Errorf("Number of decoder encoder-decoder attention weights = %d, want %d", len(encDecAttnWeights), numDecoderLayers)
	}
}

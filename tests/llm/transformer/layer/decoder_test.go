package layer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
)

func TestTransformerDecoderLayer_Forward(t *testing.T) {
	// 1. 配置
	cfg := config.TransformerConfig{
		DModel:        8,
		NumHeads:      2,
		DK:            4,
		DV:            4,
		FFNHiddenSize: 16,
		DropoutRate:   0.0,
		UseBias:       false,
	}
	batchSize := 1
	seqLen := 3

	// 2. 初始化 DecoderLayer
	decoderLayer := layer.NewTransformerDecoderLayer(cfg)
	decoderLayer.SetTraining(false)

	// 3. 准备输入数据
	xData := []float64{
		1, 2, 3, 4, 5, 6, 7, 8,
		8, 7, 6, 5, 4, 3, 2, 1,
		1, 3, 5, 7, 9, 2, 4, 6,
	}
	x := core.NewTensorFromData(xData, batchSize, seqLen, cfg.DModel)
	encoderOutputData := []float64{
		10, 11, 12, 13, 14, 15, 16, 17,
		17, 16, 15, 14, 13, 12, 11, 10,
		10, 12, 14, 16, 18, 11, 13, 15,
	}
	encoderOutput := core.NewTensorFromData(encoderOutputData, batchSize, seqLen, cfg.DModel)
	srcMask := core.Zeros(batchSize, 1, 1, seqLen)
	tgtMask := core.Zeros(batchSize, 1, seqLen, seqLen)

	// 4. 执行前向传播
	output, selfAttnWeights, encDecAttnWeights := decoderLayer.Forward(x, encoderOutput, srcMask, tgtMask)

	// 5. 验证输出维度
	expectedShape := []int{batchSize, seqLen, cfg.DModel}
	if len(output.Shape()) != len(expectedShape) {
		t.Fatalf("Output shape length = %d, want %d", len(output.Shape()), len(expectedShape))
	}
	for i, dim := range expectedShape {
		if output.Shape()[i] != dim {
			t.Errorf("Output shape dimension %d = %d, want %d", i, output.Shape()[i], dim)
		}
	}

	// 6. 验证注意力权重维度
	if len(selfAttnWeights) != cfg.NumHeads {
		t.Errorf("Number of self-attention heads = %d, want %d", len(selfAttnWeights), cfg.NumHeads)
	}
	if len(encDecAttnWeights) != cfg.NumHeads {
		t.Errorf("Number of encoder-decoder attention heads = %d, want %d", len(encDecAttnWeights), cfg.NumHeads)
	}
}

func TestTransformerDecoder_Forward(t *testing.T) {
	// 1. 配置
	numLayers := 3
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

	// 2. 初始化 Decoder
	decoder := layer.NewTransformerDecoder(cfg, numLayers)
	decoder.SetTraining(false)

	// 3. 准备输入数据
	tgtInput := make([][]int, batchSize)
	for i := range tgtInput {
		tgtInput[i] = make([]int, seqLen)
		for j := range tgtInput[i] {
			tgtInput[i][j] = (i*seqLen + j) % cfg.VocabSize
		}
	}
	encoderOutputData := make([]float64, batchSize*seqLen*cfg.DModel)
	for i := 0; i < batchSize*seqLen; i++ {
		for j := 0; j < cfg.DModel; j++ {
			encoderOutputData[i*cfg.DModel+j] = float64(i*cfg.DModel + j)
		}
	}
	encoderOutput := core.NewTensorFromData(encoderOutputData, batchSize, seqLen, cfg.DModel)

	srcMask := core.Zeros(batchSize, 1, 1, seqLen)
	tgtMask := core.Zeros(batchSize, 1, seqLen, seqLen)

	// 4. 执行前向传播
	output, allSelfAttnWeights, allEncDecAttnWeights := decoder.Forward(tgtInput, encoderOutput, srcMask, tgtMask)

	// 5. 验证输出维度
	expectedShape := []int{batchSize, seqLen, cfg.DModel}
	if len(output.Shape()) != len(expectedShape) {
		t.Fatalf("Output shape length = %d, want %d", len(output.Shape()), len(expectedShape))
	}
	for i, dim := range expectedShape {
		if output.Shape()[i] != dim {
			t.Errorf("Output shape dimension %d = %d, want %d", i, output.Shape()[i], dim)
		}
	}

	// 6. 验证注意力权重维度
	if len(allSelfAttnWeights) != numLayers {
		t.Errorf("Number of decoder self-attention weight layers = %d, want %d", len(allSelfAttnWeights), numLayers)
	}
	if len(allEncDecAttnWeights) != numLayers {
		t.Errorf("Number of decoder encoder-decoder attention weight layers = %d, want %d", len(allEncDecAttnWeights), numLayers)
	}
}

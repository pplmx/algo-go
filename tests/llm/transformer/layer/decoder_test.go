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
	seqLen := 3

	// 2. 初始化 DecoderLayer
	decoderLayer := layer.NewTransformerDecoderLayer(cfg)
	decoderLayer.SetTraining(false)

	// 3. 准备输入数据
	x := core.Matrix{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{8, 7, 6, 5, 4, 3, 2, 1},
		{1, 3, 5, 7, 9, 2, 4, 6},
	}
	encoderOutput := core.Matrix{
		{10, 11, 12, 13, 14, 15, 16, 17},
		{17, 16, 15, 14, 13, 12, 11, 10},
		{10, 12, 14, 16, 18, 11, 13, 15},
	}
	srcMask := make(core.Matrix, seqLen)
	for i := range srcMask {
		srcMask[i] = make([]float64, seqLen)
	}
	tgtMask := make(core.Matrix, seqLen)
	for i := range tgtMask {
		tgtMask[i] = make([]float64, seqLen)
	}

	// 4. 执行前向传播
	output, selfAttnWeights, encDecAttnWeights := decoderLayer.Forward(x, encoderOutput, srcMask, tgtMask)

	// 5. 验证输出维度
	if len(output) != seqLen {
		t.Errorf("Output sequence length = %d, want %d", len(output), seqLen)
	}
	if len(output[0]) != cfg.DModel {
		t.Errorf("Output dimension = %d, want %d", len(output[0]), cfg.DModel)
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
	encoderOutput := make(core.Matrix, batchSize*seqLen)
	for i := range encoderOutput {
		encoderOutput[i] = make([]float64, cfg.DModel)
		for j := range encoderOutput[i] {
			encoderOutput[i][j] = float64(i*cfg.DModel + j)
		}
	}

	srcMask := make(core.Matrix, batchSize*seqLen)
	for i := range srcMask {
		srcMask[i] = make([]float64, batchSize*seqLen)
	}
	tgtMask := make(core.Matrix, batchSize*seqLen)
	for i := range tgtMask {
		tgtMask[i] = make([]float64, batchSize*seqLen)
	}

	// 4. 执行前向传播
	output, allSelfAttnWeights, allEncDecAttnWeights := decoder.Forward(tgtInput, encoderOutput, srcMask, tgtMask)

	// 5. 验证输出维度
	if len(output) != batchSize*seqLen {
		t.Errorf("Output sequence length = %d, want %d", len(output), batchSize*seqLen)
	}
	if len(output[0]) != cfg.DModel {
		t.Errorf("Output dimension = %d, want %d", len(output[0]), cfg.DModel)
	}

	// 6. 验证注意力权重维度
	if len(allSelfAttnWeights) != numLayers {
		t.Errorf("Number of decoder self-attention weight layers = %d, want %d", len(allSelfAttnWeights), numLayers)
	}
	if len(allEncDecAttnWeights) != numLayers {
		t.Errorf("Number of decoder encoder-decoder attention weight layers = %d, want %d", len(allEncDecAttnWeights), numLayers)
	}
}

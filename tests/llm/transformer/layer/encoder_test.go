package layer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
)

func TestTransformerEncoderLayer_Forward(t *testing.T) {
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

	// 2. 初始化 EncoderLayer
	encoderLayer := layer.NewTransformerEncoderLayer(cfg)
	encoderLayer.SetTraining(false)

	// 3. 准备输入数据
	inputData := []float64{
		1, 2, 3, 4, 5, 6, 7, 8,
		8, 7, 6, 5, 4, 3, 2, 1,
		1, 3, 5, 7, 9, 2, 4, 6,
	}
	input := core.NewTensorFromData(inputData, batchSize, seqLen, cfg.DModel)

	// 4. 执行前向传播
	output, attnWeights := encoderLayer.Forward(input, nil)

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
	if len(attnWeights) != cfg.NumHeads {
		t.Errorf("Number of attention heads = %d, want %d", len(attnWeights), cfg.NumHeads)
	}
}

func TestTransformerEncoder_Forward(t *testing.T) {
	// 1. 配置
	numLayers := 3
	cfg := config.TransformerConfig{
		DModel:        8,
		NumHeads:      2,
		DK:            4,
		DV:            4,
		FFNHiddenSize: 16,
		MaxSeqLen:     10,
		VocabSize:     10, // Add vocab size
		DropoutRate:   0.0,
		UseBias:       false,
	}
	batchSize := 2
	seqLen := 5

	// 2. 初始化 Encoder
	encoder := layer.NewTransformerEncoder(cfg, numLayers)
	encoder.SetTraining(false)

	// 3. 准备输入数据
	input := make([][]int, batchSize)
	for i := range input {
		input[i] = make([]int, seqLen)
		for j := range input[i] {
			input[i][j] = (i*seqLen + j) % 10 // Use a dummy vocab size
		}
	}

	// 4. 执行前向传播
	output, allAttnWeights := encoder.Forward(input, nil)

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
	if len(allAttnWeights) != numLayers {
		t.Errorf("Number of layers in attention weights = %d, want %d", len(allAttnWeights), numLayers)
	}
	for i, layerWeights := range allAttnWeights {
		if len(layerWeights) != cfg.NumHeads {
			t.Errorf("Number of attention heads in layer %d = %d, want %d", i, len(layerWeights), cfg.NumHeads)
		}
	}
}

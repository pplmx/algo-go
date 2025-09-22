package layer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
)

func TestMultiHeadAttention_Forward(t *testing.T) {
	// 1. 配置
	cfg := config.TransformerConfig{
		DModel:      8,
		NumHeads:    2,
		DK:          4,
		DV:          4,
		UseBias:     false,
		DropoutRate: 0.0,
	}
	batchSize := 1
	seqLen := 3

	// 2. 初始化 MHA 层
	mha := layer.NewMultiHeadAttention(cfg)
	mha.SetTraining(false) // 关闭 dropout

	// 3. 准备输入数据
	inputData := []float64{
		1, 2, 3, 4, 5, 6, 7, 8,
		8, 7, 6, 5, 4, 3, 2, 1,
		1, 3, 5, 7, 9, 2, 4, 6,
	}
	input := core.NewTensorFromData(inputData, batchSize, seqLen, cfg.DModel)

	// 4. 执行前向传播
	output, weights := mha.Forward(input, input, input, nil)

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
	if len(weights) != cfg.NumHeads {
		t.Errorf("Number of attention heads = %d, want %d", len(weights), cfg.NumHeads)
	}
	for i, w := range weights {
		expectedWeightShape := []int{batchSize, seqLen, seqLen}
		if len(w.Shape()) != len(expectedWeightShape) {
			t.Fatalf("Attention weights for head %d shape length = %d, want %d", i, len(w.Shape()), len(expectedWeightShape))
		}
		for j, dim := range expectedWeightShape {
			if w.Shape()[j] != dim {
				t.Errorf("Attention weights for head %d shape dimension %d = %d, want %d", i, j, w.Shape()[j], dim)
			}
		}
	}

	// 7. 验证输出值 (这是一个比较复杂的场景，因为涉及到随机初始化的权重)
	// 我们可以做一个简单的 sanity check，比如检查输出不全为0
	isAllZero := true
	for i := 0; i < output.Shape()[0]; i++ {
		for j := 0; j < output.Shape()[1]; j++ {
			for k := 0; k < output.Shape()[2]; k++ {
				if output.Get(i, j, k) != 0 {
					isAllZero = false
					break
				}
			}
			if !isAllZero {
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

package layer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
	test_core "github.com/pplmx/algo-go/tests/llm/helpers"
)

func TestPositionalEncoding_Forward(t *testing.T) {
	// 1. 配置
	maxSeqLen := 10
	dModel := 8
	seqLen := 5

	// 2. 初始化 PositionalEncoding 层
	pe := layer.NewPositionalEncoding(maxSeqLen, dModel)

	// 3. 准备输入数据 (全零矩阵)
	input := make(core.Matrix, seqLen)
	for i := range input {
		input[i] = make([]float64, dModel)
	}

	// 4. 执行前向传播
	output := pe.Forward(input)

	// 5. 验证输出维度
	if len(output) != seqLen {
		t.Errorf("Output sequence length = %d, want %d", len(output), seqLen)
	}
	if len(output[0]) != dModel {
		t.Errorf("Output dimension = %d, want %d", len(output[0]), dModel)
	}

	// 6. 验证输出是否等于位置编码本身 (因为输入是零)
	expectedOutput := make(core.Matrix, seqLen)
	for i := 0; i < seqLen; i++ {
		expectedOutput[i] = pe.Encoding[i]
	}

	if !test_core.MatricesAlmostEqual(output, expectedOutput, 1e-9) {
		t.Errorf("Output is not equal to the positional encoding.\nGot:  %v\nWant: %v", output, expectedOutput)
	}

	// 7. 验证编码值范围
	for _, row := range output {
		for _, val := range row {
			if val < -1.0 || val > 1.0 {
				t.Errorf("Positional encoding value %f is outside the expected range [-1, 1]", val)
			}
		}
	}

	// 8. 验证输入不被修改
	for _, row := range input {
		for _, val := range row {
			if val != 0 {
				t.Errorf("Input matrix was modified during forward pass")
				break
			}
		}
	}
}

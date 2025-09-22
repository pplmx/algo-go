package layer_test

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
	"github.com/pplmx/algo-go/tests/llm/helpers"
)

func TestPositionalEncoding_Forward(t *testing.T) {
	// 1. 配置
	maxSeqLen := 10
	dModel := 8
	seqLen := 5

	// 2. 初始化 PositionalEncoding 层
	pe := layer.NewPositionalEncoding(maxSeqLen, dModel)

	// 3. 准备输入数据 (全零矩阵)
	input := core.Zeros(1, seqLen, dModel)

	// 4. 执行前向传播
	output := pe.Forward(input, seqLen)

	// 5. 验证输出维度
	if output.Shape()[1] != seqLen {
		t.Errorf("Output sequence length = %d, want %d", output.Shape()[1], seqLen)
	}
	if output.Shape()[2] != dModel {
		t.Errorf("Output dimension = %d, want %d", output.Shape()[2], dModel)
	}

	// 6. 验证输出是否等于位置编码本身 (因为输入是零)
	expectedOutput := core.NewTensor(1, seqLen, dModel)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			expectedOutput.Set(pe.Encoding.Get(i, j), 0, i, j)
		}
	}

	if !helpers.TensorsAlmostEqual(output, expectedOutput, 1e-9) {
		t.Errorf("Output is not equal to the positional encoding.\nGot:  %v\nWant: %v", output, expectedOutput)
	}

	// 7. 验证编码值范围
	for i := 0; i < output.Shape()[0]; i++ {
		for j := 0; j < output.Shape()[1]; j++ {
			for k := 0; k < output.Shape()[2]; k++ {
				val := output.Get(i, j, k)
				if val < -1.0 || val > 1.0 {
					t.Errorf("Positional encoding value %f is outside the expected range [-1, 1]", val)
				}
			}
		}
	}

	// 8. 验证输入不被修改
	for i := 0; i < input.Shape()[0]; i++ {
		for j := 0; j < input.Shape()[1]; j++ {
			for k := 0; k < input.Shape()[2]; k++ {
				if input.Get(i, j, k) != 0 {
					t.Errorf("Input matrix was modified during forward pass")
					break
				}
			}
		}
	}
}

package core

import (
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
)

func TestDropout_EvalMode(t *testing.T) {
	// 1. 初始化为评估模式
	dropout := core.NewDropout(0.5, false)

	// 2. 准备输入数据
	input := core.Matrix{{1, 2, 3}, {4, 5, 6}}

	// 3. 执行前向传播
	output := dropout.Forward(input)

	// 4. 验证输出是否与输入完全相同
	if !MatricesAlmostEqual(input, output, 1e-9) {
		t.Errorf("In eval mode, dropout output should be identical to input. Got %v, want %v", output, input)
	}
}

func TestDropout_TrainMode(t *testing.T) {
	// 1. 初始化为训练模式
	rate := 0.5
	dropout := core.NewDropout(rate, true)

	// 2. 准备输入数据
	input := core.Matrix{
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	}

	// 3. 执行前向传播
	output := dropout.Forward(input)

	// 4. 验证
	zeroCount := 0
	nonZeroCount := 0
	expectedScaledValue := 1.0 / (1.0 - rate)

	for _, row := range output {
		for _, val := range row {
			if val == 0 {
				zeroCount++
			} else {
				nonZeroCount++
				if !almostEqual(val, expectedScaledValue, 1e-9) {
					t.Errorf("Non-zero value should be scaled by 1/(1-rate). Got %f, want %f", val, expectedScaledValue)
				}
			}
		}
	}

	if zeroCount == 0 {
		t.Errorf("In train mode, some values should be zeroed out, but none were.")
	}

	if nonZeroCount == 0 {
		t.Errorf("In train mode, some values should be non-zero, but all were zeroed out.")
	}
}

// Helper for float comparison
func almostEqual(a, b, tolerance float64) bool {
	return (a-b) < tolerance && (b-a) < tolerance
}

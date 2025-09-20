package core_test

import (
	"math"
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/tests/llm/helpers"
)

func TestCrossEntropyLoss(t *testing.T) {
	// 1. 配置和初始化
	cfg := config.TransformerConfig{VocabSize: 4}
	lossFunc := core.NewCrossEntropyLoss(cfg)

	// 2. 准备输入数据
	// batchSize=2, seqLen=3, vocabSize=4
	logits := core.Matrix{
		{0.1, 0.2, 0.3, 0.4}, // batch 0, seq 0
		{0.5, 0.6, 0.7, 0.8}, // batch 0, seq 1
		{0.9, 1.0, 1.1, 1.2}, // batch 0, seq 2
		{0.4, 0.3, 0.2, 0.1}, // batch 1, seq 0
		{0.8, 0.7, 0.6, 0.5}, // batch 1, seq 1
		{1.2, 1.1, 1.0, 0.9}, // batch 1, seq 2
	}
	targets := [][]int{
		{3, 2, 1}, // batch 0 targets
		{0, 1, 2}, // batch 1 targets
	}

	// 3. 手动计算预期结果
	// 3.1. Softmax
	softmax := func(row []float64) []float64 {
		maxVal := row[0]
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}
		expVals := make([]float64, len(row))
		sumExp := 0.0
		for i, v := range row {
			expVals[i] = math.Exp(v - maxVal)
			sumExp += expVals[i]
		}
		for i := range expVals {
			expVals[i] /= sumExp
		}
		return expVals
	}

	probs := make(core.Matrix, len(logits))
	for i, row := range logits {
		probs[i] = softmax(row)
	}

	// 3.2. Loss
	expectedLoss := 0.0
	batchSize := len(targets)
	seqLen := len(targets[0])
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			rowIndex := i*seqLen + j
			targetIdx := targets[i][j]
			expectedLoss += -math.Log(probs[rowIndex][targetIdx] + 1e-10)
		}
	}
	expectedLoss /= float64(batchSize * seqLen)

	// 3.3. Gradient
	expectedGrad := make(core.Matrix, len(probs))
	for i := range probs {
		expectedGrad[i] = make([]float64, len(probs[i]))
		copy(expectedGrad[i], probs[i])
	}
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			rowIndex := i*seqLen + j
			targetIdx := targets[i][j]
			expectedGrad[rowIndex][targetIdx] -= 1.0
		}
	}
	expectedGrad = core.ScaleMatrix(expectedGrad, 1.0/float64(batchSize*seqLen))

	// 4. 执行前向和后向传播
	loss := lossFunc.Forward(logits, targets)
	grad := lossFunc.Backward(logits, targets)

	// 5. 验证结果
	if math.Abs(loss-expectedLoss) > 1e-9 {
		t.Errorf("Forward() loss = %v, want %v", loss, expectedLoss)
	}

	if !helpers.MatricesAlmostEqual(grad, expectedGrad, 1e-9) {
		t.Errorf("Backward() grad = %v, want %v", grad, expectedGrad)
	}
}

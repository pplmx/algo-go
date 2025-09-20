package core

import (
	"math"

	"github.com/pplmx/algo-go/llm/transformer/config"
)

// 交叉熵损失
type CrossEntropyLoss struct {
	Config config.TransformerConfig
}

// NewCrossEntropyLoss creates a new CrossEntropyLoss function.
func NewCrossEntropyLoss(cfg config.TransformerConfig) *CrossEntropyLoss {
	return &CrossEntropyLoss{Config: cfg}
}

// Forward calculates the cross-entropy loss.
func (c *CrossEntropyLoss) Forward(logits Matrix, targets [][]int) float64 {
	batchSize := len(targets)
	seqLen := len(targets[0])

	probs := c.softmax(logits)

	loss := 0.0
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			rowIndex := i*seqLen + j
			targetIdx := targets[i][j]
			prob := probs[rowIndex][targetIdx]
			loss += -math.Log(prob + 1e-10)
		}
	}

	return loss / float64(batchSize*seqLen)
}

// Backward calculates the gradients for the cross-entropy loss.
func (c *CrossEntropyLoss) Backward(logits Matrix, targets [][]int) Matrix {
	probs := c.softmax(logits)
	batchSize := len(targets)
	seqLen := len(targets[0])

	grad := probs // softmax returns a new matrix, so we can modify it in place

	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			rowIndex := i*seqLen + j
			targetIdx := targets[i][j]
			grad[rowIndex][targetIdx] -= 1.0
		}
	}

	return ScaleMatrix(grad, 1.0/float64(batchSize*seqLen))
}

// softmax applies the softmax function row-wise to a matrix.
func (c *CrossEntropyLoss) softmax(x Matrix) Matrix {
	result := make(Matrix, len(x))
	for i := range x {
		result[i] = make([]float64, len(x[i]))

		// 减去最大值以提高数值稳定性
		maxVal := x[i][0]
		for j := 1; j < len(x[i]); j++ {
			if x[i][j] > maxVal {
				maxVal = x[i][j]
			}
		}

		// 计算指数和
		sum := 0.0
		for j := range x[i] {
			result[i][j] = math.Exp(x[i][j] - maxVal)
			sum += result[i][j]
		}

		// 应用 softmax
		for j := range result[i] {
			result[i][j] /= sum
		}
	}
	return result
}

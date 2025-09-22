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
func (c *CrossEntropyLoss) Forward(logits *Tensor, targets [][]int) float64 {
	batchSize := len(targets)
	seqLen := len(targets[0])

	probs := logits.Softmax(len(logits.Shape()) - 1)

	loss := 0.0
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			rowIndex := i*seqLen + j
			targetIdx := targets[i][j]
			prob := probs.Data()[rowIndex*probs.Shape()[len(probs.Shape())-1]+targetIdx]
			loss += -math.Log(prob + 1e-10)
		}
	}

	return loss / float64(batchSize*seqLen)
}

// Backward calculates the gradients for the cross-entropy loss.
func (c *CrossEntropyLoss) Backward(logits *Tensor, targets [][]int) *Tensor {
	probs := logits.Softmax(len(logits.Shape()) - 1)
	batchSize := len(targets)
	seqLen := len(targets[0])

	grad := probs.Clone() // Clone to avoid modifying the original tensor

	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			rowIndex := i*seqLen + j
			targetIdx := targets[i][j]
			grad.Data()[rowIndex*grad.Shape()[len(grad.Shape())-1]+targetIdx] -= 1.0
		}
	}

	return grad.MulScalar(1.0 / float64(batchSize*seqLen))
}

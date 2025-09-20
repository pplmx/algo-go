package transformer

import (
	"math"

	"github.com/pplmx/algo-go/llm/transformer/core"
)

// Perplexity calculates the perplexity of the model's predictions.
// Perplexity is defined as exp(cross_entropy_loss).
func Perplexity(logits core.Matrix, targets [][]int, lossFunc *core.CrossEntropyLoss) float64 {
	loss := lossFunc.Forward(logits, targets)
	return math.Exp(loss)
}

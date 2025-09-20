package transformer_test

import (
	"math"
	"testing"

	"github.com/pplmx/algo-go/llm/transformer"
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
)

func TestPerplexity(t *testing.T) {
	// 1. Setup
	cfg := config.TransformerConfig{VocabSize: 4}
	lossFunc := core.NewCrossEntropyLoss(cfg)

	// 2. Prepare input data
	// batchSize=1, seqLen=1, vocabSize=4
	logits := core.Matrix{{
		math.Log(0.1), // log(prob for token 0)
		math.Log(0.2), // log(prob for token 1)
		math.Log(0.3), // log(prob for token 2)
		math.Log(0.4), // log(prob for token 3)
	}}
	targets := [][]int{{3}} // Target is token 3

	// 3. Calculate expected perplexity manually
	// Cross-entropy loss for this single example:
	// -log(P(target_token)) = -log(0.4) = 0.91629073187
	// Perplexity = exp(loss) = exp(0.91629073187) = 2.5
	expectedPerplexity := 2.5

	// 4. Calculate perplexity using the function
	calculatedPerplexity := transformer.Perplexity(logits, targets, lossFunc)

	// 5. Verify result
	if math.Abs(calculatedPerplexity-expectedPerplexity) > 1e-9 {
		t.Errorf("Perplexity incorrect. Got %f, want %f", calculatedPerplexity, expectedPerplexity)
	}
}

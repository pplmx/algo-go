package transformer_test

import (
	"math"
	"testing"

	"github.com/pplmx/algo-go/llm/transformer"
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
	test_core "github.com/pplmx/algo-go/tests/llm/transformer/core"
)

func TestTrainer_generateLookAheadMask(t *testing.T) {
	cfg := config.DefaultConfig()
	trainer := transformer.NewTrainer(nil, nil, nil, cfg) // Model, Optimizer, LossFunc are not needed for this test

	seqLen := 3
	mask := trainer.GenerateLookAheadMask(seqLen)

	expectedMask := core.Matrix{
		{0.0, -math.MaxFloat64, -math.MaxFloat64},
		{0.0, 0.0, -math.MaxFloat64},
		{0.0, 0.0, 0.0},
	}

	if !test_core.MatricesAlmostEqual(mask, expectedMask, 1e-9) {
		t.Errorf("Look-ahead mask incorrect.\nGot:  %v\nWant: %v", mask, expectedMask)
	}
}

func TestTrainer_combineMasks(t *testing.T) {
	cfg := config.DefaultConfig()
	trainer := transformer.NewTrainer(nil, nil, nil, cfg)

	mask1 := core.Matrix{
		{0.0, -math.MaxFloat64},
		{0.0, 0.0},
	}
	mask2 := core.Matrix{
		{0.0, 0.0},
		{-math.MaxFloat64, 0.0},
	}

	combined := trainer.CombineMasks(mask1, mask2)

	expectedCombined := core.Matrix{
		{0.0, -math.MaxFloat64},
		{-math.MaxFloat64, 0.0},
	}

	if !test_core.MatricesAlmostEqual(combined, expectedCombined, 1e-9) {
		t.Errorf("Combined mask incorrect.\nGot:  %v\nWant: %v", combined, expectedCombined)
	}
}

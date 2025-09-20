package core_test

import (
	"math"
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/tests/llm/helpers"
)

func TestGenerateLookAheadMask(t *testing.T) {
	seqLen := 3
	mask := core.GenerateLookAheadMask(seqLen)

	expectedMask := core.Matrix{
		{0.0, -math.MaxFloat64, -math.MaxFloat64},
		{0.0, 0.0, -math.MaxFloat64},
		{0.0, 0.0, 0.0},
	}

	if !helpers.MatricesAlmostEqual(mask, expectedMask, 1e-9) {
		t.Errorf("Look-ahead mask incorrect.\nGot:  %v\nWant: %v", mask, expectedMask)
	}
}

func TestCombineMasks(t *testing.T) {
	mask1 := core.Matrix{
		{0.0, -math.MaxFloat64},
		{0.0, 0.0},
	}
	mask2 := core.Matrix{
		{0.0, 0.0},
		{-math.MaxFloat64, 0.0},
	}

	combined := core.CombineMasks(mask1, mask2)

	expectedCombined := core.Matrix{
		{0.0, -math.MaxFloat64},
		{-math.MaxFloat64, 0.0},
	}

	if !helpers.MatricesAlmostEqual(combined, expectedCombined, 1e-9) {
		t.Errorf("Combined mask incorrect.\nGot:  %v\nWant: %v", combined, expectedCombined)
	}
}

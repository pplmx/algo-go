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

	expectedMask := core.NewTensorFromData([]float64{
		0.0, -math.MaxFloat64, -math.MaxFloat64,
		0.0, 0.0, -math.MaxFloat64,
		0.0, 0.0, 0.0,
	}, 3, 3)

	if !helpers.TensorsAlmostEqual(mask, expectedMask, 1e-9) {
		t.Errorf("Look-ahead mask incorrect.\nGot:  %v\nWant: %v", mask, expectedMask)
	}
}

func TestCombineMasks(t *testing.T) {
	seqLen := 4
	batchSize := 2

	// Create a (batch, 1, 1, seq_len) padding mask
	paddingMask := core.Zeros(batchSize, 1, 1, seqLen)
	paddingMask.Set(-math.MaxFloat64, 0, 0, 0, 3) // Mask last token of first sequence in batch
	paddingMask.Set(-math.MaxFloat64, 1, 0, 0, 2) // Mask last two tokens of second sequence
	paddingMask.Set(-math.MaxFloat64, 1, 0, 0, 3)

	// Create a (seq_len, seq_len) look-ahead mask
	lookAheadMask := core.GenerateLookAheadMask(seqLen)

	// Combine them
	combined := core.CombineMasks(paddingMask, lookAheadMask)

	// Check shape
	expectedShape := []int{batchSize, 1, seqLen, seqLen}
	if !core.Shape(combined.Shape()).Equal(expectedShape) {
		t.Fatalf("Expected shape %v, but got %v", expectedShape, combined.Shape())
	}

	// Check some values
	// Example: First sequence in batch, first row of attention scores
	// Should be able to see token 0, but not 1, 2, 3 (look-ahead)
	if combined.Get(0, 0, 0, 0) != 0 {
		t.Error("pos (0,0) should be 0")
	}
	if combined.Get(0, 0, 0, 1) == 0 {
		t.Error("pos (0,1) should be masked")
	}
	if combined.Get(0, 0, 0, 2) == 0 {
		t.Error("pos (0,2) should be masked")
	}
	if combined.Get(0, 0, 0, 3) == 0 {
		t.Error("pos (0,3) should be masked (padding)")
	}

	// Example: Second sequence, third row
	// Should see 0, 1, 2. Token 2 is a pad token, so it should be masked for all queries.
	if combined.Get(1, 0, 2, 0) != 0 {
		t.Error("pos (2,0) should be 0")
	}
	if combined.Get(1, 0, 2, 1) != 0 {
		t.Error("pos (2,1) should be 0")
	}
	if combined.Get(1, 0, 2, 2) == 0 {
		t.Error("pos (2,2) should be masked (padding)")
	}
	if combined.Get(1, 0, 2, 3) == 0 {
		t.Error("pos (2,3) should be masked (padding)")
	}
}

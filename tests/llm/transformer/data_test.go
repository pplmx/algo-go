package transformer_test

import (
	"math"
	"testing"

	"github.com/pplmx/algo-go/llm/transformer"
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/tests/llm/helpers"
)

func TestDataset(t *testing.T) {
	cfg := config.TransformerConfig{MaxSeqLen: 10, PadToken: 0}
	src := [][]int{{1, 2, 3}, {4, 5}}
	tgt := [][]int{{6, 7, 8}, {9, 10}}

	ds := transformer.NewDataset(src, tgt, cfg)

	if len(ds.SourceSentences) != 2 || len(ds.TargetSentences) != 2 {
		t.Errorf("Dataset size incorrect")
	}
}

func TestDataLoader_NextBatch(t *testing.T) {
	cfg := config.TransformerConfig{MaxSeqLen: 5, PadToken: 0}
	src := [][]int{{1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {10}}
	tgt := [][]int{{1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {10}}
	ds := transformer.NewDataset(src, tgt, cfg)
	dl := transformer.NewDataLoader(ds, 2)

	// First batch
	srcBatch, tgtBatch, srcMask, _ := dl.NextBatch() // tgtMask is not used in this test

	if len(srcBatch) != 2 || len(tgtBatch) != 2 {
		t.Errorf("Batch size incorrect, got %d", len(srcBatch))
	}

	// Check padding
	expectedSrc1 := []int{1, 2, 3, 0, 0}
	expectedSrc2 := []int{4, 5, 0, 0, 0}
	if !helpers.IntSliceEqual(srcBatch[0], expectedSrc1) || !helpers.IntSliceEqual(srcBatch[1], expectedSrc2) {
		t.Errorf("Padding incorrect for srcBatch. Got %v, %v", srcBatch[0], srcBatch[1])
	}

	// Check mask dimensions (batchSize * MaxSeqLen, batchSize * MaxSeqLen)
	expectedMaskDim := 2 * 5 // batchSize * MaxSeqLen
	if len(srcMask) != expectedMaskDim || len(srcMask[0]) != expectedMaskDim {
		t.Errorf("SrcMask dimensions incorrect. Got %dx%d, want %dx%d", len(srcMask), len(srcMask[0]), expectedMaskDim, expectedMaskDim)
	}

	// Check some mask values (e.g., padding positions should be -inf)
	// For srcBatch[0] = {1,2,3,0,0}, the last two tokens are padding
	// The corresponding rows in the mask should be -inf
	// Row for token 3 (index 2) in batch 0 is 0*5+2 = 2
	// Row for token 4 (index 3) in batch 0 is 0*5+3 = 3
	// Row for token 5 (index 4) in batch 0 is 0*5+4 = 4
	// Row for token 2 (index 1) in batch 1 is 1*5+1 = 6
	// Row for token 3 (index 2) in batch 1 is 1*5+2 = 7
	// Row for token 4 (index 3) in batch 1 is 1*5+3 = 8
	// Row for token 5 (index 4) in batch 1 is 1*5+4 = 9

	// Check a padding position (e.g., the first padded token in the first sequence)
	if srcMask[3][0] != -math.MaxFloat64 {
		t.Errorf("Padding mask value incorrect for padded token. Got %f, want -MaxFloat64", srcMask[3][0])
	}

	// Check a non-padding position
	if srcMask[0][0] != 0.0 {
		t.Errorf("Padding mask value incorrect for non-padded token. Got %f, want 0.0", srcMask[0][0])
	}

	// Second batch
	srcBatch, _, _, _ = dl.NextBatch()
	if len(srcBatch) != 2 {
		t.Errorf("Second batch size incorrect, got %d", len(srcBatch))
	}

	// No more batches
	if dl.HasNextBatch() {
		t.Errorf("HasNextBatch should be false")
	}

	dl.Reset()
	if !dl.HasNextBatch() {
		t.Errorf("After reset, HasNextBatch should be true")
	}
}

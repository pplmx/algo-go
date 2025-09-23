package transformer_test

import (
	"math"
	"testing"

	"github.com/pplmx/algo-go/llm/transformer"
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
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

	// Check mask dimensions
	expectedShape := []int{2, 1, 1, 5}
	if !core.Shape(srcMask.Shape()).Equal(expectedShape) {
		t.Errorf("SrcMask dimensions incorrect. Got %v, want %v", srcMask.Shape(), expectedShape)
	}

	// Check a padding position (e.g., the first padded token in the first sequence)
	if srcMask.Get(0, 0, 0, 3) != -math.MaxFloat64 {
		t.Errorf("Padding mask value incorrect for padded token. Got %f, want -MaxFloat64", srcMask.Get(0, 0, 0, 3))
	}

	// Check a non-padding position
	if srcMask.Get(0, 0, 0, 0) != 0.0 {
		t.Errorf("Padding mask value incorrect for non-padded token. Got %f, want 0.0", srcMask.Get(0, 0, 0, 0))
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

package transformer_test

import (
	"testing"
	"fmt"

	"github.com/pplmx/algo-go/llm/transformer"
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
)

func TestTrainer(t *testing.T) {
	// 1. Setup Configuration
	cfg := config.TransformerConfig{
		VocabSize:     10,
		DModel:        8,
		NumHeads:      2,
		DK:            8,
		DV:            8,
		FFNHiddenSize: 16,
		MaxSeqLen:     12,
		DropoutRate:   0.0, // No dropout for a simple test
		UseBias:       true,
		StartToken:    1,
		EndToken:      2,
		PadToken:      0,
	}
	appCfg := config.AppConfig{
		Transformer: cfg,
		Train: config.TrainConfig{
			NumEpochs: 5,
			BatchSize: 2,
			LearningRate: 0.01,
		},
	}

	// 2. Create a dummy dataset
	// Two simple sentences
	srcSentences := [][]int{
		{1, 3, 4, 5, 2}, // "start a b c end"
		{1, 6, 7, 2},    // "start x y end"
	}
	tgtSentences := [][]int{
		{1, 3, 4, 5, 2},
		{1, 6, 7, 2},
	}

	dataset := transformer.NewDataset(srcSentences, tgtSentences, cfg)
	loader := transformer.NewDataLoader(dataset, appCfg.Train.BatchSize)

	// 3. Create Model, Optimizer, Loss
	model := transformer.NewTransformerModel(cfg)
	optimizer := core.NewAdamOptimizer(appCfg.Train.LearningRate, 0.9, 0.98, 1e-9)
	lossFunc := core.NewCrossEntropyLoss(cfg)

	// 4. Create Trainer
	trainer := transformer.NewTrainer(model, optimizer, lossFunc, appCfg)

	// 5. Run training and check if loss decreases
	var firstLoss, lastLoss float64
	fmt.Println("Starting training test... This will take a moment.")
	lossHistory := trainer.Train(loader)
	fmt.Println("Training test finished.")

	// 6. Assert that the loss has decreased
	if len(lossHistory) < 2 {
		t.Fatalf("Expected at least 2 epochs to compare loss, but got %d", len(lossHistory))
	}

	firstLoss = lossHistory[0]
	lastLoss = lossHistory[len(lossHistory)-1]

	fmt.Printf("First epoch loss: %.4f\n", firstLoss)
	fmt.Printf("Last epoch loss: %.4f\n", lastLoss)

	if lastLoss >= firstLoss {
		t.Errorf("Training failed: loss did not decrease. First loss: %.4f, Last loss: %.4f", firstLoss, lastLoss)
	}

	// Also check that the loss is not NaN or Inf
	if core.IsNaN(lastLoss) || core.IsInf(lastLoss, 0) {
		t.Errorf("Training resulted in invalid loss: %.4f", lastLoss)
	}
}

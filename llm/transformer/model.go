package transformer

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
)

// 完整的 Transformer 模型
type TransformerModel struct {
	Config    config.TransformerConfig
	Encoder   *layer.TransformerEncoder
	Decoder   *layer.TransformerDecoder
	Generator *layer.LinearLayer

	lastSrcEmb    *core.Tensor
	lastEncOutput *core.Tensor
	lastDecOutput *core.Tensor
}

// NewTransformerModel creates a new TransformerModel.
func NewTransformerModel(cfg config.TransformerConfig) *TransformerModel {
	return &TransformerModel{
		Config:    cfg,
		Encoder:   layer.NewTransformerEncoder(cfg, 6),
		Decoder:   layer.NewTransformerDecoder(cfg, 6),
		Generator: layer.NewLinearLayer(cfg.DModel, cfg.VocabSize, true),
	}
}

// SetTraining sets the training mode for the TransformerModel.
func (t *TransformerModel) SetTraining(training bool) {
	t.Encoder.SetTraining(training)
	t.Decoder.SetTraining(training)
}

// Forward performs the forward pass for the TransformerModel.
func (t *TransformerModel) Forward(srcInput, tgtInput [][]int, srcMask, tgtMask *core.Tensor) (*core.Tensor, [][]*core.Tensor, [][]*core.Tensor, [][]*core.Tensor) {
	// 编码器前向传播
	encOutput, encAttnWeights := t.Encoder.Forward(srcInput, srcMask)
	t.lastEncOutput = encOutput

	// 解码器前向传播
	decOutput, selfAttnWeights, encDecAttnWeights := t.Decoder.Forward(tgtInput, encOutput, srcMask, tgtMask)
	t.lastDecOutput = decOutput

	// 生成器输出
	batchSize := decOutput.Shape()[0]
	seqLen := decOutput.Shape()[1]
	reshapedDecOutput := decOutput.Reshape(batchSize*seqLen, t.Config.DModel)
	logits := t.Generator.Forward(reshapedDecOutput)

	return logits, encAttnWeights, selfAttnWeights, encDecAttnWeights
}

// Backward performs the backward pass for the entire TransformerModel.
func (t *TransformerModel) Backward(gradOutput *core.Tensor) (*core.Tensor, *core.Tensor) {
	// The gradient flows from the final output (logits) back through the generator, decoder, and then encoder.

	// 1. Backpropagate through the Generator (a linear layer).
	// The input `gradOutput` is the gradient of the loss with respect to the logits.
	gradReshapedDecOutput := t.Generator.Backward(gradOutput)

	// 2. Reshape the gradient to match the decoder's output shape (batch, seq_len, d_model).
	batchSize := t.lastDecOutput.Shape()[0]
	seqLen := t.lastDecOutput.Shape()[1]
	gradDecOutput := gradReshapedDecOutput.Reshape(batchSize, seqLen, t.Config.DModel)

	// 3. Backpropagate through the Decoder.
	// This computes gradients for all parameters within the decoder and returns the gradient
	// with respect to the encoder's output, which is needed by the encoder's backward pass.
	gradTgtEmb, gradEncOutputFromDecoder := t.Decoder.Backward(gradDecOutput)

	// 4. Backpropagate through the Encoder.
	// This computes gradients for all parameters within the encoder, using the gradient from the decoder.
	gradSrcEmb := t.Encoder.Backward(gradEncOutputFromDecoder)

	// Note: The backward passes for the embedding layers are called within their respective
	// Encoder and Decoder backward methods, so there is no need to call them again here.

	// The returned gradients for source and target embeddings are not strictly necessary for training
	// but can be useful for debugging.
	return gradSrcEmb, gradTgtEmb
}

// GetParameters returns all trainable parameters of the TransformerModel.
func (t *TransformerModel) GetParameters() []*core.Tensor {
	var params []*core.Tensor
	params = append(params, t.Encoder.GetParameters()...)
	params = append(params, t.Decoder.GetParameters()...)
	params = append(params, t.Generator.GetParameters()...)
	return params
}

// GetGradients returns the gradients of all trainable parameters of the TransformerModel.
func (t *TransformerModel) GetGradients() []*core.Tensor {
	var grads []*core.Tensor
	grads = append(grads, t.Encoder.GetGradients()...)
	grads = append(grads, t.Decoder.GetGradients()...)
	grads = append(grads, t.Generator.GetGradients()...)
	return grads
}

// ZeroGradients sets the gradients of all trainable parameters to zero for the TransformerModel.
func (t *TransformerModel) ZeroGradients() {
	t.Encoder.ZeroGradients()
	t.Decoder.ZeroGradients()
	t.Generator.ZeroGradients()
}

// Save saves the TransformerModel to a file using gob encoding.
func (t *TransformerModel) Save(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(t); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}
	return nil
}

// Load loads a TransformerModel from a file using gob decoding.
func Load(filePath string) (*TransformerModel, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	model := &TransformerModel{}
	if err := decoder.Decode(model); err != nil {
		return nil, fmt.Errorf("failed to decode model: %w", err)
	}
	return model, nil
}

// Generate performs greedy decoding to generate a target sequence.
func (t *TransformerModel) Generate(srcInput [][]int, maxLen int) [][]int {
	t.SetTraining(false)
	batchSize := len(srcInput)

	// 1. Create source mask from the input
	// This should be done once.
	// Assuming a helper function to create padding mask from a batch.
	// We'll create a dummy one for now if it doesn't exist.
	srcMask := core.Zeros(batchSize, 1, len(srcInput[0])) // Placeholder

	// 2. Initialize target sequence with the start token
	tgtInput := make([][]int, batchSize)
	for i := range tgtInput {
		tgtInput[i] = []int{t.Config.StartToken}
	}

	// 3. Iteratively generate tokens
	for i := 0; i < maxLen; i++ {
		// a. Create target mask for the current sequence length
		tgtSeqLen := len(tgtInput[0])
		lookAheadMask := core.GenerateLookAheadMask(tgtSeqLen)
		// No padding in target yet, so padding mask is not strictly needed, but good practice
		tgtPaddingMask := core.Zeros(batchSize, 1, tgtSeqLen)
		tgtMask := core.CombineMasks(tgtPaddingMask, lookAheadMask)

		// b. Forward pass
		logits, _, _, _ := t.Forward(srcInput, tgtInput, srcMask, tgtMask)

		// c. Get logits for the last token of each sequence
		// logits shape: (batchSize * tgtSeqLen, vocabSize)
		// We need the logits for the last token of each sequence in the batch.
		lastTokenLogits := core.NewTensor(batchSize, t.Config.VocabSize)
		for b := 0; b < batchSize; b++ {
			// Index of the last token's logits for batch item 'b'
			logitIndex := (b*tgtSeqLen + tgtSeqLen) - 1
			// Slice the logits for the last token
			lastTokenLogits.SetRow(logits.Slice(logitIndex, logitIndex+1), b)
		}

		// d. Select the token with the highest probability (greedy decoding)
		// Assuming the tensor library has an Argmax function.
		nextTokens := lastTokenLogits.Argmax(1) // Argmax along dimension 1 (the vocab dimension)

		// e. Check if all sequences have generated the end token
		allFinished := true
		for _, token := range nextTokens.Data() {
			if int(token) != t.Config.EndToken {
				allFinished = false
				break
			}
		}
		if allFinished {
			break
		}

		// f. Append the new tokens to the target sequence for the next iteration
		newTgtInput := make([][]int, batchSize)
		for b := 0; b < batchSize; b++ {
			newTgtInput[b] = append(tgtInput[b], int(nextTokens.Data()[b]))
		}
		tgtInput = newTgtInput
	}

	return tgtInput
}

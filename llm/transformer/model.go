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

// Backward performs the backward pass for the TransformerModel.
func (t *TransformerModel) Backward(gradOutput *core.Tensor) (*core.Tensor, *core.Tensor) {
	// 1. Backward through Generator
	gradDecOutput := t.Generator.Backward(gradOutput)

	// 2. Backward through Decoder
	gradTgtEmb, gradEncOutputFromDecoder := t.Decoder.Backward(gradDecOutput)

	// 3. Backward through Encoder
	gradSrcEmb := t.Encoder.Backward(gradEncOutputFromDecoder) // Add zero matrix to match dimensions

	// 4. Backward through SrcEmbedding
	gradSrcInput := t.Encoder.SrcEmbedding.Backward(gradSrcEmb)

	// 5. Backward through TgtEmbedding
	gradTgtInput := t.Decoder.TgtEmbedding.Backward(gradTgtEmb)

	return gradSrcInput, gradTgtInput
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
	// Set model to evaluation mode
	t.SetTraining(false)

	// Initialize target sequence with start token
	tgtInput := make([][]int, len(srcInput))
	for i := range srcInput {
		tgtInput[i] = []int{t.Config.StartToken}
	}

	// Create dummy masks for generation (no actual masking needed for greedy decoding here)
	srcMask := core.Zeros(len(srcInput), 1, t.Config.MaxSeqLen)
	tgtMask := core.Zeros(len(srcInput), t.Config.MaxSeqLen, t.Config.MaxSeqLen)

	// Iteratively generate tokens
	for l := 0; l < maxLen; l++ {
		// Forward pass to get logits for the next token
		logits, _, _, _ := t.Forward(srcInput, tgtInput, srcMask, tgtMask)

		// Get the last token's logits
		lastLogits := logits.Slice(0, logits.Shape()[0]).Reshape(len(srcInput), -1, t.Config.VocabSize)

		// Select the token with the highest probability (greedy)
		nextTokens := make([]int, len(srcInput))
		for i := 0; i < len(srcInput); i++ {
			row := lastLogits.Slice(i, i+1).Reshape(t.Config.VocabSize)
			maxProb := -1.0
			maxIdx := -1
			for j, prob := range row.Data() {
				if prob > maxProb {
					maxProb = prob
					maxIdx = j
				}
			}
			nextTokens[i] = maxIdx
		}

		// Append the selected token to the target sequence
		for i := range tgtInput {
			tgtInput[i] = append(tgtInput[i], nextTokens[i])
		}

		// Check for end token
		isAllEnd := true
		for _, token := range nextTokens {
			if token != t.Config.EndToken {
				isAllEnd = false
				break
			}
		}
		if isAllEnd {
			break
		}
	}

	return tgtInput
}

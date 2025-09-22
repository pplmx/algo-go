package layer

import (
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
)

type TransformerDecoderLayer struct {
	Config           config.TransformerConfig
	SelfAttn         *MultiHeadAttention
	SelfAttnResidual *ResidualConnection
	EncDecAttn       *MultiHeadAttention
	EncDecResidual   *ResidualConnection
	FFN              *FeedForwardNetwork
	FFNResidual      *ResidualConnection

	lastX                *core.Tensor
	lastEncoderOutput    *core.Tensor
	lastSelfAttnOutput   *core.Tensor
	lastX1               *core.Tensor
	lastEncDecAttnOutput *core.Tensor
	lastX2               *core.Tensor
}

// NewTransformerDecoderLayer creates a new TransformerDecoderLayer.
func NewTransformerDecoderLayer(cfg config.TransformerConfig) *TransformerDecoderLayer {
	return &TransformerDecoderLayer{
		Config:           cfg,
		SelfAttn:         NewMultiHeadAttention(cfg),
		SelfAttnResidual: NewResidualConnection(cfg.DModel, cfg.DropoutRate),
		EncDecAttn:       NewMultiHeadAttention(cfg),
		EncDecResidual:   NewResidualConnection(cfg.DModel, cfg.DropoutRate),
		FFN:              NewFeedForwardNetwork(cfg.DModel, cfg.FFNHiddenSize, cfg.DropoutRate, cfg.UseBias),
		FFNResidual:      NewResidualConnection(cfg.DModel, cfg.DropoutRate),
	}
}

// SetTraining sets the training mode for the TransformerDecoderLayer.
func (d *TransformerDecoderLayer) SetTraining(training bool) {
	d.SelfAttn.SetTraining(training)
	d.SelfAttnResidual.SetTraining(training)
	d.EncDecAttn.SetTraining(training)
	d.EncDecResidual.SetTraining(training)
	d.FFN.SetTraining(training)
	d.FFNResidual.SetTraining(training)
}

// Forward performs the forward pass for the TransformerDecoderLayer.
func (d *TransformerDecoderLayer) Forward(x, encoderOutput, srcMask, tgtMask *core.Tensor) (*core.Tensor, []*core.Tensor, []*core.Tensor) {
	d.lastX = x
	d.lastEncoderOutput = encoderOutput

	// 1. 带掩码的多头自注意力 (Masked Multi-Head Self-Attention)
	selfAttnOutput, selfAttnWeights := d.SelfAttn.Forward(x, x, x, tgtMask)
	d.lastSelfAttnOutput = selfAttnOutput
	x1 := d.SelfAttnResidual.Forward(x, selfAttnOutput)
	d.lastX1 = x1

	// 2. 编码器-解码器注意力 (Encoder-Decoder Attention)
	encDecAttnOutput, encDecAttnWeights := d.EncDecAttn.Forward(x1, encoderOutput, encoderOutput, srcMask)
	d.lastEncDecAttnOutput = encDecAttnOutput
	d.lastX2 = d.EncDecResidual.Forward(x1, encDecAttnOutput)

	// 3. 前馈网络
	ffnOutput := d.FFN.Forward(d.lastX2)
	y := d.FFNResidual.Forward(d.lastX2, ffnOutput)

	return y, selfAttnWeights, encDecAttnWeights
}

// Backward performs the backward pass for the TransformerDecoderLayer.
func (d *TransformerDecoderLayer) Backward(gradOutput *core.Tensor) (*core.Tensor, *core.Tensor) {
	// This is a simplified placeholder. A real implementation would be more complex.
	gradX := core.Zeros(d.lastX.Shape()...)
	gradEncoderOutput := core.Zeros(d.lastEncoderOutput.Shape()...)
	return gradX, gradEncoderOutput
}

// GetParameters returns the trainable parameters of the TransformerDecoderLayer.
func (d *TransformerDecoderLayer) GetParameters() []*core.Tensor {
	params := []*core.Tensor{}
	params = append(params, d.SelfAttn.GetParameters()...)
	params = append(params, d.SelfAttnResidual.GetParameters()...)
	params = append(params, d.EncDecAttn.GetParameters()...)
	params = append(params, d.EncDecResidual.GetParameters()...)
	params = append(params, d.FFN.GetParameters()...)
	params = append(params, d.FFNResidual.GetParameters()...)
	return params
}

// GetGradients returns the gradients of the trainable parameters of the TransformerDecoderLayer.
func (d *TransformerDecoderLayer) GetGradients() []*core.Tensor {
	grads := []*core.Tensor{}
	grads = append(grads, d.SelfAttn.GetGradients()...)
	grads = append(grads, d.SelfAttnResidual.GetGradients()...)
	grads = append(grads, d.EncDecAttn.GetGradients()...)
	grads = append(grads, d.EncDecResidual.GetGradients()...)
	grads = append(grads, d.FFN.GetGradients()...)
	grads = append(grads, d.FFNResidual.GetGradients()...)
	return grads
}

// ZeroGradients sets the gradients of the trainable parameters to zero for the TransformerDecoderLayer.
func (d *TransformerDecoderLayer) ZeroGradients() {
	d.SelfAttn.ZeroGradients()
	d.SelfAttnResidual.ZeroGradients()
	d.EncDecAttn.ZeroGradients()
	d.EncDecResidual.ZeroGradients()
	d.FFN.ZeroGradients()
	d.FFNResidual.ZeroGradients()
}

// TransformerDecoder represents the entire Decoder module, stacked by N TransformerDecoderLayers.
type TransformerDecoder struct {
	Config       config.TransformerConfig
	TgtEmbedding *Embedding
	PosEnc       *PositionalEncoding
	Layers       []*TransformerDecoderLayer

	lastTgtEmb         *core.Tensor
	lastOutputs        []*core.Tensor // Stores output of each layer for backward pass
	lastEncoderOutputs []*core.Tensor // Stores encoderOutput for each layer for backward pass
}

// NewTransformerDecoder creates a new TransformerDecoder.
func NewTransformerDecoder(cfg config.TransformerConfig, numLayers int) *TransformerDecoder {
	layers := make([]*TransformerDecoderLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		layers[i] = NewTransformerDecoderLayer(cfg)
	}

	return &TransformerDecoder{
		Config:             cfg,
		TgtEmbedding:       NewEmbedding(cfg.VocabSize, cfg.DModel),
		PosEnc:             NewPositionalEncoding(cfg.MaxSeqLen, cfg.DModel),
		Layers:             layers,
		lastOutputs:        make([]*core.Tensor, numLayers),
		lastEncoderOutputs: make([]*core.Tensor, numLayers),
	}
}

// SetTraining sets the training mode for the TransformerDecoder.
func (t *TransformerDecoder) SetTraining(training bool) {
	t.TgtEmbedding.SetTraining(training)
	for _, layer := range t.Layers {
		layer.SetTraining(training)
	}
}

// Forward performs the forward pass for the TransformerDecoder.
func (t *TransformerDecoder) Forward(tgtInput [][]int, encoderOutput, srcMask, tgtMask *core.Tensor) (*core.Tensor, [][]*core.Tensor, [][]*core.Tensor) {
	// 目标语言嵌入
	tgtEmb := t.TgtEmbedding.Forward(tgtInput)
	t.lastTgtEmb = tgtEmb

	// 添加位置编码
	decoded := t.PosEnc.Forward(tgtEmb, len(tgtInput[0]))

	allSelfAttnWeights := make([][]*core.Tensor, len(t.Layers))
	allEncDecAttnWeights := make([][]*core.Tensor, len(t.Layers))
	output := decoded

	// 逐层处理
	for i, layer := range t.Layers {
		var selfAttnWeights, encDecAttnWeights []*core.Tensor
		output, selfAttnWeights, encDecAttnWeights = layer.Forward(output, encoderOutput, srcMask, tgtMask)
		allSelfAttnWeights[i] = selfAttnWeights
		allEncDecAttnWeights[i] = encDecAttnWeights
		t.lastOutputs[i] = output
		t.lastEncoderOutputs[i] = encoderOutput // Store encoderOutput for each layer
	}

	return output, allSelfAttnWeights, allEncDecAttnWeights
}

// Backward performs the backward pass for the TransformerDecoder.
func (t *TransformerDecoder) Backward(gradOutput *core.Tensor) (*core.Tensor, *core.Tensor) {
	// This is a simplified placeholder. A real implementation would be more complex.
	gradTgtInput := core.Zeros(t.lastTgtEmb.Shape()...)
	gradEncoderOutput := core.Zeros(t.lastEncoderOutputs[0].Shape()...)
	return gradTgtInput, gradEncoderOutput
}

// GetParameters returns the trainable parameters of the TransformerDecoder.
func (t *TransformerDecoder) GetParameters() []*core.Tensor {
	var params []*core.Tensor
	params = append(params, t.TgtEmbedding.GetParameters()...)
	// PosEnc has no trainable parameters
	for _, layer := range t.Layers {
		params = append(params, layer.GetParameters()...)
	}
	return params
}

// GetGradients returns the gradients of the trainable parameters of the TransformerDecoder.
func (t *TransformerDecoder) GetGradients() []*core.Tensor {
	var grads []*core.Tensor
	grads = append(grads, t.TgtEmbedding.GetGradients()...)
	for _, layer := range t.Layers {
		grads = append(grads, layer.GetGradients()...)
	}
	return grads
}

// ZeroGradients sets the gradients of the trainable parameters to zero for the TransformerDecoder.
func (t *TransformerDecoder) ZeroGradients() {
	t.TgtEmbedding.ZeroGradients()
	for _, layer := range t.Layers {
		layer.ZeroGradients()
	}
}

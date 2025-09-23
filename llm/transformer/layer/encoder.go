package layer

import (
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
)

// TransformerEncoderLayer 定义了 Encoder 的一个独立的层
// 它包含了多头自注意力模块和前馈网络模块。
type TransformerEncoderLayer struct {
	Config       config.TransformerConfig
	SelfAttn     *MultiHeadAttention
	AttnResidual *ResidualConnection
	FFN          *FeedForwardNetwork
	FFNResidual  *ResidualConnection

	lastX          *core.Tensor
	lastAttnOutput *core.Tensor
	lastX1         *core.Tensor
	lastFFNOutput  *core.Tensor
}

// NewTransformerEncoderLayer creates a new TransformerEncoderLayer.
func NewTransformerEncoderLayer(cfg config.TransformerConfig) *TransformerEncoderLayer {
	return &TransformerEncoderLayer{
		Config:       cfg,
		SelfAttn:     NewMultiHeadAttention(cfg),
		AttnResidual: NewResidualConnection(cfg.DModel, cfg.DropoutRate),
		FFN:          NewFeedForwardNetwork(cfg.DModel, cfg.FFNHiddenSize, cfg.DropoutRate, cfg.UseBias),
		FFNResidual:  NewResidualConnection(cfg.DModel, cfg.DropoutRate),
	}
}

// SetTraining sets the training mode for the TransformerEncoderLayer.
func (t *TransformerEncoderLayer) SetTraining(training bool) {
	t.SelfAttn.SetTraining(training)
	t.AttnResidual.SetTraining(training)
	t.FFN.SetTraining(training)
	t.FFNResidual.SetTraining(training)
}

// Forward performs the forward pass for the TransformerEncoderLayer.
func (t *TransformerEncoderLayer) Forward(x *core.Tensor, mask *core.Tensor) (*core.Tensor, []*core.Tensor) {
	t.lastX = x

	// 自注意力层
	attnOutput, attnWeights := t.SelfAttn.Forward(x, x, x, mask)
	t.lastAttnOutput = attnOutput

	// 残差连接和层归一化 (Add & Norm)
	x1 := t.AttnResidual.Forward(x, attnOutput)
	t.lastX1 = x1

	// 前馈网络
	ffnOutput := t.FFN.Forward(x1)
	t.lastFFNOutput = ffnOutput

	// 第二个残差连接和层归一化 (Add & Norm)
	y := t.FFNResidual.Forward(x1, ffnOutput)

	return y, attnWeights
}

// Backward performs the backward pass for the TransformerEncoderLayer.
// It computes the gradient of the loss with respect to the layer's input.
func (t *TransformerEncoderLayer) Backward(gradOutput *core.Tensor) *core.Tensor {
	// The forward pass is: y = FFNResidual(x1, FFN(x1)) where x1 = AttnResidual(x, SelfAttn(x))
	// We backpropagate through this structure in reverse.

	// 1. Backpropagate through the second residual connection (FFNResidual).
	// This gives us the gradient for x1 (from the residual path) and for the FFN output.
	gradX1_from_ffn, gradFFNOutput := t.FFNResidual.Backward(gradOutput)

	// 2. Backpropagate through the FeedForwardNetwork.
	// This gives us the gradient for x1 (from the FFN path).
	gradX1_from_ffn_layer := t.FFN.Backward(gradFFNOutput)

	// 3. Sum the gradients for x1 from its two usages (residual path and FFN path).
	gradX1 := gradX1_from_ffn.Add(gradX1_from_ffn_layer)

	// 4. Backpropagate through the first residual connection (AttnResidual).
	// This gives us the gradient for x (from the residual path) and for the Self-Attention output.
	gradX_from_attn, gradAttnOutput := t.AttnResidual.Backward(gradX1)

	// 5. Backpropagate through the MultiHeadAttention layer.
	// Since Q, K, and V inputs are all from x, we get three gradients which must be summed.
	gradX_from_q, gradX_from_k, gradX_from_v := t.SelfAttn.Backward(gradAttnOutput)

	// 6. Sum all gradients for x from its four usages (residual path, Q, K, and V).
	gradX := gradX_from_attn.Add(gradX_from_q).Add(gradX_from_k).Add(gradX_from_v)

	return gradX
}

// GetParameters returns the trainable parameters of the TransformerEncoderLayer.
func (t *TransformerEncoderLayer) GetParameters() []*core.Tensor {
	var params []*core.Tensor
	params = append(params, t.SelfAttn.GetParameters()...)
	params = append(params, t.AttnResidual.GetParameters()...)
	params = append(params, t.FFN.GetParameters()...)
	params = append(params, t.FFNResidual.GetParameters()...)
	return params
}

// GetGradients returns the gradients of the trainable parameters of the TransformerEncoderLayer.
func (t *TransformerEncoderLayer) GetGradients() []*core.Tensor {
	var grads []*core.Tensor
	grads = append(grads, t.SelfAttn.GetGradients()...)
	grads = append(grads, t.AttnResidual.GetGradients()...)
	grads = append(grads, t.FFN.GetGradients()...)
	grads = append(grads, t.FFNResidual.GetGradients()...)
	return grads
}

// ZeroGradients sets the gradients of the trainable parameters to zero for the TransformerEncoderLayer.
func (t *TransformerEncoderLayer) ZeroGradients() {
	t.SelfAttn.ZeroGradients()
	t.AttnResidual.ZeroGradients()
	t.FFN.ZeroGradients()
	t.FFNResidual.ZeroGradients()
}

// TransformerEncoder represents the entire Encoder module, stacked by N TransformerEncoderLayers.
type TransformerEncoder struct {
	Config       config.TransformerConfig
	SrcEmbedding *Embedding
	PosEnc       *PositionalEncoding
	Layers       []*TransformerEncoderLayer

	lastEncoded *core.Tensor
	lastOutputs []*core.Tensor // Stores output of each layer for backward pass
}

// NewTransformerEncoder creates a new TransformerEncoder.
func NewTransformerEncoder(cfg config.TransformerConfig, numLayers int) *TransformerEncoder {
	layers := make([]*TransformerEncoderLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		layers[i] = NewTransformerEncoderLayer(cfg)
	}

	return &TransformerEncoder{
		Config:       cfg,
		SrcEmbedding: NewEmbedding(cfg.VocabSize, cfg.DModel),
		PosEnc:       NewPositionalEncoding(cfg.MaxSeqLen, cfg.DModel),
		Layers:       layers,
		lastOutputs:  make([]*core.Tensor, numLayers), // Initialize lastOutputs
	}
}

// SetTraining sets the training mode for the TransformerEncoder.
func (t *TransformerEncoder) SetTraining(training bool) {
	t.SrcEmbedding.SetTraining(training)
	for _, layer := range t.Layers {
		layer.SetTraining(training)
	}
}

// Forward performs the forward pass for the TransformerEncoder.
func (t *TransformerEncoder) Forward(srcInput [][]int, mask *core.Tensor) (*core.Tensor, [][]*core.Tensor) {
	// 源语言嵌入
	srcEmb := t.SrcEmbedding.Forward(srcInput)

	// 添加位置编码
	encoded := t.PosEnc.Forward(srcEmb, len(srcInput[0]))
	t.lastEncoded = encoded

	allAttnWeights := make([][]*core.Tensor, len(t.Layers))
	output := encoded

	// 逐层处理
	for i, layer := range t.Layers {
		var attnWeights []*core.Tensor
		output, attnWeights = layer.Forward(output, mask)
		allAttnWeights[i] = attnWeights
		// Store output of each layer for backward pass
		t.lastOutputs[i] = output
	}

	return output, allAttnWeights
}

// Backward performs the backward pass for the TransformerEncoder.
func (t *TransformerEncoder) Backward(gradOutput *core.Tensor) *core.Tensor {
	currentGrad := gradOutput

	// 1. Backpropagate through the layers in reverse order
	for i := len(t.Layers) - 1; i >= 0; i-- {
		layer := t.Layers[i]
		currentGrad = layer.Backward(currentGrad)
	}

	// At this point, currentGrad is the gradient with respect to the output of PosEnc

	// 2. Backpropagate through PositionalEncoding
	gradEmb := t.PosEnc.Backward(currentGrad)

	// 3. Backpropagate through the source embedding layer.
	// This will calculate the gradients for the embedding weights.
	// The return value is typically nil as we don't backpropagate to the integer token IDs.
	t.SrcEmbedding.Backward(gradEmb)

	// We return the gradient w.r.t the embedding output, which can be used for debugging or other purposes.
	return gradEmb
}

// GetParameters returns the trainable parameters of the TransformerEncoder.
func (t *TransformerEncoder) GetParameters() []*core.Tensor {
	var params []*core.Tensor
	params = append(params, t.SrcEmbedding.GetParameters()...)
	// PosEnc has no trainable parameters
	for _, layer := range t.Layers {
		params = append(params, layer.GetParameters()...)
	}
	return params
}

// GetGradients returns the gradients of the trainable parameters of the TransformerEncoder.
func (t *TransformerEncoder) GetGradients() []*core.Tensor {
	var grads []*core.Tensor
	grads = append(grads, t.SrcEmbedding.GetGradients()...)
	for _, layer := range t.Layers {
		grads = append(grads, layer.GetGradients()...)
	}
	return grads
}

// ZeroGradients sets the gradients of the trainable parameters to zero for the TransformerEncoder.
func (t *TransformerEncoder) ZeroGradients() {
	t.SrcEmbedding.ZeroGradients()
	for _, layer := range t.Layers {
		layer.ZeroGradients()
	}
}

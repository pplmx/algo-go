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

	lastX          core.Matrix
	lastAttnOutput core.Matrix
	lastX1         core.Matrix
	lastFFNOutput  core.Matrix
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
func (t *TransformerEncoderLayer) Forward(x core.Matrix, mask core.Matrix) (core.Matrix, []core.Matrix) {
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
func (t *TransformerEncoderLayer) Backward(gradOutput core.Matrix) core.Matrix {
	// 1. Backward through FFNResidual
	gradX1_from_FFNResidual, _ := t.FFNResidual.Backward(gradOutput)

	// 2. Backward through FFN
	gradX1_from_FFN := t.FFN.Backward(gradX1_from_FFNResidual)

	// 3. Backward through AttnResidual
	gradX_from_AttnResidual, gradAttnOutput := t.AttnResidual.Backward(core.AddMatrices(gradX1_from_FFNResidual, gradX1_from_FFN))

	// 4. Backward through SelfAttn
	gradX_from_SelfAttn, _, _ := t.SelfAttn.Backward(gradAttnOutput)

	// Combine gradients for x
	gradX := core.AddMatrices(gradX_from_AttnResidual, gradX_from_SelfAttn)

	return gradX
}

// GetParameters returns the trainable parameters of the TransformerEncoderLayer.
func (t *TransformerEncoderLayer) GetParameters() []core.Matrix {
	params := []core.Matrix{}
	params = append(params, t.SelfAttn.GetParameters()...)
	params = append(params, t.AttnResidual.GetParameters()...)
	params = append(params, t.FFN.GetParameters()...)
	params = append(params, t.FFNResidual.GetParameters()...)
	return params
}

// GetGradients returns the gradients of the trainable parameters of the TransformerEncoderLayer.
func (t *TransformerEncoderLayer) GetGradients() []core.Matrix {
	grads := []core.Matrix{}
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
	Config config.TransformerConfig
	PosEnc *PositionalEncoding
	Layers []*TransformerEncoderLayer

	lastEncoded core.Matrix
	lastOutputs []core.Matrix // Stores output of each layer for backward pass
}

// NewTransformerEncoder creates a new TransformerEncoder.
func NewTransformerEncoder(cfg config.TransformerConfig, numLayers int) *TransformerEncoder {
	layers := make([]*TransformerEncoderLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		layers[i] = NewTransformerEncoderLayer(cfg)
	}

	return &TransformerEncoder{
		Config: cfg,
		PosEnc: NewPositionalEncoding(cfg.MaxSeqLen, cfg.DModel),
		Layers: layers,
		lastOutputs: make([]core.Matrix, numLayers), // Initialize lastOutputs
	}
}

// SetTraining sets the training mode for the TransformerEncoder.
func (t *TransformerEncoder) SetTraining(training bool) {
	for _, layer := range t.Layers {
		layer.SetTraining(training)
	}
}

// Forward performs the forward pass for the TransformerEncoder.
func (t *TransformerEncoder) Forward(x core.Matrix, mask core.Matrix) (core.Matrix, [][]core.Matrix) {
	// 添加位置编码
	encoded := t.PosEnc.Forward(x)
	t.lastEncoded = encoded

	allAttnWeights := make([][]core.Matrix, len(t.Layers))
	output := encoded

	// 逐层处理
	for i, layer := range t.Layers {
		var attnWeights []core.Matrix
		output, attnWeights = layer.Forward(output, mask)
		allAttnWeights[i] = attnWeights
		// Store output of each layer for backward pass
		t.lastOutputs[i] = output
	}

	return output, allAttnWeights
}

// Backward performs the backward pass for the TransformerEncoder.
func (t *TransformerEncoder) Backward(gradOutput core.Matrix) core.Matrix {
	// Initialize gradInput with zeros
	gradInput := core.Zeros(len(gradOutput), len(gradOutput[0]))

	// 1. Backward through layers in reverse order
	currentGrad := gradOutput
	for i := len(t.Layers) - 1; i >= 0; i-- {
		layer := t.Layers[i]
		// The input to the current layer's forward pass was t.lastOutputs[i-1] or t.lastEncoded
		// We need to pass the correct input to the layer's backward method if it needs it
		// For now, assuming layer.Backward only needs gradOutput
		currentGrad = layer.Backward(currentGrad)
	}

	// 2. Backward through PosEnc
	gradInput = t.PosEnc.Backward(currentGrad)

	return gradInput
}

// GetParameters returns the trainable parameters of the TransformerEncoder.
func (t *TransformerEncoder) GetParameters() []core.Matrix {
	params := []core.Matrix{}
	// PosEnc has no trainable parameters
	for _, layer := range t.Layers {
		params = append(params, layer.GetParameters()...)
	}
	return params
}

// GetGradients returns the gradients of the trainable parameters of the TransformerEncoder.
func (t *TransformerEncoder) GetGradients() []core.Matrix {
	grads := []core.Matrix{}
	for _, layer := range t.Layers {
		grads = append(grads, layer.GetGradients()...)
	}
	return grads
}

// ZeroGradients sets the gradients of the trainable parameters to zero for the TransformerEncoder.
func (t *TransformerEncoder) ZeroGradients() {
	for _, layer := range t.Layers {
		layer.ZeroGradients()
	}
}

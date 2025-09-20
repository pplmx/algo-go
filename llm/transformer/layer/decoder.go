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

	lastX                core.Matrix
	lastEncoderOutput    core.Matrix
	lastSelfAttnOutput   core.Matrix
	lastX1               core.Matrix
	lastEncDecAttnOutput core.Matrix
	lastX2               core.Matrix
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
func (d *TransformerDecoderLayer) Forward(x, encoderOutput, srcMask, tgtMask core.Matrix) (core.Matrix, []core.Matrix, []core.Matrix) {
	d.lastX = x
	d.lastEncoderOutput = encoderOutput

	// 1. 带掩码的多头自注意力 (Masked Multi-Head Self-Attention)
	//    Q, K, V 都来自解码器的上一层输出 x
	//    使用 tgtMask 来防止看到未来的 token
	selfAttnOutput, selfAttnWeights := d.SelfAttn.Forward(x, x, x, tgtMask)
	d.lastSelfAttnOutput = selfAttnOutput
	x1 := d.SelfAttnResidual.Forward(x, selfAttnOutput)
	d.lastX1 = x1

	// 2. 编码器-解码器注意力 (Encoder-Decoder Attention)
	//    Q 来自解码器的上一层输出 x1
	//    K, V 来自编码器的最终输出 encoderOutput
	//    使用 srcMask 来忽略源序列中的 padding token
	encDecAttnOutput, encDecAttnWeights := d.EncDecAttn.Forward(x1, encoderOutput, encoderOutput, srcMask)
	d.lastEncDecAttnOutput = encDecAttnOutput
	d.lastX2 = d.EncDecResidual.Forward(x1, encDecAttnOutput)

	// 3. 前馈网络
	ffnOutput := d.FFN.Forward(d.lastX2)
	y := d.FFNResidual.Forward(d.lastX2, ffnOutput)

	return y, selfAttnWeights, encDecAttnWeights
}

// Backward performs the backward pass for the TransformerDecoderLayer.
func (d *TransformerDecoderLayer) Backward(gradOutput core.Matrix) (gradX, gradEncoderOutput core.Matrix) {
	// 1. Backward through FFNResidual
	gradX2_from_FFNResidual, _ := d.FFNResidual.Backward(gradOutput)

	// 2. Backward through FFN
	gradX2_from_FFN := d.FFN.Backward(gradX2_from_FFNResidual)

	// 3. Backward through EncDecResidual
	gradX1_from_EncDecResidual, gradEncDecAttnOutput := d.EncDecResidual.Backward(core.AddMatrices(gradX2_from_FFNResidual, gradX2_from_FFN))

	// 4. Backward through EncDecAttn
	gradX1_from_EncDecAttn, gradEncoderOutput_from_EncDecAttn, _ := d.EncDecAttn.Backward(gradEncDecAttnOutput)

	// 5. Backward through SelfAttnResidual
	gradX_from_SelfAttnResidual, gradSelfAttnOutput := d.SelfAttnResidual.Backward(core.AddMatrices(gradX1_from_EncDecResidual, gradX1_from_EncDecAttn))

	// 6. Backward through SelfAttn
	gradX_from_SelfAttn, _, _ := d.SelfAttn.Backward(gradSelfAttnOutput)

	// Combine gradients for x
	gradX = core.AddMatrices(gradX_from_SelfAttnResidual, gradX_from_SelfAttn)

	// Combine gradients for encoderOutput
	gradEncoderOutput = gradEncoderOutput_from_EncDecAttn

	return gradX, gradEncoderOutput
}

// GetParameters returns the trainable parameters of the TransformerDecoderLayer.
func (d *TransformerDecoderLayer) GetParameters() []core.Matrix {
	params := []core.Matrix{}
	params = append(params, d.SelfAttn.GetParameters()...)
	params = append(params, d.SelfAttnResidual.GetParameters()...)
	params = append(params, d.EncDecAttn.GetParameters()...)
	params = append(params, d.EncDecResidual.GetParameters()...)
	params = append(params, d.FFN.GetParameters()...)
	params = append(params, d.FFNResidual.GetParameters()...)
	return params
}

// GetGradients returns the gradients of the trainable parameters of the TransformerDecoderLayer.
func (d *TransformerDecoderLayer) GetGradients() []core.Matrix {
	grads := []core.Matrix{}
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

	lastTgtEmb         core.Matrix
	lastOutputs        []core.Matrix // Stores output of each layer for backward pass
	lastEncoderOutputs []core.Matrix // Stores encoderOutput for each layer for backward pass
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
		lastOutputs:        make([]core.Matrix, numLayers),
		lastEncoderOutputs: make([]core.Matrix, numLayers),
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
func (t *TransformerDecoder) Forward(tgtInput [][]int, encoderOutput, srcMask, tgtMask core.Matrix) (core.Matrix, [][]core.Matrix, [][]core.Matrix) {
	// 目标语言嵌入
	tgtEmb := t.TgtEmbedding.Forward(tgtInput)
	t.lastTgtEmb = tgtEmb

	// 添加位置编码
	decoded := t.PosEnc.Forward(tgtEmb)

	allSelfAttnWeights := make([][]core.Matrix, len(t.Layers))
	allEncDecAttnWeights := make([][]core.Matrix, len(t.Layers))
	output := decoded

	// 逐层处理
	for i, layer := range t.Layers {
		var selfAttnWeights, encDecAttnWeights []core.Matrix
		output, selfAttnWeights, encDecAttnWeights = layer.Forward(output, encoderOutput, srcMask, tgtMask)
		allSelfAttnWeights[i] = selfAttnWeights
		allEncDecAttnWeights[i] = encDecAttnWeights
		t.lastOutputs[i] = output
		t.lastEncoderOutputs[i] = encoderOutput // Store encoderOutput for each layer
	}

	return output, allSelfAttnWeights, allEncDecAttnWeights
}

// Backward performs the backward pass for the TransformerDecoder.
func (t *TransformerDecoder) Backward(gradOutput core.Matrix) (gradTgtInput, gradEncoderOutput core.Matrix) {
	// Initialize gradients
	gradTgtInput = core.Zeros(len(t.lastTgtEmb), len(t.lastTgtEmb[0]))
	gradEncoderOutput = core.Zeros(len(t.lastEncoderOutputs[0]), len(t.lastEncoderOutputs[0][0]))

	currentGrad := gradOutput
	for i := len(t.Layers) - 1; i >= 0; i-- {
		layer := t.Layers[i]
		// Pass the correct encoderOutput for this layer's backward pass
		gradX, gradEncOut := layer.Backward(currentGrad)

		// Accumulate gradients for encoderOutput
		gradEncoderOutput = core.AddMatrices(gradEncoderOutput, gradEncOut)

		currentGrad = gradX
	}

	// Backward through PosEnc
	gradTgtEmb := t.PosEnc.Backward(currentGrad)

	// Backward through TgtEmbedding
	gradTgtInput = t.TgtEmbedding.Backward(gradTgtEmb)

	return gradTgtInput, gradEncoderOutput
}

// GetParameters returns the trainable parameters of the TransformerDecoder.
func (t *TransformerDecoder) GetParameters() []core.Matrix {
	params := []core.Matrix{}
	params = append(params, t.TgtEmbedding.GetParameters()...)
	// PosEnc has no trainable parameters
	for _, layer := range t.Layers {
		params = append(params, layer.GetParameters()...)
	}
	return params
}

// GetGradients returns the gradients of the trainable parameters of the TransformerDecoder.
func (t *TransformerDecoder) GetGradients() []core.Matrix {
	grads := []core.Matrix{}
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

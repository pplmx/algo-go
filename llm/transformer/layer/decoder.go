package layer

import (
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
)

// TransformerDecoderLayer 定义了 Decoder 的一个独立的层
type TransformerDecoderLayer struct {
	Config           config.TransformerConfig
	SelfAttn         *MultiHeadAttention
	SelfAttnResidual *ResidualConnection
	EncDecAttn       *MultiHeadAttention
	EncDecResidual   *ResidualConnection
	FFN              *FeedForwardNetwork
	FFNResidual      *ResidualConnection
}

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

func (d *TransformerDecoderLayer) SetTraining(training bool) {
	d.SelfAttn.SetTraining(training)
	d.SelfAttnResidual.SetTraining(training)
	d.EncDecAttn.SetTraining(training)
	d.EncDecResidual.SetTraining(training)
	d.FFN.SetTraining(training)
	d.FFNResidual.SetTraining(training)
}

func (d *TransformerDecoderLayer) Forward(x, encoderOutput, srcMask, tgtMask core.Matrix) (core.Matrix, []core.Matrix, []core.Matrix) {
	// 1. 带掩码的多头自注意力 (Masked Multi-Head Self-Attention)
	//    Q, K, V 都来自解码器的上一层输出 x
	//    使用 tgtMask 来防止看到未来的 token
	selfAttnOutput, selfAttnWeights := d.SelfAttn.Forward(x, x, x, tgtMask)
	x1 := d.SelfAttnResidual.Forward(x, selfAttnOutput)

	// 2. 编码器-解码器注意力 (Encoder-Decoder Attention)
	//    Q 来自解码器的上一层输出 x1
	//    K, V 来自编码器的最终输出 encoderOutput
	//    使用 srcMask 来忽略源序列中的 padding token
	encDecAttnOutput, encDecAttnWeights := d.EncDecAttn.Forward(x1, encoderOutput, encoderOutput, srcMask)
	x2 := d.EncDecResidual.Forward(x1, encDecAttnOutput)

	// 3. 前馈网络
	ffnOutput := d.FFN.Forward(x2)
	y := d.FFNResidual.Forward(x2, ffnOutput)

	return y, selfAttnWeights, encDecAttnWeights
}

// TransformerDecoder 代表了由 N 个 TransformerDecoderLayer 堆叠而成的整个 Decoder 模块
type TransformerDecoder struct {
	Config       config.TransformerConfig
	TgtEmbedding *Embedding
	PosEnc       *PositionalEncoding
	Layers       []*TransformerDecoderLayer
}

func NewTransformerDecoder(cfg config.TransformerConfig, numLayers int) *TransformerDecoder {
	layers := make([]*TransformerDecoderLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		layers[i] = NewTransformerDecoderLayer(cfg)
	}

	return &TransformerDecoder{
		Config:       cfg,
		TgtEmbedding: NewEmbedding(cfg.VocabSize, cfg.DModel),
		PosEnc:       NewPositionalEncoding(cfg.MaxSeqLen, cfg.DModel),
		Layers:       layers,
	}
}

func (t *TransformerDecoder) SetTraining(training bool) {
	t.TgtEmbedding.SetTraining(training)
	for _, layer := range t.Layers {
		layer.SetTraining(training)
	}
}

func (t *TransformerDecoder) Forward(tgtInput [][]int, encoderOutput, srcMask, tgtMask core.Matrix) (core.Matrix, [][]core.Matrix, [][]core.Matrix) {
	// 目标语言嵌入
	tgtEmb := t.TgtEmbedding.Forward(tgtInput)

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
	}

	return output, allSelfAttnWeights, allEncDecAttnWeights
}

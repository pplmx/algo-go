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
}

func NewTransformerEncoderLayer(cfg config.TransformerConfig) *TransformerEncoderLayer {
	return &TransformerEncoderLayer{
		Config:       cfg,
		SelfAttn:     NewMultiHeadAttention(cfg),
		AttnResidual: NewResidualConnection(cfg.DModel, cfg.DropoutRate),
		FFN:          NewFeedForwardNetwork(cfg.DModel, cfg.FFNHiddenSize, cfg.DropoutRate, cfg.UseBias),
		FFNResidual:  NewResidualConnection(cfg.DModel, cfg.DropoutRate),
	}
}

func (t *TransformerEncoderLayer) SetTraining(training bool) {
	t.SelfAttn.SetTraining(training)
	t.AttnResidual.SetTraining(training)
	t.FFN.SetTraining(training)
	t.FFNResidual.SetTraining(training)
}

func (t *TransformerEncoderLayer) Forward(x core.Matrix, mask core.Matrix) (core.Matrix, []core.Matrix) {
	// 自注意力层
	attnOutput, attnWeights := t.SelfAttn.Forward(x, mask)

	// 残差连接和层归一化 (Add & Norm)
	x1 := t.AttnResidual.Forward(x, attnOutput)

	// 前馈网络
	ffnOutput := t.FFN.Forward(x1)

	// 第二个残差连接和层归一化 (Add & Norm)
	y := t.FFNResidual.Forward(x1, ffnOutput)

	return y, attnWeights
}

// TransformerEncoder 代表了由 N 个 TransformerEncoderLayer 堆叠而成的整个 Encoder 模块
type TransformerEncoder struct {
	Config config.TransformerConfig
	PosEnc *PositionalEncoding
	Layers []*TransformerEncoderLayer
}

func NewTransformerEncoder(cfg config.TransformerConfig, numLayers int) *TransformerEncoder {
	layers := make([]*TransformerEncoderLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		layers[i] = NewTransformerEncoderLayer(cfg)
	}

	return &TransformerEncoder{
		Config: cfg,
		PosEnc: NewPositionalEncoding(cfg.MaxSeqLen, cfg.DModel),
		Layers: layers,
	}
}

func (t *TransformerEncoder) SetTraining(training bool) {
	for _, layer := range t.Layers {
		layer.SetTraining(training)
	}
}

func (t *TransformerEncoder) Forward(x core.Matrix, mask core.Matrix) (core.Matrix, [][]core.Matrix) {
	// 添加位置编码
	encoded := t.PosEnc.Forward(x)

	allAttnWeights := make([][]core.Matrix, len(t.Layers))
	output := encoded

	// 逐层处理
	for i, layer := range t.Layers {
		var attnWeights []core.Matrix
		output, attnWeights = layer.Forward(output, mask)
		allAttnWeights[i] = attnWeights
	}

	return output, allAttnWeights
}

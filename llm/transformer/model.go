package transformer

import (
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
	"github.com/pplmx/algo-go/llm/transformer/layer"
)

// 完整的 Transformer 模型
type TransformerModel struct {
	Config       config.TransformerConfig
	SrcEmbedding *layer.Embedding
	TgtEmbedding *layer.Embedding
	Encoder      *layer.TransformerEncoder
	Decoder      *layer.TransformerDecoder
	Generator    *layer.LinearLayer
}

func NewTransformerModel(cfg config.TransformerConfig) *TransformerModel {
	return &TransformerModel{
		Config:       cfg,
		SrcEmbedding: layer.NewEmbedding(cfg.VocabSize, cfg.DModel),
		TgtEmbedding: layer.NewEmbedding(cfg.VocabSize, cfg.DModel),
		Encoder:      layer.NewTransformerEncoder(cfg, 6),
		Decoder:      layer.NewTransformerDecoder(cfg, 6),
		Generator:    layer.NewLinearLayer(cfg.DModel, cfg.VocabSize, true),
	}
}

func (t *TransformerModel) SetTraining(training bool) {
	t.Encoder.SetTraining(training)
	t.Decoder.SetTraining(training)
}

func (t *TransformerModel) Forward(srcInput, tgtInput [][]int, srcMask, tgtMask core.Matrix) (core.Matrix, [][]core.Matrix, [][]core.Matrix, [][]core.Matrix) {
	// 源语言嵌入
	srcEmb := t.SrcEmbedding.Forward(srcInput)

	// 编码器前向传播
	encOutput, encAttnWeights := t.Encoder.Forward(srcEmb, srcMask)

	// 解码器前向传播
	decOutput, selfAttnWeights, encDecAttnWeights := t.Decoder.Forward(tgtInput, encOutput, srcMask, tgtMask)

	// 生成器输出
	logits := t.Generator.Forward(decOutput)

	return logits, encAttnWeights, selfAttnWeights, encDecAttnWeights
}

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
	Encoder      *layer.TransformerEncoder
	Generator    *layer.LinearLayer
}

func NewTransformerModel(cfg config.TransformerConfig) *TransformerModel {
	return &TransformerModel{
		Config:       cfg,
		SrcEmbedding: layer.NewEmbedding(cfg.VocabSize, cfg.DModel),
		Encoder:      layer.NewTransformerEncoder(cfg, 6),
		Generator:    layer.NewLinearLayer(cfg.DModel, cfg.VocabSize, true),
	}
}

func (t *TransformerModel) SetTraining(training bool) {
	t.Encoder.SetTraining(training)
}

func (t *TransformerModel) Forward(srcInput [][]int, mask core.Matrix) (core.Matrix, [][]core.Matrix) {
	// 源语言嵌入
	srcEmb := t.SrcEmbedding.Forward(srcInput)

	// 编码器前向传播
	encOutput, encAttnWeights := t.Encoder.Forward(srcEmb, mask)

	// 生成器输出
	logits := t.Generator.Forward(encOutput)

	return logits, encAttnWeights
}

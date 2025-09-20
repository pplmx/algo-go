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

	lastSrcEmb  core.Matrix
	lastEncOutput core.Matrix
	lastDecOutput core.Matrix
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
	t.lastSrcEmb = srcEmb

	// 编码器前向传播
	encOutput, encAttnWeights := t.Encoder.Forward(srcEmb, srcMask)
	t.lastEncOutput = encOutput

	// 解码器前向传播
	decOutput, selfAttnWeights, encDecAttnWeights := t.Decoder.Forward(tgtInput, encOutput, srcMask, tgtMask)
	t.lastDecOutput = decOutput

	// 生成器输出
	logits := t.Generator.Forward(decOutput)

	return logits, encAttnWeights, selfAttnWeights, encDecAttnWeights
}

func (t *TransformerModel) Backward(gradOutput core.Matrix) (gradSrcInput, gradTgtInput core.Matrix) {
	// 1. Backward through Generator
	gradDecOutput := t.Generator.Backward(gradOutput)

	// 2. Backward through Decoder
	gradTgtEmb, gradEncOutput_from_Decoder := t.Decoder.Backward(gradDecOutput)

	// 3. Backward through Encoder
	gradSrcEmb := t.Encoder.Backward(core.AddMatrices(gradEncOutput_from_Decoder, core.Zeros(len(t.lastEncOutput), len(t.lastEncOutput[0])))) // Add zero matrix to match dimensions

	// 4. Backward through SrcEmbedding
	gradSrcInput = t.SrcEmbedding.Backward(gradSrcEmb)

	// 5. Backward through TgtEmbedding
	gradTgtInput = t.TgtEmbedding.Backward(gradTgtEmb)

	return gradSrcInput, gradTgtInput
}

func (t *TransformerModel) GetParameters() []core.Matrix {
	params := []core.Matrix{}
	params = append(params, t.SrcEmbedding.GetParameters()...)
	params = append(params, t.TgtEmbedding.GetParameters()...)
	params = append(params, t.Encoder.GetParameters()...)
	params = append(params, t.Decoder.GetParameters()...)
	params = append(params, t.Generator.GetParameters()...)
	return params
}

func (t *TransformerModel) GetGradients() []core.Matrix {
	grads := []core.Matrix{}
	grads = append(grads, t.SrcEmbedding.GetGradients()...)
	grads = append(grads, t.TgtEmbedding.GetGradients()...)
	grads = append(grads, t.Encoder.GetGradients()...)
	grads = append(grads, t.Decoder.GetGradients()...)
	grads = append(grads, t.Generator.GetGradients()...)
	return grads
}

func (t *TransformerModel) ZeroGradients() {
	t.SrcEmbedding.ZeroGradients()
	t.TgtEmbedding.ZeroGradients()
	t.Encoder.ZeroGradients()
	t.Decoder.ZeroGradients()
	t.Generator.ZeroGradients()
}

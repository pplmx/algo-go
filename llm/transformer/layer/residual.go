package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 残差连接模块
type ResidualConnection struct {
	Norm    *core.LayerNorm
	Dropout *core.Dropout
}

func NewResidualConnection(dModel int, dropoutRate float64) *ResidualConnection {
	return &ResidualConnection{
		Norm:    core.NewLayerNorm(dModel, 1e-5),
		Dropout: core.NewDropout(dropoutRate, true),
	}
}

func (r *ResidualConnection) SetTraining(training bool) {
	r.Dropout.SetTraining(training)
}

func (r *ResidualConnection) Forward(x, sublayer core.Matrix) core.Matrix {
	// 应用子层并添加 dropout
	subOutput := r.Dropout.Forward(sublayer)

	// 残差连接和层归一化
	return r.Norm.Forward(core.AddMatrices(x, subOutput))
}

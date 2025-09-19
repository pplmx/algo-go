package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 线性变换层
type LinearLayer struct {
	Weight  core.Matrix
	Bias    core.Matrix
	UseBias bool
}

func NewLinearLayer(inFeatures, outFeatures int, useBias bool) *LinearLayer {
	pi := &core.ParameterInitializer{}
	layer := &LinearLayer{
		Weight:  pi.XavierUniform(inFeatures, outFeatures, 1.0),
		UseBias: useBias,
	}
	if useBias {
		layer.Bias = pi.Zeros(1, outFeatures)
	}
	return layer
}

func (l *LinearLayer) Forward(x core.Matrix) core.Matrix {
	result := core.MatMul(x, l.Weight)

	if l.UseBias {
		for i := range result {
			for j := range result[i] {
				result[i][j] += l.Bias[0][j]
			}
		}
	}

	return result
}

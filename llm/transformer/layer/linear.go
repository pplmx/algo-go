package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 线性变换层
type LinearLayer struct {
	Weight  core.Matrix
	Bias    core.Matrix
	UseBias bool

	lastInput core.Matrix // Store input for backward pass

	gradWeight core.Matrix
	gradBias   core.Matrix
}

func NewLinearLayer(inFeatures, outFeatures int, useBias bool) *LinearLayer {
	pi := &core.ParameterInitializer{}
	layer := &LinearLayer{
		Weight:  pi.XavierUniform(inFeatures, outFeatures, 1.0),
		UseBias: useBias,
	}
	if useBias {
		layer.Bias = core.Zeros(1, outFeatures)
	}
	return layer
}

func (l *LinearLayer) Forward(x core.Matrix) core.Matrix {
	l.lastInput = x // Store input
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

func (l *LinearLayer) Backward(gradOutput core.Matrix) core.Matrix {
	// Calculate gradient with respect to weights
	l.gradWeight = core.MatMul(core.Transpose(l.lastInput), gradOutput)

	// Calculate gradient with respect to bias
	if l.UseBias {
		l.gradBias = core.SumRows(gradOutput)
	}

	// Calculate gradient with respect to input
	gradInput := core.MatMul(gradOutput, core.Transpose(l.Weight))

	return gradInput
}

func (l *LinearLayer) GetParameters() []core.Matrix {
	if l.UseBias {
		return []core.Matrix{l.Weight, l.Bias}
	}
	return []core.Matrix{l.Weight}
}

func (l *LinearLayer) GetGradients() []core.Matrix {
	if l.UseBias {
		return []core.Matrix{l.gradWeight, l.gradBias}
	}
	return []core.Matrix{l.gradWeight}
}

func (l *LinearLayer) ZeroGradients() {
	if l.gradWeight != nil {
		l.gradWeight = core.Zeros(len(l.gradWeight), len(l.gradWeight[0]))
	}
	if l.UseBias && l.gradBias != nil {
		l.gradBias = core.Zeros(len(l.gradBias), len(l.gradBias[0]))
	}
}

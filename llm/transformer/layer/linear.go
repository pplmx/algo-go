package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 线性变换层
type LinearLayer struct {
	Weight  *core.Tensor
	Bias    *core.Tensor
	UseBias bool

	lastInput *core.Tensor // Store input for backward pass

	gradWeight *core.Tensor
	gradBias   *core.Tensor
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

func (l *LinearLayer) Forward(x *core.Tensor, args ...interface{}) *core.Tensor {
	l.lastInput = x // Store input
	result := x.Dot(l.Weight)

	if l.UseBias {
		result = result.Add(l.Bias) // Broadcasting will handle the addition
	}

	return result
}

func (l *LinearLayer) Backward(gradOutput *core.Tensor) *core.Tensor {
	// Calculate gradient with respect to weights
	l.gradWeight = l.lastInput.Transpose().Dot(gradOutput)

	// Calculate gradient with respect to bias
	if l.UseBias {
		gradB := gradOutput.Sum(0) // Sum along the batch dimension, result is (out_features)
		l.gradBias = gradB.Reshape(1, -1) // Reshape to (1, out_features) to match bias shape
	}

	// Calculate gradient with respect to input
	gradInput := gradOutput.Dot(l.Weight.Transpose())

	return gradInput
}

func (l *LinearLayer) GetParameters() []*core.Tensor {
	if l.UseBias {
		return []*core.Tensor{l.Weight, l.Bias}
	}
	return []*core.Tensor{l.Weight}
}

func (l *LinearLayer) GetGradients() []*core.Tensor {
	if l.UseBias {
		return []*core.Tensor{l.gradWeight, l.gradBias}
	}
	return []*core.Tensor{l.gradWeight}
}

func (l *LinearLayer) ZeroGradients() {
	if l.gradWeight != nil {
		l.gradWeight = core.Zeros(l.gradWeight.Shape()...)
	}
	if l.UseBias && l.gradBias != nil {
		l.gradBias = core.Zeros(l.gradBias.Shape()...)
	}
}

func (l *LinearLayer) SetTraining(training bool) {
	// No-op for linear layer
}

package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 前馈网络模块
type FeedForwardNetwork struct {
	fc1     *LinearLayer
	fc2     *LinearLayer
	dropout *core.Dropout

	lastH        core.Matrix // Output of fc1 before ReLU
	lastReLUOutput core.Matrix // Output of ReLU before dropout
}

// NewFeedForwardNetwork creates a new FeedForwardNetwork layer.
func NewFeedForwardNetwork(dModel, ffnHiddenSize int, dropoutRate float64, useBias bool) *FeedForwardNetwork {
	return &FeedForwardNetwork{
		fc1:     NewLinearLayer(dModel, ffnHiddenSize, useBias),
		fc2:     NewLinearLayer(ffnHiddenSize, dModel, useBias),
		dropout: core.NewDropout(dropoutRate, true),
	}
}

// SetTraining sets the training mode for the FeedForwardNetwork layer.
func (f *FeedForwardNetwork) SetTraining(training bool) {
	f.dropout.SetTraining(training)
}

// Forward performs the forward pass for the FeedForwardNetwork layer.
func (f *FeedForwardNetwork) Forward(x core.Matrix) core.Matrix {
	// 第一层: ReLU激活
	h := f.fc1.Forward(x)
	f.lastH = h // Store for backward pass

	reluOutput := make(core.Matrix, len(h))
	for i := range h {
		reluOutput[i] = make([]float64, len(h[i]))
		for j := range h[i] {
			if h[i][j] < 0 {
				reluOutput[i][j] = 0
			} else {
				reluOutput[i][j] = h[i][j]
			}
		}
	}
	f.lastReLUOutput = reluOutput // Store for backward pass

	// 应用 dropout
	dropped := f.dropout.Forward(reluOutput)

	// 第二层: 线性变换
	return f.fc2.Forward(dropped)
}

// Backward performs the backward pass for the FeedForwardNetwork layer.
func (f *FeedForwardNetwork) Backward(gradOutput core.Matrix) core.Matrix {
	// 1. Backward through fc2
	gradDropped := f.fc2.Backward(gradOutput)

	// 2. Backward through dropout
	gradReLUOutput := f.dropout.Backward(gradDropped)

	// 3. Backward through ReLU
	gradH := make(core.Matrix, len(gradReLUOutput))
	for i := range gradReLUOutput {
		gradH[i] = make([]float64, len(gradReLUOutput[i]))
		for j := range gradReLUOutput[i] {
			if f.lastReLUOutput[i][j] > 0 {
				gradH[i][j] = gradReLUOutput[i][j]
			} else {
				gradH[i][j] = 0
			}
		}
	}

	// 4. Backward through fc1
	gradInput := f.fc1.Backward(gradH)

	return gradInput
}

// GetParameters returns the trainable parameters of the FeedForwardNetwork layer.
func (f *FeedForwardNetwork) GetParameters() []core.Matrix {
	params := []core.Matrix{}
	params = append(params, f.fc1.GetParameters()...)
	params = append(params, f.fc2.GetParameters()...)
	// Dropout has no trainable parameters
	return params
}

// GetGradients returns the gradients of the trainable parameters of the FeedForwardNetwork layer.
func (f *FeedForwardNetwork) GetGradients() []core.Matrix {
	grads := []core.Matrix{}
	grads = append(grads, f.fc1.GetGradients()...)
	grads = append(grads, f.fc2.GetGradients()...)
	return grads
}

// ZeroGradients sets the gradients of the trainable parameters to zero.
func (f *FeedForwardNetwork) ZeroGradients() {
	f.fc1.ZeroGradients()
	f.fc2.ZeroGradients()
}

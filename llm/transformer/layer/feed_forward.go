package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 前馈网络模块
type FeedForwardNetwork struct {
	dModel        int
	ffnHiddenSize int
	fc1           *LinearLayer
	fc2           *LinearLayer
	dropout       *core.Dropout

	lastH          *core.Tensor // Output of fc1 before ReLU
	lastReLUOutput *core.Tensor // Output of ReLU before dropout
}

// NewFeedForwardNetwork creates a new FeedForwardNetwork layer.
func NewFeedForwardNetwork(dModel, ffnHiddenSize int, dropoutRate float64, useBias bool) *FeedForwardNetwork {
	return &FeedForwardNetwork{
		dModel:        dModel,
		ffnHiddenSize: ffnHiddenSize,
		fc1:           NewLinearLayer(dModel, ffnHiddenSize, useBias),
		fc2:           NewLinearLayer(ffnHiddenSize, dModel, useBias),
		dropout:       core.NewDropout(dropoutRate, true),
	}
}

// SetTraining sets the training mode for the FeedForwardNetwork layer.
func (f *FeedForwardNetwork) SetTraining(training bool) {
	f.dropout.SetTraining(training)
}

// Forward performs the forward pass for the FeedForwardNetwork layer.
func (f *FeedForwardNetwork) Forward(x *core.Tensor) *core.Tensor {
	batchSize := x.Shape()[0]
	seqLen := x.Shape()[1]

	// Reshape for linear layers
	reshapedX := x.Reshape(batchSize*seqLen, f.dModel)

	// First linear layer
	h := f.fc1.Forward(reshapedX)
	f.lastH = h // Save for backward pass

	// Activation
	reluOutput := h.ReLU()
	f.lastReLUOutput = reluOutput // Save for backward pass

	// Dropout
	droppedOutput := f.dropout.Forward(reluOutput)

	// Second linear layer
	output := f.fc2.Forward(droppedOutput)

	// Reshape back
	output = output.Reshape(batchSize, seqLen, f.dModel)

	return output
}

// Backward performs the backward pass for the FeedForwardNetwork layer.
func (f *FeedForwardNetwork) Backward(gradOutput *core.Tensor) *core.Tensor {
	batchSize := gradOutput.Shape()[0]
	seqLen := gradOutput.Shape()[1]

	// Reshape gradOutput to match the output of fc2
	reshapedGradOutput := gradOutput.Reshape(batchSize*seqLen, f.dModel)

	// 1. Backward through fc2
	gradDropped := f.fc2.Backward(reshapedGradOutput)

	// 2. Backward through dropout
	gradReLUOutput := f.dropout.Backward(gradDropped)

	// 3. Backward through ReLU
	gradH := gradReLUOutput.Clone()
	// The gradient of ReLU is 1 for input > 0, and 0 otherwise.
	// We apply this to the incoming gradient gradReLUOutput.
	for i, val := range f.lastH.Data() {
		if val <= 0 {
			gradH.Data()[i] = 0
		}
	}

	// 4. Backward through fc1
	gradInputReshaped := f.fc1.Backward(gradH)

	// 5. Reshape gradInput back to the original input shape
	gradInput := gradInputReshaped.Reshape(batchSize, seqLen, f.dModel)

	return gradInput
}

// GetParameters returns the trainable parameters of the FeedForwardNetwork layer.
func (f *FeedForwardNetwork) GetParameters() []*core.Tensor {
	var params []*core.Tensor
	params = append(params, f.fc1.GetParameters()...)
	params = append(params, f.fc2.GetParameters()...)
	// Dropout has no trainable parameters
	return params
}

// GetGradients returns the gradients of the trainable parameters of the FeedForwardNetwork layer.
func (f *FeedForwardNetwork) GetGradients() []*core.Tensor {
	var grads []*core.Tensor
	grads = append(grads, f.fc1.GetGradients()...)
	grads = append(grads, f.fc2.GetGradients()...)
	return grads
}

// ZeroGradients sets the gradients of the trainable parameters to zero.
func (f *FeedForwardNetwork) ZeroGradients() {
	f.fc1.ZeroGradients()
	f.fc2.ZeroGradients()
}

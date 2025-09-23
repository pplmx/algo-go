package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 残差连接模块
type ResidualConnection struct {
	Norm    *core.LayerNorm
	Dropout *core.Dropout

	lastX              *core.Tensor
	lastSublayerOutput *core.Tensor
	lastAddedOutput    *core.Tensor
}

// NewResidualConnection creates a new ResidualConnection layer.
func NewResidualConnection(dModel int, dropoutRate float64) *ResidualConnection {
	return &ResidualConnection{
		Norm:    core.NewLayerNorm(dModel, 1e-5),
		Dropout: core.NewDropout(dropoutRate, true),
	}
}

// SetTraining sets the training mode for the ResidualConnection layer.
func (r *ResidualConnection) SetTraining(training bool) {
	r.Dropout.SetTraining(training)
}

// Forward performs the forward pass for the ResidualConnection layer.
func (r *ResidualConnection) Forward(x, sublayer *core.Tensor) *core.Tensor {
	r.lastX = x
	r.lastSublayerOutput = sublayer

	// 应用子层并添加 dropout
	droppedSubOutput := r.Dropout.Forward(sublayer)

	// 残差连接和层归一化
	addedOutput := x.Add(droppedSubOutput)
	r.lastAddedOutput = addedOutput

	return r.Norm.Forward(addedOutput)
}

// Backward performs the backward pass for the ResidualConnection layer.
func (r *ResidualConnection) Backward(gradOutput *core.Tensor) (*core.Tensor, *core.Tensor) {
	// 1. Backward through LayerNorm
	gradAddedOutput := r.Norm.Backward(gradOutput)

	// 2. Backward through Add
	// The gradient is distributed to both inputs of the addition.
	// We clone to prevent accidental modification of the same tensor data.
	gradX := gradAddedOutput.Clone()
	gradDroppedSubOutput := gradAddedOutput.Clone()

	// 3. Backward through Dropout
	gradSublayer := r.Dropout.Backward(gradDroppedSubOutput)

	return gradX, gradSublayer
}

// GetParameters returns the trainable parameters of the ResidualConnection layer.
func (r *ResidualConnection) GetParameters() []*core.Tensor {
	var params []*core.Tensor
	params = append(params, r.Norm.GetParameters()...)
	// Dropout has no trainable parameters
	return params
}

// GetGradients returns the gradients of the trainable parameters of the ResidualConnection layer.
func (r *ResidualConnection) GetGradients() []*core.Tensor {
	var grads []*core.Tensor
	grads = append(grads, r.Norm.GetGradients()...)
	return grads
}

// ZeroGradients sets the gradients of the trainable parameters to zero.
func (r *ResidualConnection) ZeroGradients() {
	r.Norm.ZeroGradients()
}

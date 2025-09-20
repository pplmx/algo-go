package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 残差连接模块
type ResidualConnection struct {
	Norm    *core.LayerNorm
	Dropout *core.Dropout

	lastX              core.Matrix
	lastSublayerOutput core.Matrix
	lastAddedOutput    core.Matrix
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
func (r *ResidualConnection) Forward(x, sublayer core.Matrix) core.Matrix {
	r.lastX = x
	r.lastSublayerOutput = sublayer

	// 应用子层并添加 dropout
	droppedSubOutput := r.Dropout.Forward(sublayer)

	// 残差连接和层归一化
	addedOutput := core.AddMatrices(x, droppedSubOutput)
	r.lastAddedOutput = addedOutput

	return r.Norm.Forward(addedOutput)
}

// Backward performs the backward pass for the ResidualConnection layer.
func (r *ResidualConnection) Backward(gradOutput core.Matrix) (gradX, gradSublayer core.Matrix) {
	// 1. Backward through LayerNorm
	gradAddedOutput := r.Norm.Backward(gradOutput)

	// 2. Backward through AddMatrices
	// Gradients are simply passed through for addition
	gradX = gradAddedOutput
	gradDroppedSubOutput := gradAddedOutput

	// 3. Backward through Dropout
	gradSublayer = r.Dropout.Backward(gradDroppedSubOutput)

	return gradX, gradSublayer
}

// GetParameters returns the trainable parameters of the ResidualConnection layer.
func (r *ResidualConnection) GetParameters() []core.Matrix {
	params := []core.Matrix{}
	params = append(params, r.Norm.GetParameters()...)
	// Dropout has no trainable parameters
	return params
}

// GetGradients returns the gradients of the trainable parameters of the ResidualConnection layer.
func (r *ResidualConnection) GetGradients() []core.Matrix {
	grads := []core.Matrix{}
	grads = append(grads, r.Norm.GetGradients()...)
	return grads
}

// ZeroGradients sets the gradients of the trainable parameters to zero.
func (r *ResidualConnection) ZeroGradients() {
	r.Norm.ZeroGradients()
}

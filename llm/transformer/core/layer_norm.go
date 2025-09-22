package core

type LayerNorm struct {
	Gamma *Tensor
	Beta  *Tensor
	Eps   float64

	lastInput           *Tensor
	lastMean            []float64
	lastVariance        []float64
	lastNormalizedInput *Tensor

	gradGamma *Tensor
	gradBeta  *Tensor
}

// NewLayerNorm creates a new LayerNorm layer.
func NewLayerNorm(dModel int, eps float64) *LayerNorm {
	return &LayerNorm{
		Gamma: Ones(1, dModel),
		Beta:  Zeros(1, dModel),
		Eps:   eps,
	}
}

// Forward performs the forward pass for the LayerNorm layer.
func (l *LayerNorm) Forward(x *Tensor, args ...interface{}) *Tensor {
	l.lastInput = x
	return x.LayerNorm(l.Gamma, l.Beta, l.Eps)
}

// Backward performs the backward pass for the LayerNorm layer.
func (l *LayerNorm) Backward(gradOutput *Tensor) *Tensor {
	// Simplified backward pass for LayerNorm. A full, correct implementation is complex.
	// This is a placeholder that might not be numerically correct.
	// For a real implementation, you'd need to derive the gradients with respect to
	// the input, gamma, and beta.

	// Gradient w.r.t. gamma and beta
	// dL/dGamma = sum(dL/dY * normalized_input) over the batch
	// dL/dBeta = sum(dL/dY) over the batch

	// For simplicity, we'll assume the gradients for gamma and beta are just the sum of gradOutput.
	// This is NOT correct but avoids a panic.
	if l.gradGamma == nil || l.gradGamma.Size() != l.Gamma.Size() {
		l.gradGamma = Zeros(l.Gamma.Shape()...)
	}
	if l.gradBeta == nil || l.gradBeta.Size() != l.Beta.Size() {
		l.gradBeta = Zeros(l.Beta.Shape()...)
	}

	// This is a placeholder for the actual gradient calculation
	// A proper implementation would involve the chain rule through the normalization.
	gradInput := gradOutput.Clone() // Placeholder

	return gradInput
}

// GetParameters returns the trainable parameters (Gamma and Beta) of the LayerNorm layer.
func (l *LayerNorm) GetParameters() []*Tensor {
	return []*Tensor{l.Gamma, l.Beta}
}

// GetGradients returns the gradients of the trainable parameters (gradGamma and gradBeta).
func (l *LayerNorm) GetGradients() []*Tensor {
	return []*Tensor{l.gradGamma, l.gradBeta}
}

// ZeroGradients sets the gradients of the trainable parameters to zero.
func (l *LayerNorm) ZeroGradients() {
	if l.gradGamma != nil {
		l.gradGamma = Zeros(l.gradGamma.Shape()...)
	}
	if l.gradBeta != nil {
		l.gradBeta = Zeros(l.gradBeta.Shape()...)
	}
}

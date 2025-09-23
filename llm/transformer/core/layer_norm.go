package core

type LayerNorm struct {
	Gamma *Tensor
	Beta  *Tensor
	Eps   float64

	lastInput           *Tensor
	lastMean            []float64
	lastVariance        *Tensor
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

	// Calculate mean and variance along the last dimension
	mean := x.Mean(x.Ndim() - 1)
	variance := x.Variance(x.Ndim() - 1)

	// Keep for backward pass
	l.lastMean = mean.Data()
	l.lastVariance = variance

	// Normalize
	mean_expanded := mean.ExpandDims(mean.Ndim())
	variance_expanded := variance.AddScalar(l.Eps).Sqrt().ExpandDims(variance.Ndim())
	normalized := x.Sub(mean_expanded).Div(variance_expanded)
	l.lastNormalizedInput = normalized

	// Scale and shift
	output := normalized.Mul(l.Gamma.ExpandAs(x.Shape()...)).Add(l.Beta.ExpandAs(x.Shape()...))

	return output
}

// Backward performs the backward pass for the LayerNorm layer.
// It computes the gradients with respect to the input x, and the learnable parameters gamma and beta.
// The implementation is based on the formulas derived from the chain rule, as detailed in various online resources.
// See: https://robotchinwag.com/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/
func (l *LayerNorm) Backward(gradOutput *Tensor) *Tensor {
	hiddenSize := l.lastInput.Shape()[1]
	H := float64(hiddenSize)

	// Restore intermediate values from the forward pass
	normalizedInput := l.lastNormalizedInput
	sigma := l.lastVariance.AddScalar(l.Eps).Sqrt()                 // Shape (B, S)
	sigma_expanded := sigma.ExpandDims(sigma.Ndim())                 // Shape (B, S, 1)
	mu := normalizedInput.Mul(sigma_expanded)                        // Shape (B, S, D)

	// 1. Calculate gradients for the learnable parameters gamma and beta.
	l.gradGamma = gradOutput.Mul(normalizedInput).Sum(0)
	l.gradBeta = gradOutput.Sum(0)

	// 2. Calculate the gradient for the input x (dL/dx).
	gamma_expanded := l.Gamma.ExpandAs(gradOutput.Shape()...)

	// Term 1: The direct path from the output through the scaling by gamma and sigma.
	term1 := gradOutput.Mul(gamma_expanded).Div(sigma_expanded)

	// Term 2: The path through the mean.
	term2_sum := term1.Sum(1) // Shape (B, D)
	term2_sum_expanded := term2_sum.ExpandDims(1) // Shape (B, 1, D)
	term2 := term2_sum_expanded.ExpandAs(term1.Shape()...).MulScalar(-1.0 / H)

	// Term 3: The path through the variance.
	sigma3_expanded := sigma_expanded.Power(3)
	term3_sum_inner := gradOutput.Mul(gamma_expanded).Mul(mu).Div(sigma3_expanded)
	term3_sum := term3_sum_inner.Sum(1) // Shape (B, D)
	term3_sum_expanded := term3_sum.ExpandDims(1) // Shape (B, 1, D)
	term3 := mu.Mul(term3_sum_expanded.ExpandAs(mu.Shape()...)).MulScalar(-1.0 / H)

	// The final gradient is the sum of these three terms.
	gradInput := term1.Add(term2).Add(term3)

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

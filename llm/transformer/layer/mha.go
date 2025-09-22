package layer

import (
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
)

type MultiHeadAttention struct {
	Config    config.TransformerConfig
	WQ        *LinearLayer
	WK        *LinearLayer
	WV        *LinearLayer
	WO        *LinearLayer
	Attention *core.ScaledDotProductAttention

	lastQ            *core.Tensor
	lastK            *core.Tensor
	lastV            *core.Tensor
	lastQHeads       []*core.Tensor
	lastKHeads       []*core.Tensor
	lastVHeads       []*core.Tensor
	lastOutputs      []*core.Tensor
	lastConcatOutput *core.Tensor
}

// NewMultiHeadAttention creates a new MultiHeadAttention module.
func NewMultiHeadAttention(cfg config.TransformerConfig) *MultiHeadAttention {
	dKPerHead := cfg.DK / cfg.NumHeads
	dVPerHead := cfg.DV / cfg.NumHeads

	return &MultiHeadAttention{
		Config:    cfg,
		WQ:        NewLinearLayer(cfg.DModel, dKPerHead*cfg.NumHeads, cfg.UseBias),
		WK:        NewLinearLayer(cfg.DModel, dKPerHead*cfg.NumHeads, cfg.UseBias),
		WV:        NewLinearLayer(cfg.DModel, dVPerHead*cfg.NumHeads, cfg.UseBias),
		WO:        NewLinearLayer(dVPerHead*cfg.NumHeads, cfg.DModel, cfg.UseBias),
		Attention: core.NewScaledDotProductAttention(dKPerHead, cfg.DropoutRate),
	}
}

// SetTraining sets the training mode for the MultiHeadAttention module.
func (m *MultiHeadAttention) SetTraining(training bool) {
	m.Attention.SetTraining(training)
}

// Forward performs the forward pass for the MultiHeadAttention module.
func (m *MultiHeadAttention) Forward(qInput, kInput, vInput *core.Tensor, mask *core.Tensor) (*core.Tensor, []*core.Tensor) {
	batchSize := qInput.Shape()[0]
	seqLen := qInput.Shape()[1]

	// Reshape input for linear layers
	reshapedQInput := qInput.Reshape(batchSize*seqLen, m.Config.DModel)
	reshapedKInput := kInput.Reshape(batchSize*seqLen, m.Config.DModel)
	reshapedVInput := vInput.Reshape(batchSize*seqLen, m.Config.DModel)

	// Apply linear transformations
	linearQ := m.WQ.Forward(reshapedQInput)
	linearK := m.WK.Forward(reshapedKInput)
	linearV := m.WV.Forward(reshapedVInput)

	// Reshape back to (batch_size, seq_len, d_model)
	m.lastQ = linearQ.Reshape(batchSize, seqLen, m.Config.DK)
	m.lastK = linearK.Reshape(batchSize, seqLen, m.Config.DK)
	m.lastV = linearV.Reshape(batchSize, seqLen, m.Config.DV)

	// 分割多头
	qHeads := m.splitHeads(m.lastQ)
	kHeads := m.splitHeads(m.lastK)
	vHeads := m.splitHeads(m.lastV)

	// 计算每个头的注意力
	outputs := make([]*core.Tensor, m.Config.NumHeads)
	weights := make([]*core.Tensor, m.Config.NumHeads)

	var headMask *core.Tensor
	// Reshape mask to (batch_size * num_heads, seq_len, seq_len) for easier slicing
	if mask != nil {
		mask = mask.Reshape(batchSize*m.Config.NumHeads, seqLen, seqLen)
	}

	for i := 0; i < m.Config.NumHeads; i++ {
		if mask != nil {
			// Slice the mask for the current head
			start := i * batchSize
			end := (i + 1) * batchSize
			headMask = mask.Slice(start, end)
		} else {
			headMask = nil
		}
		outputs[i], weights[i] = m.Attention.Forward(qHeads[i], kHeads[i], vHeads[i], headMask)
	}
	m.lastOutputs = outputs

	// 合并多头输出
	concatOutput := m.concatHeads(outputs)
	m.lastConcatOutput = concatOutput

	// 最终线性变换
	reshapedConcatOutput := concatOutput.Reshape(batchSize*seqLen, m.Config.DV)
	output := m.WO.Forward(reshapedConcatOutput)
	output = output.Reshape(batchSize, seqLen, m.Config.DModel)

	return output, weights
}

// splitHeads splits the input tensor into multiple heads for multi-head attention.
func (m *MultiHeadAttention) splitHeads(x *core.Tensor) []*core.Tensor {
	batchSize := x.Shape()[0]
	seqLen := x.Shape()[1]
	dModel := x.Shape()[2]
	dHead := dModel / m.Config.NumHeads

	// Reshape to (batch_size, seq_len, num_heads, d_head)
	x = x.Reshape(batchSize, seqLen, m.Config.NumHeads, dHead)

	// Transpose to (num_heads, batch_size, seq_len, d_head)
	x = x.Transpose(2, 0, 1, 3)

	heads := make([]*core.Tensor, m.Config.NumHeads)
	for i := 0; i < m.Config.NumHeads; i++ {
		// Extract each head. This creates a view.
		heads[i] = x.Slice(i, i+1).Reshape(batchSize, seqLen, dHead)
	}

	return heads
}

// concatHeads concatenates the outputs from multiple attention heads.
func (m *MultiHeadAttention) concatHeads(heads []*core.Tensor) *core.Tensor {
	// This is a simplified placeholder. A real implementation would be more efficient.
	batchSize := heads[0].Shape()[0]
	seqLen := heads[0].Shape()[1]
	dv := heads[0].Shape()[2] * m.Config.NumHeads
	return core.Zeros(batchSize, seqLen, dv)
}

// Backward performs the backward pass for the MultiHeadAttention module.
func (m *MultiHeadAttention) Backward(gradOutput *core.Tensor) (*core.Tensor, *core.Tensor, *core.Tensor) {
	// This is a simplified placeholder. A real implementation would be more complex.
	gradQInput := m.WQ.Backward(core.Zeros(m.lastQ.Shape()...))
	gradKInput := m.WK.Backward(core.Zeros(m.lastK.Shape()...))
	gradVInput := m.WV.Backward(core.Zeros(m.lastV.Shape()...))
	return gradQInput, gradKInput, gradVInput
}

// GetParameters returns the trainable parameters of the MultiHeadAttention module.
func (m *MultiHeadAttention) GetParameters() []*core.Tensor {
	var params []*core.Tensor
	params = append(params, m.WQ.GetParameters()...)
	params = append(params, m.WK.GetParameters()...)
	params = append(params, m.WV.GetParameters()...)
	params = append(params, m.WO.GetParameters()...)
	// ScaledDotProductAttention has no trainable parameters
	return params
}

// GetGradients returns the gradients of the trainable parameters of the MultiHeadAttention module.
func (m *MultiHeadAttention) GetGradients() []*core.Tensor {
	var grads []*core.Tensor
	grads = append(grads, m.WQ.GetGradients()...)
	grads = append(grads, m.WK.GetGradients()...)
	grads = append(grads, m.WV.GetGradients()...)
	grads = append(grads, m.WO.GetGradients()...)
	return grads
}

// ZeroGradients sets the gradients of the trainable parameters to zero.
func (m *MultiHeadAttention) ZeroGradients() {
	m.WQ.ZeroGradients()
	m.WK.ZeroGradients()
	m.WV.ZeroGradients()
	m.WO.ZeroGradients()
}

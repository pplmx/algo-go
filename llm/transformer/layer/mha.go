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

	lastQ            core.Matrix
	lastK            core.Matrix
	lastV            core.Matrix
	lastQHeads       []core.Matrix
	lastKHeads       []core.Matrix
	lastVHeads       []core.Matrix
	lastOutputs      []core.Matrix
	lastConcatOutput core.Matrix
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
func (m *MultiHeadAttention) Forward(qInput, kInput, vInput core.Matrix, mask core.Matrix) (core.Matrix, []core.Matrix) {
	m.lastQ = m.WQ.Forward(qInput)
	m.lastK = m.WK.Forward(kInput)
	m.lastV = m.WV.Forward(vInput)

	// 分割多头
	qHeads := m.splitHeads(m.lastQ)
	kHeads := m.splitHeads(m.lastK)
	vHeads := m.splitHeads(m.lastV)

	// 计算每个头的注意力
	outputs := make([]core.Matrix, m.Config.NumHeads)
	weights := make([]core.Matrix, m.Config.NumHeads)

	for i := 0; i < m.Config.NumHeads; i++ {
		outputs[i], weights[i] = m.Attention.Forward(qHeads[i], kHeads[i], vHeads[i], mask)
	}
	m.lastOutputs = outputs

	// 合并多头输出
	concatOutput := m.concatHeads(outputs)
	m.lastConcatOutput = concatOutput

	// 最终线性变换
	output := m.WO.Forward(concatOutput)

	return output, weights
}

// splitHeads splits the input matrix into multiple heads for multi-head attention.
func (m *MultiHeadAttention) splitHeads(x core.Matrix) []core.Matrix {
	seqLen := len(x)
	dHead := m.Config.DK / m.Config.NumHeads
	heads := make([]core.Matrix, m.Config.NumHeads)

	for i := 0; i < m.Config.NumHeads; i++ {
		startCol := i * dHead
		endCol := startCol + dHead
		heads[i] = make(core.Matrix, seqLen)
		for j := 0; j < seqLen; j++ {
			heads[i][j] = x[j][startCol:endCol]
		}
	}

	return heads
}

// concatHeads concatenates the outputs from multiple attention heads.
func (m *MultiHeadAttention) concatHeads(heads []core.Matrix) core.Matrix {
	seqLen := len(heads[0])
	concat := make(core.Matrix, seqLen)

	for i := 0; i < seqLen; i++ {
		concat[i] = make([]float64, m.Config.DV)
		idx := 0
		for j := 0; j < m.Config.NumHeads; j++ {
			for k := 0; k < len(heads[j][i]); k++ {
				concat[i][idx] = heads[j][i][k]
				idx++
			}
		}
	}

	return concat
}

// Backward performs the backward pass for the MultiHeadAttention module.
func (m *MultiHeadAttention) Backward(gradOutput core.Matrix) (gradQInput, gradKInput, gradVInput core.Matrix) {
	// 1. Backward through WO
	gradConcatOutput := m.WO.Backward(gradOutput)

	// 2. Backward through concatHeads
	gradOutputs := m.unconcatHeads(gradConcatOutput)

	// 3. Backward through Attention for each head
	gradQHeads := make([]core.Matrix, m.Config.NumHeads)
	gradKHeads := make([]core.Matrix, m.Config.NumHeads)
	gradVHeads := make([]core.Matrix, m.Config.NumHeads)

	for i := 0; i < m.Config.NumHeads; i++ {
		// Pass the gradOutputs[i] to the Attention.Backward
		// The Attention.Backward returns gradQuery, gradKey, gradValue for that head
		gradQHeads[i], gradKHeads[i], gradVHeads[i] = m.Attention.Backward(gradOutputs[i])
	}

	// 4. Backward through splitHeads
	gradQ := m.unsplitHeads(gradQHeads, len(m.lastQ[0]))
	gradK := m.unsplitHeads(gradKHeads, len(m.lastK[0]))
	gradV := m.unsplitHeads(gradVHeads, len(m.lastV[0]))

	// 5. Backward through WQ, WK, WV
	gradQInput = m.WQ.Backward(gradQ)
	gradKInput = m.WK.Backward(gradK)
	gradVInput = m.WV.Backward(gradV)

	return gradQInput, gradKInput, gradVInput
}

// GetParameters returns the trainable parameters of the MultiHeadAttention module.
func (m *MultiHeadAttention) GetParameters() []core.Matrix {
	params := []core.Matrix{}
	params = append(params, m.WQ.GetParameters()...)
	params = append(params, m.WK.GetParameters()...)
	params = append(params, m.WV.GetParameters()...)
	params = append(params, m.WO.GetParameters()...)
	// ScaledDotProductAttention has no trainable parameters
	return params
}

// GetGradients returns the gradients of the trainable parameters of the MultiHeadAttention module.
func (m *MultiHeadAttention) GetGradients() []core.Matrix {
	grads := []core.Matrix{}
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

// unconcatHeads performs the reverse operation of concatHeads.
func (m *MultiHeadAttention) unconcatHeads(gradConcatOutput core.Matrix) []core.Matrix {
	seqLen := len(gradConcatOutput)
	dVPerHead := m.Config.DV / m.Config.NumHeads
	gradOutputs := make([]core.Matrix, m.Config.NumHeads)

	for i := 0; i < m.Config.NumHeads; i++ {
		gradOutputs[i] = make(core.Matrix, seqLen)
		for r := 0; r < seqLen; r++ {
			gradOutputs[i][r] = make([]float64, dVPerHead)
			copy(gradOutputs[i][r], gradConcatOutput[r][i*dVPerHead:(i+1)*dVPerHead])
		}
	}
	return gradOutputs
}

// unsplitHeads performs the reverse operation of splitHeads.
func (m *MultiHeadAttention) unsplitHeads(gradHeads []core.Matrix, originalDim int) core.Matrix {
	seqLen := len(gradHeads[0])
	dHead := len(gradHeads[0][0]) // dKPerHead or dVPerHead
	gradInput := make(core.Matrix, seqLen)
	for r := 0; r < seqLen; r++ {
		gradInput[r] = make([]float64, originalDim)
	}

	for i := 0; i < m.Config.NumHeads; i++ {
		for r := 0; r < seqLen; r++ {
			startCol := i * dHead
			for c := 0; c < dHead; c++ {
				gradInput[r][startCol+c] += gradHeads[i][r][c]
			}
		}
	}
	return gradInput
}

package layer

import (
	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
)

// 多头注意力模块
type MultiHeadAttention struct {
	Config    config.TransformerConfig
	WQ        *LinearLayer
	WK        *LinearLayer
	WV        *LinearLayer
	WO        *LinearLayer
	Attention *core.ScaledDotProductAttention
}

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

func (m *MultiHeadAttention) SetTraining(training bool) {
	m.Attention.SetTraining(training)
}

func (m *MultiHeadAttention) Forward(x core.Matrix, mask core.Matrix) (core.Matrix, []core.Matrix) {
	// 线性变换得到 Q, K, V
	q := m.WQ.Forward(x)
	k := m.WK.Forward(x)
	v := m.WV.Forward(x)

	// 分割多头
	qHeads := m.splitHeads(q)
	kHeads := m.splitHeads(k)
	vHeads := m.splitHeads(v)

	// 计算每个头的注意力
	outputs := make([]core.Matrix, m.Config.NumHeads)
	weights := make([]core.Matrix, m.Config.NumHeads)

	for i := 0; i < m.Config.NumHeads; i++ {
		outputs[i], weights[i] = m.Attention.Forward(qHeads[i], kHeads[i], vHeads[i], mask)
	}

	// 合并多头输出
	concatOutput := m.concatHeads(outputs)

	// 最终线性变换
	output := m.WO.Forward(concatOutput)

	return output, weights
}

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

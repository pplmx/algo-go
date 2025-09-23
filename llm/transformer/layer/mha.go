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
	lastAttnWeights  []*core.Tensor
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
	m.lastQHeads, m.lastKHeads, m.lastVHeads = qHeads, kHeads, vHeads // Save for backward pass

	// 计算每个头的注意力
	outputs := make([]*core.Tensor, m.Config.NumHeads)
	weights := make([]*core.Tensor, m.Config.NumHeads)

	// --- 掩码处理 ---
	// MHA的注意力分数形状为 (B, H, S, S).
	// 输入的掩码 `mask` 有多种可能形状，这里统一处理成 (B, H, S, S).
	if mask != nil {
		numHeads := m.Config.NumHeads
		qSeqLen := qInput.Shape()[1] // Q的序列长度决定了掩码的尺寸

		// Case 1: 4D padding mask (B, 1, 1, S_k) from encoder
		if len(mask.Shape()) == 4 && mask.Shape()[1] == 1 && mask.Shape()[2] == 1 {
			kSeqLen := mask.Shape()[3]
			tiledMask := core.NewTensor(batchSize, numHeads, qSeqLen, kSeqLen)
			for b := 0; b < batchSize; b++ {
				for h := 0; h < numHeads; h++ {
					for s1 := 0; s1 < qSeqLen; s1++ {
						for s2 := 0; s2 < kSeqLen; s2++ {
							padVal := mask.Get(b, 0, 0, s2)
							if padVal != 0 {
								tiledMask.Set(padVal, b, h, s1, s2)
							}
						}
					}
				}
			}
			mask = tiledMask
		} else if len(mask.Shape()) == 4 && mask.Shape()[1] == 1 {
			// Case 2: 4D combined mask (B, 1, S_q, S_k) from decoder self-attention
			qSeqLen := mask.Shape()[2]
			kSeqLen := mask.Shape()[3]
			tiledMask := core.NewTensor(batchSize, numHeads, qSeqLen, kSeqLen)
			for b := 0; b < batchSize; b++ {
				for h := 0; h < numHeads; h++ {
					for s1 := 0; s1 < qSeqLen; s1++ {
						for s2 := 0; s2 < kSeqLen; s2++ {
							val := mask.Get(b, 0, s1, s2)
							if val != 0 {
								tiledMask.Set(val, b, h, s1, s2)
							}
						}
					}
				}
			}
			mask = tiledMask
		}
	}

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
	m.lastAttnWeights = weights // Save attention weights for backward pass

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

// concatHeads 合并多个注意力头的输出。
// 输入 `heads` 是一个张量切片，其中每个张量的形状为 (batchSize, seqLen, dHead)。
// 该函数将这些头沿最后一个维度拼接起来，生成一个形状为 (batchSize, seqLen, numHeads * dHead) 的张量。
//
// 由于核心张量库缺少高效的 `stack` 或 `concat` 操作，此实现通过手动计算步长和复制数据来完成。
// 这种方法功能上是正确的，但在性能上不是最优的，因为它涉及大量的数据复制。
func (m *MultiHeadAttention) concatHeads(heads []*core.Tensor) *core.Tensor {
	batchSize := heads[0].Shape()[0]
	seqLen := heads[0].Shape()[1]
	dHead := heads[0].Shape()[2]
	numHeads := m.Config.NumHeads
	totalDV := numHeads * dHead

	// 最终拼接后的数据切片
	finalData := make([]float64, batchSize*seqLen*totalDV)

	// 计算最终张量 (batch, seq, total_dv) 的步长
	stride0 := seqLen * totalDV
	stride1 := totalDV

	// 遍历每个头，将其数据复制到 finalData 的正确位置
	for h, head := range heads {
		headData := head.Data()
		headStride0 := head.Strides()[0] // head 的第0维步长 (seq * d_head)
		headStride1 := head.Strides()[1] // head 的第1维步长 (d_head)

		for i := 0; i < batchSize; i++ {
			for j := 0; j < seqLen; j++ {
				// 计算源数据（单个头）中的偏移量
				srcOffset := i*headStride0 + j*headStride1
				// 计算目标数据（拼接后）中的偏移量
				destOffset := i*stride0 + j*stride1 + h*dHead
				// 复制数据
				copy(finalData[destOffset:destOffset+dHead], headData[srcOffset:srcOffset+dHead])
			}
		}
	}

	return core.NewTensorFromData(finalData, batchSize, seqLen, totalDV)
}

// splitGradsToHeads splits the gradient tensor back into gradients for each head.
// It's the reverse operation of concatHeads.
// grad is the gradient w.r.t the concatenated output, shape (batchSize, seqLen, totalDV)
func (m *MultiHeadAttention) splitGradsToHeads(grad *core.Tensor) []*core.Tensor {
	batchSize := grad.Shape()[0]
	seqLen := grad.Shape()[1]
	totalDV := grad.Shape()[2]
	dHead := totalDV / m.Config.NumHeads
	numHeads := m.Config.NumHeads

	gradHeads := make([]*core.Tensor, numHeads)

	// Pre-calculate strides for efficiency
	gradStride0 := grad.Strides()[0]
	gradStride1 := grad.Strides()[1]

	for h := 0; h < numHeads; h++ {
		// Create a new tensor for each head's gradient
		headData := make([]float64, batchSize*seqLen*dHead)
		headStride0 := seqLen * dHead
		headStride1 := dHead

		for i := 0; i < batchSize; i++ {
			for j := 0; j < seqLen; j++ {
				// Calculate the source offset in the concatenated gradient tensor
				srcOffset := i*gradStride0 + j*gradStride1 + h*dHead
				// Calculate the destination offset in the individual head's gradient tensor
				destOffset := i*headStride0 + j*headStride1
				// Copy the relevant slice of the gradient
				copy(headData[destOffset:destOffset+dHead], grad.Data()[srcOffset:srcOffset+dHead])
			}
		}
		gradHeads[h] = core.NewTensorFromData(headData, batchSize, seqLen, dHead)
	}
	return gradHeads
}

// combineGradHeads combines the gradient heads back into a single tensor.
// It's the reverse of splitHeads.
func (m *MultiHeadAttention) combineGradHeads(gradHeads []*core.Tensor) *core.Tensor {
	// This is essentially the same as concatHeads
	return m.concatHeads(gradHeads)
}

// Backward performs the backward pass for the MultiHeadAttention module.
// It computes the gradients with respect to the inputs Q, K, and V.
// The process is the reverse of the forward pass, applying the chain rule at each step.
func (m *MultiHeadAttention) Backward(gradOutput *core.Tensor) (*core.Tensor, *core.Tensor, *core.Tensor) {
	batchSize := gradOutput.Shape()[0]
	seqLen := gradOutput.Shape()[1]

	// Step 1: Backpropagate through the final linear layer WO.
	// The gradient `gradOutput` is first reshaped to match the output of WO.
	reshapedGradOutput := gradOutput.Reshape(batchSize*seqLen, m.Config.DModel)
	gradConcatOutputReshaped := m.WO.Backward(reshapedGradOutput)
	gradConcatOutput := gradConcatOutputReshaped.Reshape(batchSize, seqLen, m.Config.DV)

	// Step 2: Split the gradient for each head. This is the reverse of `concatHeads`.
	gradHeadOutputs := m.splitGradsToHeads(gradConcatOutput)

	// Initialize slices to hold gradients for Q, K, V from each head.
	gradQHeads := make([]*core.Tensor, m.Config.NumHeads)
	gradKHeads := make([]*core.Tensor, m.Config.NumHeads)
	gradVHeads := make([]*core.Tensor, m.Config.NumHeads)

	// Step 3: Backpropagate through ScaledDotProductAttention for each head.
	for i := 0; i < m.Config.NumHeads; i++ {
		// Restore the state (Q, K, V, weights) of the attention module for the backward pass of the current head.
		m.Attention.LastQuery = m.lastQHeads[i]
		m.Attention.LastKey = m.lastKHeads[i]
		m.Attention.LastValue = m.lastVHeads[i]
		m.Attention.LastWeights = m.lastAttnWeights[i]

		gradQHeads[i], gradKHeads[i], gradVHeads[i] = m.Attention.Backward(gradHeadOutputs[i])
	}

	// Step 4: Combine the gradient heads back into single tensors. This is the reverse of `splitHeads`.
	gradQ := m.combineGradHeads(gradQHeads)
	gradK := m.combineGradHeads(gradKHeads)
	gradV := m.combineGradHeads(gradVHeads)

	// Step 5: Reshape gradients to match the output shape of the initial linear layers (WQ, WK, WV).
	reshapedGradQ := gradQ.Reshape(batchSize*seqLen, m.Config.DK)
	reshapedGradK := gradK.Reshape(batchSize*seqLen, m.Config.DK)
	reshapedGradV := gradV.Reshape(batchSize*seqLen, m.Config.DV)

	// Step 6: Backpropagate through the initial linear layers WQ, WK, WV.
	gradQInputReshaped := m.WQ.Backward(reshapedGradQ)
	gradKInputReshaped := m.WK.Backward(reshapedGradK)
	gradVInputReshaped := m.WV.Backward(reshapedGradV)

	// Step 7: Reshape the final gradients to match the original input shapes.
	gradQInput := gradQInputReshaped.Reshape(batchSize, seqLen, m.Config.DModel)
	gradKInput := gradKInputReshaped.Reshape(batchSize, seqLen, m.Config.DModel)
	gradVInput := gradVInputReshaped.Reshape(batchSize, seqLen, m.Config.DModel)

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

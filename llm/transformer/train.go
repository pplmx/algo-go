package transformer

import (
	"fmt"

	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
)

// Trainer 结构体用于管理训练过程
type Trainer struct {
	Model     *TransformerModel
	Optimizer *core.AdamOptimizer
	LossFunc  *core.CrossEntropyLoss
	Config    config.AppConfig
}

// NewTrainer 创建一个新的 Trainer 实例
func NewTrainer(model *TransformerModel, optimizer *core.AdamOptimizer, lossFunc *core.CrossEntropyLoss, cfg config.AppConfig) *Trainer {
	return &Trainer{
		Model:     model,
		Optimizer: optimizer,
		LossFunc:  lossFunc,
		Config:    cfg,
	}
}

// Train 执行训练循环
func (t *Trainer) Train(loader *DataLoader) {
	t.Model.SetTraining(true)

	for epoch := 1; epoch <= t.Config.Train.NumEpochs; epoch++ {
		loader.Reset()
		totalLoss := 0.0
		batchCount := 0

		for loader.HasNextBatch() {
			srcBatch, tgtBatch, srcPaddingMask, _ := loader.NextBatch()

			// 创建目标序列的输入和实际目标
			// tgtInput 是实际输入到解码器的序列 (通常是目标序列去掉最后一个token，并加上起始token)
			// targets 是用于计算损失的实际目标 (通常是目标序列去掉第一个token)
			tgtInput := make([][]int, len(tgtBatch))
			targets := make([][]int, len(tgtBatch))
			for i, seq := range tgtBatch {
				tgtInput[i] = make([]int, len(seq))
				targets[i] = make([]int, len(seq))
				// 假设起始 token 是 1, 结束 token 是 2
				tgtInput[i][0] = t.Config.Transformer.StartToken
				copy(tgtInput[i][1:], seq[:len(seq)-1])
				copy(targets[i], seq[1:])
			}

			// 生成前瞻掩码 (Look-ahead Mask)
			// 这是一个上三角矩阵，用于防止解码器在训练时看到未来的 token
			lookAheadMask := core.GenerateLookAheadMask(len(tgtInput[0]))

			// 合并填充掩码和前瞻掩码
			// tgtMask 应该同时包含填充信息和前瞻信息
			tgtPaddingMask := loader.generatePaddingMask(tgtBatch) // Re-generate for clarity, though it's already returned by NextBatch
			combinedTgtMask := core.CombineMasks(tgtPaddingMask, lookAheadMask)

			// 前向传播
			logits, _, _, _ := t.Model.Forward(srcBatch, tgtInput, srcPaddingMask, combinedTgtMask)

			// 计算损失
			loss := t.LossFunc.Forward(logits, targets)
			totalLoss += loss

			// 反向传播
			// gradLoss := t.LossFunc.Backward(logits, targets)
			// gradModelInput, _ := t.Model.Backward(gradLoss)

			// 更新模型参数
			params := t.Model.GetParameters()
			grads := t.Model.GetGradients()

			// Zero out gradients before updating
			t.Model.ZeroGradients()

			// Update parameters
			for i := range params {
				t.Optimizer.Update(params[i], grads[i])
			}

			batchCount++
		}

		avgLoss := totalLoss / float64(batchCount)
		fmt.Printf("Epoch %d, Average Loss: %.4f\n", epoch, avgLoss)
	}
}

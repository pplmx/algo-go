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

// Train 执行完整的训练循环。
func (t *Trainer) Train(loader *DataLoader) []float64 {
	t.Model.SetTraining(true) // 将模型设置为训练模式（启用 dropout 等）
	lossHistory := []float64{}

	for epoch := 1; epoch <= t.Config.Train.NumEpochs; epoch++ {
		loader.Reset() // 在每个 epoch 开始时重置并打乱数据加载器
		totalLoss := 0.0
		batchCount := 0

		for loader.HasNextBatch() {
			// 从数据加载器获取一个批次的数据和掩码
			srcBatch, tgtBatch, srcPaddingMask, tgtPaddingMask := loader.NextBatch()

			// 1. 准备解码器的输入和目标序列
			// 对于一个目标序列 [t1, t2, t3, end],
			// 解码器输入应该是 [start, t1, t2, t3]
			// 真实目标应该是   [t1, t2, t3, end]
			seqLen := len(tgtBatch[0])
			decoderInput := make([][]int, len(tgtBatch))
			targets := make([][]int, len(tgtBatch))
			for i, seq := range tgtBatch {
				decoderInput[i] = make([]int, seqLen)
				targets[i] = make([]int, seqLen)
				decoderInput[i][0] = t.Config.Transformer.StartToken // 添加起始符
				copy(decoderInput[i][1:], seq[:seqLen-1])
				copy(targets[i], seq[1:])
				targets[i][seqLen-1] = t.Config.Transformer.EndToken // 确保最后一个是结束符
			}

			// 2. 创建目标序列的组合掩码
			// a. 创建前瞻掩码 (look-ahead mask) 防止解码器看到未来的 token
			lookAheadMask := core.GenerateLookAheadMask(seqLen)
			// b. 与填充掩码 (padding mask) 合并
			combinedTgtMask := core.CombineMasks(tgtPaddingMask, lookAheadMask)

			// 3. 将模型中所有参数的梯度清零
			t.Model.ZeroGradients()

			// 4. 执行前向传播
			logits, _, _, _ := t.Model.Forward(srcBatch, decoderInput, srcPaddingMask, combinedTgtMask)

			// 5. 计算损失
			loss := t.LossFunc.Forward(logits, targets)
			totalLoss += loss

			// 6. 执行反向传播
			// a. 计算损失函数关于 logits 的梯度
			gradLoss := t.LossFunc.Backward(logits, targets)
			// b. 将梯度反向传播到整个模型，计算所有参数的梯度
			t.Model.Backward(gradLoss)

			// 7. 使用优化器更新模型参数
			params := t.Model.GetParameters()
			grads := t.Model.GetGradients()
			for i := range params {
				t.Optimizer.Update(params[i], grads[i])
			}

			batchCount++
		}

		avgLoss := totalLoss / float64(batchCount)
		fmt.Printf("Epoch %d, Average Loss: %.4f\n", epoch, avgLoss)
		lossHistory = append(lossHistory, avgLoss)
	}
	return lossHistory
}

package transformer

import (
	"math"
	"math/rand"
	"time"

	"github.com/pplmx/algo-go/llm/transformer/config"
	"github.com/pplmx/algo-go/llm/transformer/core"
)

// Dataset 结构体用于存储训练数据
type Dataset struct {
	SourceSentences [][]int
	TargetSentences [][]int
	Config          config.TransformerConfig
}

// NewDataset 创建一个新的 Dataset 实例
func NewDataset(srcSentences, tgtSentences [][]int, cfg config.TransformerConfig) *Dataset {
	return &Dataset{
		SourceSentences: srcSentences,
		TargetSentences: tgtSentences,
		Config:          cfg,
	}
}

// DataLoader 结构体用于批量加载数据
type DataLoader struct {
	Dataset    *Dataset
	BatchSize  int
	CurrentIdx int
	rng        *rand.Rand
}

// NewDataLoader 创建一个新的 DataLoader 实例
func NewDataLoader(dataset *Dataset, batchSize int) *DataLoader {
	return &DataLoader{
		Dataset:    dataset,
		BatchSize:  batchSize,
		CurrentIdx: 0,
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// HasNextBatch 检查是否还有下一个批次
func (dl *DataLoader) HasNextBatch() bool {
	return dl.CurrentIdx < len(dl.Dataset.SourceSentences)
}

// NextBatch 获取下一个批次的数据
func (dl *DataLoader) NextBatch() ([][]int, [][]int, core.Matrix, core.Matrix) {
	start := dl.CurrentIdx
	end := dl.CurrentIdx + dl.BatchSize
	if end > len(dl.Dataset.SourceSentences) {
		end = len(dl.Dataset.SourceSentences)
	}

	srcBatch := dl.Dataset.SourceSentences[start:end]
	tgtBatch := dl.Dataset.TargetSentences[start:end]

	// 填充批次到最大序列长度
	paddedSrcBatch := dl.padBatch(srcBatch, dl.Dataset.Config.MaxSeqLen)
	paddedTgtBatch := dl.padBatch(tgtBatch, dl.Dataset.Config.MaxSeqLen)

	// 生成掩码
	srcMask := dl.generatePaddingMask(paddedSrcBatch)
	tgtMask := dl.generatePaddingMask(paddedTgtBatch)

	dl.CurrentIdx = end

	return paddedSrcBatch, paddedTgtBatch, srcMask, tgtMask
}

// Reset 重置 DataLoader 的索引，通常在每个 epoch 开始时调用
func (dl *DataLoader) Reset() {
	dl.CurrentIdx = 0
	// 每次重置时打乱数据，以确保每个 epoch 的批次顺序不同
	dl.shuffleDataset()
}

// padBatch 填充批次中的序列到指定长度
func (dl *DataLoader) padBatch(batch [][]int, maxLen int) [][]int {
	paddedBatch := make([][]int, len(batch))
	for i, seq := range batch {
		paddedSeq := make([]int, maxLen)
		copy(paddedSeq, seq)
		for j := len(seq); j < maxLen; j++ {
			paddedSeq[j] = dl.Dataset.Config.PadToken // 使用填充 token
		}
		paddedBatch[i] = paddedSeq
	}
	return paddedBatch
}

// generatePaddingMask 生成填充掩码
// 掩码的形状为 (batchSize * seqLen, batchSize * seqLen)
// 对于填充位置，掩码值为 -inf，否则为 0
func (dl *DataLoader) generatePaddingMask(batch [][]int) core.Matrix {
	batchSize := len(batch)
	seqLen := len(batch[0])
	mask := make(core.Matrix, batchSize*seqLen)
	for i := range mask {
		mask[i] = make([]float64, batchSize*seqLen)
	}

	for b := 0; b < batchSize; b++ {
		for r := 0; r < seqLen; r++ {
			// 如果当前 token 是填充 token，则掩盖其在注意力中的所有连接
			if batch[b][r] == dl.Dataset.Config.PadToken {
				// 掩盖整个行，因为这个 token 不应该关注任何东西
				for c := 0; c < seqLen; c++ {
					mask[b*seqLen+r][b*seqLen+c] = -math.MaxFloat64 // -inf
				}
			}
		}
	}
	return mask
}

// shuffleDataset 打乱数据集
func (dl *DataLoader) shuffleDataset() {
	combined := make([][2][]int, len(dl.Dataset.SourceSentences))
	for i := range dl.Dataset.SourceSentences {
		combined[i] = [2][]int{dl.Dataset.SourceSentences[i], dl.Dataset.TargetSentences[i]}
	}

	dl.rng.Shuffle(len(combined), func(i, j int) {
		combined[i], combined[j] = combined[j], combined[i]
	})

	for i := range dl.Dataset.SourceSentences {
		dl.Dataset.SourceSentences[i] = combined[i][0]
		dl.Dataset.TargetSentences[i] = combined[i][1]
	}
}

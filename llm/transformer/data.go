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

type DataLoader struct {
	Dataset    *Dataset
	BatchSize  int
	CurrentIdx int
	rng        *rand.Rand
}

// NewDataLoader creates a new DataLoader instance.
// The random number generator is seeded with the current time, which might lead to non-deterministic
// behavior if multiple DataLoaders are created in rapid succession. For deterministic behavior
// in tests, consider passing a fixed rand.Source.
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
func (dl *DataLoader) NextBatch() ([][]int, [][]int, *core.Tensor, *core.Tensor) {
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

// generatePaddingMask 生成填充掩码，形状为 (batch, 1, 1, seq_len)
// 这样它就可以被广播到 (batch, num_heads, seq_len, seq_len) 的注意力分数张量上。
func (dl *DataLoader) generatePaddingMask(batch [][]int) *core.Tensor {
	batchSize := len(batch)
	seqLen := len(batch[0])
	mask := core.Zeros(batchSize, 1, 1, seqLen)

	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			if batch[i][j] == dl.Dataset.Config.PadToken {
				mask.Set(-math.MaxFloat64, i, 0, 0, j)
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

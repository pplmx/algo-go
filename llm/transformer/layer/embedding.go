package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 词嵌入层
type Embedding struct {
	Weight core.Matrix
}

func NewEmbedding(vocabSize, dModel int) *Embedding {
	pi := &core.ParameterInitializer{}
	return &Embedding{
		Weight: pi.XavierUniform(vocabSize, dModel, 1.0),
	}
}

func (e *Embedding) Forward(input [][]int) core.Matrix {
	batchSize := len(input)
	seqLen := len(input[0])

	result := make(core.Matrix, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			idx := i*seqLen + j
			result[idx] = make([]float64, len(e.Weight[0]))
			copy(result[idx], e.Weight[input[i][j]])
		}
	}

	return result
}

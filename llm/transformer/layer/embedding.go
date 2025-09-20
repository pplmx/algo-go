package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 词嵌入层
type Embedding struct {
	Weight core.Matrix

	lastInput [][]int // Store input token IDs for backward pass
	gradWeight core.Matrix
}

func NewEmbedding(vocabSize, dModel int) *Embedding {
	pi := &core.ParameterInitializer{}
	return &Embedding{
		Weight: pi.XavierUniform(vocabSize, dModel, 1.0),
	}
}

func (e *Embedding) Forward(input [][]int) core.Matrix {
	e.lastInput = input // Store input
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

func (e *Embedding) Backward(gradOutput core.Matrix) core.Matrix {
	// Initialize gradWeight to zeros
	e.gradWeight = core.Zeros(len(e.Weight), len(e.Weight[0]))

	batchSize := len(e.lastInput)
	seqLen := len(e.lastInput[0])

	// Accumulate gradients for embedding weights
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			tokenID := e.lastInput[i][j]
			gradOutputRowIndex := i*seqLen + j
			for k := range gradOutput[gradOutputRowIndex] {
				e.gradWeight[tokenID][k] += gradOutput[gradOutputRowIndex][k]
			}
		}
	}

	// Gradient with respect to input token IDs is not typically calculated
	return nil // Or core.Matrix{} if an empty matrix is preferred
}

func (e *Embedding) GetParameters() []core.Matrix {
	return []core.Matrix{e.Weight}
}

func (e *Embedding) GetGradients() []core.Matrix {
	return []core.Matrix{e.gradWeight}
}

func (e *Embedding) ZeroGradients() {
	if e.gradWeight != nil {
		e.gradWeight = core.Zeros(len(e.gradWeight), len(e.gradWeight[0]))
	}
}

func (e *Embedding) SetTraining(training bool) {
	// Embedding layer does not have training-dependent behavior
}

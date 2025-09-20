package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 词嵌入层
type Embedding struct {
	Weight core.Matrix

	lastInput [][]int // Store input token IDs for backward pass
	gradWeight core.Matrix
}

// NewEmbedding creates a new Embedding layer.
func NewEmbedding(vocabSize, dModel int) *Embedding {
	pi := &core.ParameterInitializer{}
	return &Embedding{
		Weight: pi.XavierUniform(vocabSize, dModel, 1.0),
	}
}

// Forward performs the forward pass for the Embedding layer.
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

// Backward performs the backward pass for the Embedding layer.
// It calculates gradients with respect to the embedding matrix.
// The gradient with respect to input token IDs is not typically calculated.
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

// GetParameters returns the trainable parameters (embedding weights) of the Embedding layer.
func (e *Embedding) GetParameters() []core.Matrix {
	return []core.Matrix{e.Weight}
}

// GetGradients returns the gradients of the trainable parameters (gradWeight).
func (e *Embedding) GetGradients() []core.Matrix {
	return []core.Matrix{e.gradWeight}
}

// ZeroGradients sets the gradients of the trainable parameters to zero.
func (e *Embedding) ZeroGradients() {
	if e.gradWeight != nil {
		e.gradWeight = core.Zeros(len(e.gradWeight), len(e.gradWeight[0]))
	}
}

// SetTraining sets the training mode for the Embedding layer.
// The Embedding layer does not have training-dependent behavior, so this method does nothing.
func (e *Embedding) SetTraining(training bool) {
	// Embedding layer does not have training-dependent behavior
}

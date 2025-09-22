package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 词嵌入层
type Embedding struct {
	Weight *core.Tensor

	lastInput  [][]int // Store input token IDs for backward pass
	gradWeight *core.Tensor
}

// NewEmbedding creates a new Embedding layer.
func NewEmbedding(vocabSize, dModel int) *Embedding {
	pi := &core.ParameterInitializer{}
	return &Embedding{
		Weight: pi.XavierUniform(vocabSize, dModel, 1.0),
	}
}

// Forward performs the forward pass for the Embedding layer.
func (e *Embedding) Forward(input [][]int, args ...interface{}) *core.Tensor {
	e.lastInput = input // Store input
	batchSize := len(input)
	seqLen := len(input[0])
	dModel := e.Weight.Shape()[1]

	result := core.NewTensor(batchSize, seqLen, dModel)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			tokenID := input[i][j]
			// This is a simplified way to gather embeddings. A real implementation would be more efficient.
			for k := 0; k < dModel; k++ {
				result.Set(e.Weight.Get(tokenID, k), i, j, k)
			}
		}
	}

	return result
}

// Backward performs the backward pass for the Embedding layer.
// It calculates gradients with respect to the embedding matrix.
// The gradient with respect to input token IDs is not typically calculated.
func (e *Embedding) Backward(gradOutput *core.Tensor) *core.Tensor {
	// Initialize gradWeight to zeros
	e.gradWeight = core.Zeros(e.Weight.Shape()...)

	batchSize := len(e.lastInput)
	seqLen := len(e.lastInput[0])
	dModel := e.Weight.Shape()[1]

	// Accumulate gradients for embedding weights
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			tokenID := e.lastInput[i][j]
			for k := 0; k < dModel; k++ {
				grad := e.gradWeight.Get(tokenID, k)
				e.gradWeight.Set(grad+gradOutput.Get(i, j, k), tokenID, k)
			}
		}
	}

	// Gradient with respect to input token IDs is not typically calculated
	return nil
}

// GetParameters returns the trainable parameters (embedding weights) of the Embedding layer.
func (e *Embedding) GetParameters() []*core.Tensor {
	return []*core.Tensor{e.Weight}
}

// GetGradients returns the gradients of the trainable parameters (gradWeight).
func (e *Embedding) GetGradients() []*core.Tensor {
	return []*core.Tensor{e.gradWeight}
}

// ZeroGradients sets the gradients of the trainable parameters to zero.
func (e *Embedding) ZeroGradients() {
	if e.gradWeight != nil {
		e.gradWeight = core.Zeros(e.gradWeight.Shape()...)
	}
}

// SetTraining sets the training mode for the Embedding layer.
// The Embedding layer does not have training-dependent behavior, so this method does nothing.
func (e *Embedding) SetTraining(training bool) {
	// Embedding layer does not have training-dependent behavior
}

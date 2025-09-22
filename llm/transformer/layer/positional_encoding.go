package layer

import (
	"math"

	"github.com/pplmx/algo-go/llm/transformer/core"
)

// 位置编码模块
type PositionalEncoding struct {
	Encoding  *core.Tensor
	MaxSeqLen int
	DModel    int
}

func NewPositionalEncoding(maxSeqLen, dModel int) *PositionalEncoding {
	pe := &PositionalEncoding{
		MaxSeqLen: maxSeqLen,
		DModel:    dModel,
		Encoding:  core.NewTensor(maxSeqLen, dModel),
	}
	pe.initializeEncoding()
	return pe
}

func (pe *PositionalEncoding) initializeEncoding() {
	for pos := 0; pos < pe.MaxSeqLen; pos++ {
		for i := 0; i < pe.DModel; i += 2 {
			angle := float64(pos) / math.Pow(10000.0, float64(i)/float64(pe.DModel))
			pe.Encoding.Set(math.Sin(angle), pos, i)
			if i+1 < pe.DModel {
				pe.Encoding.Set(math.Cos(angle), pos, i+1)
			}
		}
	}
}

func (pe *PositionalEncoding) Forward(input *core.Tensor, seqLen int) *core.Tensor {
	if seqLen > pe.MaxSeqLen {
		panic("Input sequence exceeds maximum length")
	}

	return input.Add(pe.Encoding.Slice(0, seqLen).ExpandDims(0))
}

func (pe *PositionalEncoding) Backward(gradOutput *core.Tensor) *core.Tensor {
	// PositionalEncoding has no trainable parameters and its forward pass is an element-wise addition.
	// So, the gradient with respect to the input is simply the gradOutput.
	return gradOutput
}

func (pe *PositionalEncoding) GetParameters() []*core.Tensor {
	return []*core.Tensor{}
}

func (pe *PositionalEncoding) GetGradients() []*core.Tensor {
	return []*core.Tensor{}
}

func (pe *PositionalEncoding) ZeroGradients() {}

func (pe *PositionalEncoding) SetTraining(training bool) {}

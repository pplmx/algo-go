package layer

import (
	"math"

	"github.com/pplmx/algo-go/llm/transformer/core"
)

// 位置编码模块
type PositionalEncoding struct {
	Encoding  core.Matrix
	MaxSeqLen int
	DModel    int
}

func NewPositionalEncoding(maxSeqLen, dModel int) *PositionalEncoding {
	pe := &PositionalEncoding{
		MaxSeqLen: maxSeqLen,
		DModel:    dModel,
		Encoding:  make(core.Matrix, maxSeqLen),
	}
	pe.initializeEncoding()
	return pe
}

func (pe *PositionalEncoding) initializeEncoding() {
	for pos := 0; pos < pe.MaxSeqLen; pos++ {
		pe.Encoding[pos] = make([]float64, pe.DModel)
		for i := 0; i < pe.DModel; i += 2 {
			angle := float64(pos) / math.Pow(10000.0, float64(i)/float64(pe.DModel))
			pe.Encoding[pos][i] = math.Sin(angle)
			if i+1 < pe.DModel {
				pe.Encoding[pos][i+1] = math.Cos(angle)
			}
		}
	}
}

func (pe *PositionalEncoding) Forward(input core.Matrix) core.Matrix {
	seqLen := len(input)
	if seqLen > pe.MaxSeqLen {
		panic("Input sequence exceeds maximum length")
	}

	result := make(core.Matrix, seqLen)
	for i := 0; i < seqLen; i++ {
		result[i] = make([]float64, len(input[i]))
		for j := range input[i] {
			result[i][j] = input[i][j] + pe.Encoding[i][j]
		}
	}
	return result
}

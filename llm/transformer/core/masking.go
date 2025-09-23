package core

import (
	"math"
)

// GenerateLookAheadMask generates a look-ahead mask for self-attention in the decoder.
// The mask has a shape of (seqLen, seqLen).
// It is an upper triangular matrix where elements above the diagonal are -inf, and others are 0.
func GenerateLookAheadMask(seqLen int) *Tensor {
	mask := Zeros(seqLen, seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j > i {
				mask.Set(-math.MaxFloat64, i, j) // -inf
			}
		}
	}
	return mask
}

// CombineMasks combines a padding mask and a look-ahead mask.
// It handles the broadcasting of a (batch, 1, 1, seq_len) padding mask and
// a (seq_len, seq_len) look-ahead mask into a (batch, 1, seq_len, seq_len) combined mask.
func CombineMasks(paddingMask, lookAheadMask *Tensor) *Tensor {
	if lookAheadMask.Ndim() != 2 {
		panic("lookAheadMask must be 2D")
	}
	if paddingMask.Ndim() != 4 || paddingMask.Shape()[1] != 1 || paddingMask.Shape()[2] != 1 {
		panic("paddingMask must be 4D with shape (batch, 1, 1, seq_len)")
	}

	batchSize := paddingMask.Shape()[0]
	seqLen := lookAheadMask.Shape()[0]

	if paddingMask.Shape()[3] != seqLen {
		panic("sequence length mismatch between padding mask and lookahead mask")
	}

	// Manually combine the masks to avoid relying on the incomplete broadcasting implementation.
	combined := NewTensor(batchSize, 1, seqLen, seqLen)

	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				// Get value from look-ahead mask (same for all in batch)
				laVal := lookAheadMask.Get(i, j)
				// Get value from padding mask (depends on batch and j)
				padVal := paddingMask.Get(b, 0, 0, j)
				// Combine them by taking the minimum. If either is -inf, the result is -inf.
				combined.Set(math.Min(laVal, padVal), b, 0, i, j)
			}
		}
	}
	return combined
}

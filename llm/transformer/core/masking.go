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

// CombineMasks combines two masks by taking the minimum of their corresponding elements.
// Both masks must have the same shape.
func CombineMasks(mask1, mask2 *Tensor) *Tensor {
	if !mask1.sameShape(mask2) {
		panic("masks must have the same shape")
	}
	result := NewTensor(mask1.Shape()...)
	for i := 0; i < mask1.Size(); i++ {
		result.data[i] = math.Min(mask1.data[i], mask2.data[i])
	}
	return result
}

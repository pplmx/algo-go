package core

import (
	"math"
)

// GenerateLookAheadMask generates a look-ahead mask for self-attention in the decoder.
// The mask has a shape of (seqLen, seqLen).
// It is an upper triangular matrix where elements above the diagonal are -inf, and others are 0.
func GenerateLookAheadMask(seqLen int) Matrix {
	mask := make(Matrix, seqLen)
	for i := range mask {
		mask[i] = make([]float64, seqLen)
		for j := range mask[i] {
			if j > i {
				mask[i][j] = -math.MaxFloat64 // -inf
			}
		}
	}
	return mask
}

// CombineMasks combines two masks by taking the minimum of their corresponding elements.
// Both masks must have the same shape.
func CombineMasks(mask1, mask2 Matrix) Matrix {
	// Assuming mask1 and mask2 have the same shape
	combinedMask := make(Matrix, len(mask1))
	for i := range mask1 {
		combinedMask[i] = make([]float64, len(mask1[i]))
		for j := range mask1[i] {
			combinedMask[i][j] = math.Min(mask1[i][j], mask2[i][j])
		}
	}
	return combinedMask
}

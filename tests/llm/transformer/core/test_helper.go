package core

import (
	"math"

	"github.com/pplmx/algo-go/llm/transformer/core"
)

// MatricesAlmostEqual 比较两个矩阵在给定的误差范围内是否相等
func MatricesAlmostEqual(a, b core.Matrix, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Abs(a[i][j]-b[i][j]) > tolerance {
				return false
			}
		}
	}
	return true
}

package helpers

import (
	"math"

	"github.com/pplmx/algo-go/llm/transformer/core"
)

// TensorsAlmostEqual 比较两个张量在给定的误差范围内是否相等
func TensorsAlmostEqual(a, b *core.Tensor, tolerance float64) bool {
	if !core.Shape(a.Shape()).Equal(core.Shape(b.Shape())) {
		return false
	}
	for i := 0; i < a.Size(); i++ {
		if math.Abs(a.Data()[i]-b.Data()[i]) > tolerance {
			return false
		}
	}
	return true
}

// IntSliceEqual 比较两个 int 切片是否相等
func IntSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

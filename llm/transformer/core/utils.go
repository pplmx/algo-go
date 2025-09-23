package core

import "math"

type Shape []int

func (s Shape) Equal(other Shape) bool {
	if len(s) != len(other) {
		return false
	}
	for i, v := range s {
		if v != other[i] {
			return false
		}
	}
	return true
}

// IsNaN checks if a float64 is NaN.
func IsNaN(f float64) bool {
	return math.IsNaN(f)
}

// IsInf checks if a float64 is positive or negative infinity.
func IsInf(f float64, sign int) bool {
	return math.IsInf(f, sign)
}

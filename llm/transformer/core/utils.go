package core

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

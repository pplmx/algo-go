package core

// Trainable 接口定义了可训练层应实现的方法
type Trainable interface {
	GetParameters() []*Tensor
	GetGradients() []*Tensor
	ZeroGradients()
}

// Layer 接口定义了所有层应实现的方法
type Layer interface {
	Forward(input *Tensor, args ...interface{}) *Tensor
	Backward(gradOutput *Tensor) *Tensor
	SetTraining(training bool)
}

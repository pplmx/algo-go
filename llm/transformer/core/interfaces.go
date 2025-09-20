package core

// Trainable 接口定义了可训练层应实现的方法
type Trainable interface {
	GetParameters() []Matrix
	GetGradients() []Matrix
	ZeroGradients()
}

// Layer 接口定义了所有层应实现的方法
type Layer interface {
	Forward(input Matrix, args ...interface{}) Matrix
	Backward(gradOutput Matrix) Matrix
	SetTraining(training bool)
}

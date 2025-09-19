package layer

import "github.com/pplmx/algo-go/llm/transformer/core"

// 前馈网络模块
type FeedForwardNetwork struct {
	fc1     *LinearLayer
	fc2     *LinearLayer
	dropout *core.Dropout
}

func NewFeedForwardNetwork(dModel, ffnHiddenSize int, dropoutRate float64, useBias bool) *FeedForwardNetwork {
	return &FeedForwardNetwork{
		fc1:     NewLinearLayer(dModel, ffnHiddenSize, useBias),
		fc2:     NewLinearLayer(ffnHiddenSize, dModel, useBias),
		dropout: core.NewDropout(dropoutRate, true),
	}
}

func (f *FeedForwardNetwork) SetTraining(training bool) {
	f.dropout.SetTraining(training)
}

func (f *FeedForwardNetwork) Forward(x core.Matrix) core.Matrix {
	// 第一层: ReLU激活
	h := f.fc1.Forward(x)
	for i := range h {
		for j := range h[i] {
			if h[i][j] < 0 {
				h[i][j] = 0
			}
		}
	}

	// 应用 dropout
	h = f.dropout.Forward(h)

	// 第二层: 线性变换
	return f.fc2.Forward(h)
}

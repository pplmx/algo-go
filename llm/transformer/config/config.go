package config

// 配置结构体
type TransformerConfig struct {
	DModel        int
	NumHeads      int
	DK            int
	DV            int
	FFNHiddenSize int
	MaxSeqLen     int
	VocabSize     int
	BatchSize     int
	DropoutRate   float64
	UseBias       bool
	LearningRate  float64
	Beta1         float64
	Beta2         float64
	Eps           float64
}

// 默认配置
func DefaultConfig() TransformerConfig {
	return TransformerConfig{
		DModel:        512,
		NumHeads:      8,
		DK:            64,
		DV:            64,
		FFNHiddenSize: 2048,
		MaxSeqLen:     512,
		VocabSize:     30000,
		BatchSize:     32,
		DropoutRate:   0.1,
		UseBias:       true,
		LearningRate:  0.001,
		Beta1:         0.9,
		Beta2:         0.999,
		Eps:           1e-8,
	}
}
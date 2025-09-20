package config

// TransformerConfig holds all hyperparameters for the Transformer model.
type TransformerConfig struct {
	VocabSize   int
	DModel      int // Dimension of the model (embedding dimension)
	NHead       int // Number of attention heads
	NLayer      int // Number of encoder and decoder layers
	DFF         int // Dimension of the feed-forward network (FFNInnerDim)
	DropoutRate float64
	Epsilon     float64 // For Layer Normalization
	MaxSeqLen   int     // Maximum sequence length for positional encoding
	StartToken  int     // Start of sequence token ID
	EndToken    int     // End of sequence token ID
	PadToken    int     // Padding token ID

	// New fields for layers
	FFNHiddenSize int  // Hidden size for FeedForward Network (same as DFF, but explicit for clarity)
	UseBias       bool // Whether to use bias in linear layers
	DK            int  // Dimension of K (key) and Q (query) in attention
	DV            int  // Dimension of V (value) in attention
	NumHeads      int  // Number of attention heads (redundant with NHead, but kept for explicit mapping if needed)
}

// TrainConfig holds hyperparameters for the training process.
type TrainConfig struct {
	LearningRate float64
	BatchSize    int
	NumEpochs    int
	Beta1        float64 // Adam optimizer parameter
	Beta2        float64 // Adam optimizer parameter
	Eps          float64 // Adam optimizer parameter (epsilon for numerical stability)
	// 可以在这里添加其他训练相关的参数，例如优化器类型、warmup 步数等
}

// AppConfig (您的 XConfig) holds all application-wide configurations.
type AppConfig struct {
	Transformer TransformerConfig
	Train       TrainConfig
	// 可以在这里添加其他全局配置，例如数据路径、模型保存路径、日志设置等
}

// NewDefaultTransformerConfig creates a default TransformerConfig.
func NewDefaultTransformerConfig() TransformerConfig {
	return TransformerConfig{
		VocabSize:   10000,
		DModel:      512,
		NHead:       8,
		NLayer:      6,
		DFF:         2048,
		DropoutRate: 0.1,
		Epsilon:     1e-6,
		MaxSeqLen:   512,
		StartToken:  1,
		EndToken:    2,
		PadToken:    0,

		// Default values for new fields
		FFNHiddenSize: 2048, // Typically DFF
		UseBias:       true,
		DK:            64, // DModel / NHead
		DV:            64, // DModel / NHead
		NumHeads:      8,  // Same as NHead
	}
}

// NewDefaultTrainConfig creates a default TrainConfig.
func NewDefaultTrainConfig() TrainConfig {
	return TrainConfig{
		LearningRate: 1e-4,
		BatchSize:    32,
		NumEpochs:    10,
		Beta1:        0.9,
		Beta2:        0.999,
		Eps:          1e-8,
	}
}

// NewDefaultAppConfig creates a default AppConfig.
func NewDefaultAppConfig() AppConfig {
	return AppConfig{
		Transformer: NewDefaultTransformerConfig(),
		Train:       NewDefaultTrainConfig(),
	}
}

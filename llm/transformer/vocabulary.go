package transformer

import (
	"strings"
)

// Vocabulary 结构体用于存储词汇表和 token 到 ID 的映射
type Vocabulary struct {
	tokenToID  map[string]int
	idToToken  []string
	size       int
	unkToken   string
	padToken   string
	startToken string
	endToken   string
}

// NewVocabulary 创建一个新的 Vocabulary 实例，并初始化特殊 token
func NewVocabulary(unk, pad, start, end string) *Vocabulary {
	vocab := &Vocabulary{
		tokenToID:  make(map[string]int),
		idToToken:  []string{},
		size:       0,
		unkToken:   unk,
		padToken:   pad,
		startToken: start,
		endToken:   end,
	}

	// 添加特殊 token
	vocab.AddToken(unk)
	vocab.AddToken(pad)
	vocab.AddToken(start)
	vocab.AddToken(end)

	return vocab
}

// AddToken 向词汇表添加一个 token，如果已存在则不添加
func (v *Vocabulary) AddToken(token string) int {
	if _, ok := v.tokenToID[token]; !ok {
		v.tokenToID[token] = v.size
		v.idToToken = append(v.idToToken, token)
		v.size++
	}
	return v.tokenToID[token]
}

// LookupID 返回给定 token 的 ID，如果不存在则返回 UNK token 的 ID
func (v *Vocabulary) LookupID(token string) int {
	id, ok := v.tokenToID[token]
	if !ok {
		return v.tokenToID[v.unkToken]
	}
	return id
}

// LookupToken 返回给定 ID 的 token，如果 ID 无效则返回 UNK token
func (v *Vocabulary) LookupToken(id int) string {
	if id < 0 || id >= v.size {
		return v.unkToken
	}
	return v.idToToken[id]
}

// Size 返回词汇表的大小
func (v *Vocabulary) Size() int {
	return v.size
}

// BuildVocabulary 从句子列表中构建词汇表
func BuildVocabulary(sentences [][]string, unk, pad, start, end string) *Vocabulary {
	vocab := NewVocabulary(unk, pad, start, end)
	for _, sentence := range sentences {
		for _, token := range sentence {
			vocab.AddToken(token)
		}
	}
	return vocab
}

// Encode 将一个句子（token 列表）编码为 ID 列表
func (v *Vocabulary) Encode(sentence []string) []int {
	encoded := make([]int, len(sentence))
	for i, token := range sentence {
		encoded[i] = v.LookupID(token)
	}
	return encoded
}

// Decode 将一个 ID 列表解码为句子（token 列表）
func (v *Vocabulary) Decode(ids []int) []string {
	decoded := make([]string, len(ids))
	for i, id := range ids {
		decoded[i] = v.LookupToken(id)
	}
	return decoded
}

// SimpleTokenizer 简单的分词器，按空格分割
func SimpleTokenizer(text string) []string {
	return strings.Fields(text)
}

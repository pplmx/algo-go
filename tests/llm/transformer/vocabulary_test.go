package transformer_test

import (
	"reflect"
	"testing"

	"github.com/pplmx/algo-go/llm/transformer"
)

func TestNewVocabulary(t *testing.T) {
	vocab := transformer.NewVocabulary("<unk>", "<pad>", "<sos>", "<eos>")

	if vocab.Size() != 4 {
		t.Errorf("Expected vocabulary size 4, got %d", vocab.Size())
	}

	if vocab.LookupID("<unk>") != 0 || vocab.LookupID("<pad>") != 1 || vocab.LookupID("<sos>") != 2 || vocab.LookupID("<eos>") != 3 {
		t.Errorf("Special tokens not initialized correctly")
	}
}

func TestAddToken(t *testing.T) {
	vocab := transformer.NewVocabulary("<unk>", "<pad>", "<sos>", "<eos>")

	vocab.AddToken("hello")
	vocab.AddToken("world")
	vocab.AddToken("hello") // Add existing token

	if vocab.Size() != 6 {
		t.Errorf("Expected vocabulary size 6, got %d", vocab.Size())
	}

	if vocab.LookupID("hello") != 4 || vocab.LookupID("world") != 5 {
		t.Errorf("Tokens not added correctly")
	}
}

func TestLookupID(t *testing.T) {
	vocab := transformer.NewVocabulary("<unk>", "<pad>", "<sos>", "<eos>")
	vocab.AddToken("test")

	if vocab.LookupID("test") != 4 {
		t.Errorf("Expected ID 4 for 'test', got %d", vocab.LookupID("test"))
	}

	if vocab.LookupID("nonexistent") != vocab.LookupID("<unk>") {
		t.Errorf("Expected UNK ID for nonexistent token, got %d", vocab.LookupID("nonexistent"))
	}
}

func TestLookupToken(t *testing.T) {
	vocab := transformer.NewVocabulary("<unk>", "<pad>", "<sos>", "<eos>")
	vocab.AddToken("test")

	if vocab.LookupToken(4) != "test" {
		t.Errorf("Expected token 'test' for ID 4, got %s", vocab.LookupToken(4))
	}

	if vocab.LookupToken(999) != "<unk>" {
		t.Errorf("Expected UNK token for invalid ID, got %s", vocab.LookupToken(999))
	}
}

func TestBuildVocabulary(t *testing.T) {
	sentences := [][]string{
		{"hello", "world"},
		{"go", "lang", "hello"},
	}
	vocab := transformer.BuildVocabulary(sentences, "<unk>", "<pad>", "<sos>", "<eos>")

	if vocab.Size() != 8 {
		t.Errorf("Expected vocabulary size 8, got %d", vocab.Size())
	}

	if vocab.LookupID("go") == vocab.LookupID("<unk>") {
		t.Errorf("Expected 'go' to be in vocabulary")
	}
}

func TestEncodeDecode(t *testing.T) {
	vocab := transformer.NewVocabulary("<unk>", "<pad>", "<sos>", "<eos>")
	vocab.AddToken("apple")
	vocab.AddToken("banana")
	vocab.AddToken("orange")

	sentence := []string{"apple", "banana", "grape", "orange"}
	// Expected IDs: apple (4), banana (5), grape (<unk>=0), orange (6)
	expectedIDs := []int{4, 5, 0, 6}

	encoded := vocab.Encode(sentence)
	if !reflect.DeepEqual(encoded, expectedIDs) {
		t.Errorf("Encode incorrect. Got %v, want %v", encoded, expectedIDs)
	}

	decoded := vocab.Decode(encoded)
	expectedDecoded := []string{"apple", "banana", "<unk>", "orange"}
	if !reflect.DeepEqual(decoded, expectedDecoded) {
		t.Errorf("Decode incorrect. Got %v, want %v", decoded, expectedDecoded)
	}
}

func TestSimpleTokenizer(t *testing.T) {
	text := "hello world this is a test"
	expectedTokens := []string{"hello", "world", "this", "is", "a", "test"}

	tokens := transformer.SimpleTokenizer(text)

	if !reflect.DeepEqual(tokens, expectedTokens) {
		t.Errorf("SimpleTokenizer incorrect. Got %v, want %v", tokens, expectedTokens)
	}
}

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run scripts/new_problem.go <id> <slug>")
		os.Exit(1)
	}

	// Format ID to 5 digits
	idNum, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Println("Invalid problem ID:", os.Args[1])
		os.Exit(1)
	}
	idStr := fmt.Sprintf("%05d", idNum)

	slug := strings.ToLower(os.Args[2])
	filename := fmt.Sprintf("p%s_%s", idStr, slug)

	funcName := toCamelCase(slug)

	// Create source & test directories
	srcDir := "leetcode"
	testDir := filepath.Join("tests", "leetcode")
	os.MkdirAll(srcDir, 0755)
	os.MkdirAll(testDir, 0755)

	// File paths
	srcPath := filepath.Join(srcDir, filename+".go")
	testPath := filepath.Join(testDir, filename+"_test.go")

	// Write source file
	srcContent := fmt.Sprintf(`package leetcode

func %s() {
	// TODO: implement
}
`, funcName)
	if err := os.WriteFile(srcPath, []byte(srcContent), 0644); err != nil {
		fmt.Println("Error writing source file:", err)
		os.Exit(1)
	}

	// Write test file
	testContent := fmt.Sprintf(`package leetcode_test

import (
	"testing"

	"github.com/pplmx/algo-go/leetcode"
)

func Test%s(t *testing.T) {
	// TODO: implement test
	_ = leetcode.%s
}
`, funcName, funcName)
	if err := os.WriteFile(testPath, []byte(testContent), 0644); err != nil {
		fmt.Println("Error writing test file:", err)
		os.Exit(1)
	}

	fmt.Println("Created problem:", filename)
}

// Convert slug like "add_two_numbers" -> "AddTwoNumbers"
func toCamelCase(s string) string {
	parts := strings.Split(s, "_")
	for i := range parts {
		if len(parts[i]) > 0 {
			parts[i] = strings.ToUpper(parts[i][:1]) + parts[i][1:]
		}
	}
	return strings.Join(parts, "")
}

# algo-go

Algorithm practice in Go.

## How to Run Tests

You can use the `make` command to run all tests:

```shell
make test
```

## How to Add a New Problem

```bash
# Usage:
#     go run scripts/new_problem.go <problem_id> <problem_name>
# Example:
go run scripts/new_problem.go 2 add_two_numbers
```

## Directory Structure

- `leetcode/`: LeetCode problem solutions
- `tests/leetcode/`: Corresponding unit tests for LeetCode problems

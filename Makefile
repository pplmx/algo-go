test:
	@go test -v ./... ./tests/...

fmt:
	@gofmt -l -s -w .
	@goimports -l -w .

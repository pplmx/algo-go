test:
	@go test -v ./...

fmt:
	@gofmt -l -s -w .
	@goimports -l -w .

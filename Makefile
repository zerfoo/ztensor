.PHONY: test lint bench vet

test:
	go test ./... -count=1 -timeout 300s

lint: vet
	@echo "Lint passed (go vet)"

bench:
	go test ./... -bench=. -benchmem -run=^$$ -timeout 600s

vet:
	go vet ./...

.PHONY: install proto build run test clean

# Install dependencies
install:
	# Install Go dependencies
	cd backend && go mod tidy

	# Create Python virtual environments
	python -m venv stt_service/venv
	python -m venv nlp_service/venv
	python -m venv tts_service/venv

	# Install Python dependencies
	. stt_service/venv/bin/activate && pip install -r stt_service/requirements.txt
	. nlp_service/venv/bin/activate && pip install -r nlp_service/requirements.txt
	. tts_service/venv/bin/activate && pip install -r tts_service/requirements.txt

# Generate Protocol Buffer code
proto:
	./scripts/generate_proto.sh

# Build Docker images
build:
	docker-compose build

# Run services
run:
	docker-compose up

# Run tests
test:
	# Run Go tests
	cd backend && go test ./...

	# Run Python tests
	. stt_service/venv/bin/activate && cd stt_service && pytest
	. nlp_service/venv/bin/activate && cd nlp_service && pytest
	. tts_service/venv/bin/activate && cd tts_service && pytest

# Clean up
clean:
	# Remove Python virtual environments
	rm -rf stt_service/venv
	rm -rf nlp_service/venv
	rm -rf tts_service/venv

	# Remove generated files
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	find . -name "*.pb.go" -delete
	find . -name "*.pb.py" -delete
	find . -name "*_grpc.py" -delete

	# Stop and remove containers
	docker-compose down 
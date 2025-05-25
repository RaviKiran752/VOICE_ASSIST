#!/bin/bash

# Create proto directory in each service
mkdir -p stt_service/src/proto
mkdir -p nlp_service/src/proto
mkdir -p tts_service/src/proto

# Generate Python gRPC code
python -m grpc_tools.protoc \
    -I./proto \
    --python_out=stt_service/src/proto \
    --grpc_python_out=stt_service/src/proto \
    ./proto/voice_assist.proto

python -m grpc_tools.protoc \
    -I./proto \
    --python_out=nlp_service/src/proto \
    --grpc_python_out=nlp_service/src/proto \
    ./proto/voice_assist.proto

python -m grpc_tools.protoc \
    -I./proto \
    --python_out=tts_service/src/proto \
    --grpc_python_out=tts_service/src/proto \
    ./proto/voice_assist.proto

# Generate Go gRPC code
protoc \
    --go_out=backend/internal/proto \
    --go_opt=paths=source_relative \
    --go-grpc_out=backend/internal/proto \
    --go-grpc_opt=paths=source_relative \
    ./proto/voice_assist.proto 
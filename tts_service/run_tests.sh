#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Generate proto files
cd ..
python generate_proto.py

# Install the package in development mode
pip install -e .

# Run tests
cd tts_service
python -m pytest tests/ -v 
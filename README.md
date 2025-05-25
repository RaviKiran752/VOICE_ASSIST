# Voice Assistant

A comprehensive voice assistant system with Text-to-Speech (TTS), Speech-to-Text (STT), and Natural Language Processing (NLP) capabilities.

## Components

### TTS Service
- Text-to-Speech conversion using TTS library
- Supports gRPC and REST API endpoints
- Uses the LJSpeech model for high-quality speech synthesis

### STT Service
- Speech-to-Text conversion
- Converts audio input to text

### NLP Service
- Natural Language Processing capabilities
- Processes and understands user queries

## Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)
- Sufficient disk space (~2-3GB)
- 4GB+ RAM recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RaviKiran752/VOICE_ASSIST.git
cd VOICE_ASSIST
```

2. Set up each service:
```bash
# TTS Service
cd tts_service
pip install -r requirements.txt

# STT Service
cd ../stt_service
pip install -r requirements.txt

# NLP Service
cd ../nlp_service
pip install -r requirements.txt
```

### Running the Services

1. TTS Service:
```bash
cd tts_service
python main.py
```
- FastAPI server runs on port 8000
- gRPC server runs on port 50053

## API Documentation

### TTS Service Endpoints

#### REST API
- POST `/synthesize`
  - Input: `text` (string)
  - Output: Audio data in WAV format

#### gRPC
- `SynthesizeSpeech`
  - Input: Text string
  - Output: Audio data

## License
MIT License

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. 